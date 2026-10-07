"""Generic decode of MCAP channels into flat scalar time series.

Handles two channel kinds (see :mod:`.discovery`):

- ``telemetry`` (protobuf / ROS1 / ROS2 structured messages): decoded from
  the schema embedded in the MCAP file, the same way FiftyOne's own
  multimodal decoders do it (``fiftyone.multimodal.decoders.generic_protobuf``
  and ``generic_ros``): protobuf via the ``FileDescriptorSet`` and
  ``google.protobuf``, ROS via the message-definition text and ``rosbags``.
  Both ship with ``fiftyone[multimodal]``, so no ``mcap-*-support`` package
  is needed; those are still used as a fallback when installed. Every
  numeric field is then walked generically off the message's own field
  descriptors (protobuf) or fields (ROS) -- never off a hardcoded field
  name -- so this works on any structured schema.
- ``scalar_sidecar`` (flat JSON messages): decoded directly, keeping only
  numeric leaf values.

Messages with no numeric content (e.g. a string-only instructions message)
simply produce an empty record; callers skip those.
"""

import dataclasses
import json
import math
from functools import lru_cache

import numpy as np
from mcap.reader import make_reader

from .discovery import SCALAR_SIDECAR, TELEMETRY

# protobuf FieldDescriptor.cpp_type values that are numeric or boolean.
_PROTOBUF_NUMERIC_CPP_TYPES = frozenset(range(1, 8))
_PROTOBUF_MESSAGE_CPP_TYPE = 10

# Submessage types that hold a clock rather than a measurement, skipped by
# type name so no field-name matching is involved. See _iter_numeric_fields.
_PROTOBUF_TIME_TYPES = frozenset({"google.protobuf.Timestamp", "google.protobuf.Duration"})
_ROS_TIME_TYPES = frozenset({"Time", "Duration"})

# Guard against a pathologically deep schema; real kinematics sit one or two
# levels down (``Odometry.pose.position.x`` is the deepest common case).
_MAX_NESTING_DEPTH = 4
_TRANSFORM_TOLERANCE = 1e-6


def decode_channel(filepath, channel):
    """Decodes one channel into a time-ordered list of flat scalar records.

    Args:
        filepath: path to the MCAP file
        channel: a :class:`.discovery.ChannelInfo` with ``kind`` in
            ``{"telemetry", "scalar_sidecar"}``

    Returns:
        a list of ``(log_time_ns, publish_time_ns, {field: value})``
        3-tuples, in log-time order
    """
    if channel.kind == SCALAR_SIDECAR:
        return _decode_json_channel(filepath, channel.topic)
    if channel.kind == TELEMETRY:
        return _decode_structured_channel(filepath, channel.topic)

    raise ValueError(f"Cannot decode a '{channel.kind}' channel as scalar telemetry")


def first_numeric_fields(filepath, channel):
    """Returns the numeric fields on a channel's first message.

    This is the lightweight field-schema probe used by the operator form to
    offer individual ``channel -> field group`` motion signals. It decodes one
    message only; scoring still performs the full channel decode later.
    """
    if channel.kind == SCALAR_SIDECAR:
        return _first_json_fields(filepath, channel.topic)
    if channel.kind == TELEMETRY:
        return _first_structured_fields(filepath, channel.topic)
    return {}


def channels_log_times(filepath, topics):
    """Returns log timestamps for multiple channels in one body-free MCAP scan."""
    times_by_topic = {topic: [] for topic in topics}
    with open(filepath, "rb") as f:
        for _, channel, message in make_reader(f).iter_messages(
            topics=list(times_by_topic)
        ):
            times_by_topic[channel.topic].append(message.log_time)
    return times_by_topic


def _first_json_fields(filepath, topic):
    """The numeric fields on a channel's first message, or ``{}`` if it has none."""
    with open(filepath, "rb") as f:
        for _, _, message in make_reader(f).iter_messages(topics=[topic]):
            return _json_numeric_fields(message.data)
    return {}


def _iter_structured_messages(filepath, topic):
    """Yields ``(message, decoded)`` for one protobuf/ROS channel, in log-time order.

    Stops quietly if no decoder can handle the channel. MCAP files come from
    arbitrary producers, so a schema can be something the decoder refuses to
    build types for: a ROS 2 message definition with a lowercase constant
    name, a schema referencing a package it can't resolve, a truncated
    definition. That is a property of the file rather than a bug here, and it
    must not take down the episode -- one unreadable channel out of twenty
    should cost that channel and nothing else. Callers see the same empty
    result they'd get from a channel with no numeric content, and anything
    decoded before a mid-stream failure is kept.
    """
    with open(filepath, "rb") as f:
        reader = make_reader(f)
        try:
            # Read up front: ros2idl schemas reference sibling schemas in the
            # same file, and the summary can't be fetched mid-iteration.
            summary = reader.get_summary()
            schemas = tuple(summary.schemas.values()) if summary else ()
            decoders = {}
            for schema, channel, message in reader.iter_messages(topics=[topic]):
                key = (channel.message_encoding, schema.id if schema else None)
                if key not in decoders:
                    decoders[key] = _decoder_for(channel.message_encoding, schema, schemas)
                decode = decoders[key]
                if decode is None:
                    return
                yield message, decode(message.data)
        except Exception:
            return


def _first_structured_fields(filepath, topic):
    """As `_first_json_fields`, for protobuf/ROS channels (needs the decoders)."""
    for _, decoded in _iter_structured_messages(filepath, topic):
        return _walk_fields(decoded)
    return {}


def _json_numeric_fields(data):
    """Keeps only the top-level numeric entries of a JSON message.

    Sidecar channels routinely mix scalars with strings and nested objects
    (labels, ids, status blobs). Only flat numerics can become a time series,
    so everything else is dropped rather than coerced.
    """
    return {key: float(value) for key, value in json.loads(data).items() if _is_number(value)}


def _decode_json_channel(filepath, topic):
    records = []
    with open(filepath, "rb") as f:
        for _, _, message in make_reader(f).iter_messages(topics=[topic]):
            records.append((message.log_time, message.publish_time, _json_numeric_fields(message.data)))
    return records


def _decode_structured_channel(filepath, topic):
    return [
        (message.log_time, message.publish_time, _walk_fields(decoded))
        for message, decoded in _iter_structured_messages(filepath, topic)
    ]


def _decoder_for(message_encoding, schema, schemas):
    """A ``bytes -> message`` callable for one channel's schema, or ``None``.

    Tries the built-in schema-driven decoders first, then whichever
    ``mcap-*-support`` decoders happen to be installed.

    Args:
        message_encoding: the channel's ``message_encoding``
        schema: the channel's ``mcap.records.Schema``, or ``None``
        schemas: every schema in the file (for ros2idl cross-references)
    """
    if schema is None:
        return None

    builder = _BUILTIN_DECODERS.get(schema.encoding)
    if builder is not None:
        try:
            return builder(schema, schemas)
        except Exception:
            # Missing dependency or a schema it can't parse; let an
            # installed mcap-*-support decoder have a go.
            pass

    for factory in _optional_decoder_factories():
        decode = factory.decoder_for(message_encoding, schema)
        if decode is not None:
            return decode
    return None


def _protobuf_decoder(schema, schemas):
    return _protobuf_message_class(schema.name, bytes(schema.data)).FromString


@lru_cache(maxsize=64)
def _protobuf_message_class(schema_name, schema_data):
    """Builds a message class from an MCAP-embedded ``FileDescriptorSet``.

    Mirrors ``fiftyone.multimodal.decoders.foxglove._proto_util.message_class_from_schema``.
    Decoding with a real message class (rather than ``MessageToDict``) keeps
    the field descriptors that :func:`_iter_protobuf_fields` walks.
    """
    from google.protobuf import descriptor_pb2
    from google.protobuf.message_factory import GetMessages

    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(schema_data)
    return GetMessages(list(fds.file))[schema_name]


def _ros_decoder(schema, schemas):
    extra_schemas = ()
    if schema.encoding == "ros2idl":
        # Unlike ros2msg/ros1msg, ros2idl doesn't inline its dependencies.
        extra_schemas = tuple(
            sorted(
                (s.name, bytes(s.data))
                for s in schemas
                if s.encoding == "ros2idl" and s.id != schema.id
            )
        )

    stores, typename = _ros_typestores(
        schema.encoding, schema.name, bytes(schema.data), extra_schemas
    )
    method = "deserialize_ros1" if schema.encoding == "ros1msg" else "deserialize_cdr"
    candidates = [getattr(store, method) for store in stores]

    def decode(data):
        for i, deserialize in enumerate(candidates):
            try:
                message = deserialize(data, typename)
            except Exception:
                if i == len(candidates) - 1:
                    raise
                continue
            if i:
                # Remember which store works so later messages try it first.
                candidates.insert(0, candidates.pop(i))
            return message

    return decode


@lru_cache(maxsize=64)
def _ros_typestores(schema_encoding, schema_name, schema_data, extra_schemas):
    """Parses ROS schema text into candidate ``rosbags`` typestores.

    Mirrors ``fiftyone.multimodal.decoders.generic_ros._build_typestores``:
    the types parsed from the file alone, then the same types merged over the
    standard ROS distro store, so common ``std_msgs``/``geometry_msgs`` types
    resolve even when a schema doesn't embed them.

    Returns:
        ``(stores, normalized_type_name)``
    """
    from rosbags.typesys import (
        Stores,
        get_types_from_idl,
        get_types_from_msg,
        get_typestore,
    )
    from rosbags.typesys.msg import normalize_msgtype

    typename = normalize_msgtype(schema_name)

    types = {}
    for name, data in ((schema_name, schema_data),) + extra_schemas:
        text = data.decode("utf-8", errors="replace")
        try:
            if schema_encoding == "ros2idl":
                types.update(get_types_from_idl(text))
            else:
                types.update(get_types_from_msg(text, name))
        except Exception:
            continue

    if typename not in types:
        raise ValueError(f"{schema_name!r} not found in its {schema_encoding} schema")

    custom_store = get_typestore(Stores.EMPTY)
    custom_store.register(types)
    stores = [custom_store]

    distro = Stores.ROS1_NOETIC if schema_encoding == "ros1msg" else Stores.ROS2_HUMBLE
    distro_store = get_typestore(distro)
    try:
        distro_store.register(types)
        stores.append(distro_store)
    except Exception:
        # A custom type clashes with a same-named standard type.
        pass

    return tuple(stores), typename


# Keyed by the MCAP *schema* encoding, which is what says how to read the
# schema bytes (the channel's message encoding only names the wire format).
_BUILTIN_DECODERS = {
    "protobuf": _protobuf_decoder,
    "ros1msg": _ros_decoder,
    "ros2msg": _ros_decoder,
    "ros2idl": _ros_decoder,
}


@lru_cache(maxsize=1)
def _optional_decoder_factories():
    """Whichever ``mcap-*-support`` decoder factories are installed.

    These are a fallback only: the built-in decoders above cover the same
    encodings with packages ``fiftyone[multimodal]`` already installs, which
    matters on FiftyOne Enterprise where adding packages to the server image
    is a deployment change rather than a ``pip install``.
    """
    factories = []
    for module_name, class_name in (
        ("mcap_protobuf.decoder", "DecoderFactory"),
        ("mcap_ros2.decoder", "DecoderFactory"),
        ("mcap_ros1.decoder", "DecoderFactory"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError:
            continue
        factories.append(getattr(module, class_name)())
    return tuple(factories)


# Maps each structured message encoding to the modules that can decode it
# (built-in dependency first, optional mcap-*-support fallback second) and
# the pip package to suggest when neither imports. ROS 2 recordings write
# their wire format's name (`cdr`) rather than a `ros2` label, so both map to
# the ROS 2 stack. JSON is absent deliberately: it decodes with the stdlib.
_ENCODING_DECODER_PACKAGES = {
    "protobuf": (("google.protobuf.message_factory", "mcap_protobuf.decoder"), "protobuf"),
    "ros1": (("rosbags.typesys", "mcap_ros1.decoder"), "rosbags"),
    "ros2": (("rosbags.typesys", "mcap_ros2.decoder"), "rosbags"),
    "cdr": (("rosbags.typesys", "mcap_ros2.decoder"), "rosbags"),
}


def missing_decoder_package(message_encoding):
    """The pip package needed to decode ``message_encoding``, or ``None`` if decodable now.

    Lets callers distinguish "this channel decoded to no numeric fields" (a
    property of the data) from "nothing on this machine can decode this
    channel at all" (a property of the environment). The two must not be
    conflated: the first justifies excluding a channel, the second is the
    user's call plus a warning naming the package to install -- learned on a
    FiftyOne Enterprise pod where every protobuf channel silently read as
    "no numeric signal" and the whole Motion family shut itself off.

    Both suggested packages (``protobuf``, ``rosbags``) are part of
    ``fiftyone[multimodal]``, so this should only fire on a server that
    wasn't installed with the multimodal extra.

    Args:
        message_encoding: a channel's ``message_encoding`` string

    Returns:
        the pip package name to install, or ``None`` if the encoding is
        decodable right now (a decoder imports, or JSON which needs none)
    """
    entry = _ENCODING_DECODER_PACKAGES.get(message_encoding)
    if entry is None:
        return None

    module_names, package = entry
    for module_name in module_names:
        try:
            __import__(module_name)
        except ImportError:
            continue
        return None
    return package


def _walk_fields(message):
    """Flattens a decoded message to ``{field_name: scalar}``.

    Everything downstream (windowing, the motion metrics) works on flat named
    scalars, so vectors are exploded into indexed names here. The naming is
    what lets `windowing.field_groups` reassemble the original vector later:
    ``position_0..5`` regroups, and a 4x4 transform is emitted as named
    translation/euler components instead of 16 opaque matrix cells that would
    be differentiated as if they were coordinates.
    """
    out = {}
    for name, values in _iter_numeric_fields(message):
        if not values:
            continue
        if _is_homogeneous_transform(values):
            _flatten_transform(name, values, out)
        elif len(values) == 1:
            out[name] = values[0]
        else:
            out.update({f"{name}_{i}": v for i, v in enumerate(values)})
    return out


def _iter_numeric_fields(message, depth=0):
    """Yields ``(field_name, [float, ...])`` for every numeric field on a decoded message.

    Supports protobuf messages (via ``DESCRIPTOR.fields``), the ``rosbags``
    dataclasses the built-in ROS decoder produces, and the ``__slots__``
    classes the optional ``mcap-ros*-support`` decoders produce.

    Walks nested submessages, joining each level's field name with an
    underscore, because standard schemas put the kinematics one or two levels
    down rather than at the top: a ``foxglove.Odometry`` keeps its velocity in
    a ``Vector3`` submessage and a ROS ``sensor_msgs/msg/Imu`` keeps its
    angular rate in one. A top-level-only walk finds nothing at all on the
    first and only the flat covariance arrays on the second, so the channel
    that carries the actual motion looks empty while its metadata looks
    scorable.

    Two things are deliberately not walked. Repeated submessages are
    variable-length, so they can't form the fixed-width vector series every
    metric here needs. Timestamp and duration submessages are skipped by type
    name: a clock is a monotonic ramp, so differentiating it manufactures a
    smooth constant-velocity signal that has nothing to do with the robot.
    """
    if depth > _MAX_NESTING_DEPTH:
        return
    if hasattr(message, "DESCRIPTOR"):
        yield from _iter_protobuf_fields(message, depth)
    else:
        yield from _iter_ros_fields(message, depth)


def _iter_protobuf_fields(message, depth):
    for field in message.DESCRIPTOR.fields:
        value = getattr(message, field.name)
        if field.cpp_type in _PROTOBUF_NUMERIC_CPP_TYPES:
            yield field.name, [float(v) for v in value] if field.is_repeated else [float(value)]
        elif (
            field.cpp_type == _PROTOBUF_MESSAGE_CPP_TYPE
            and not field.is_repeated
            and field.message_type.full_name not in _PROTOBUF_TIME_TYPES
        ):
            for name, values in _iter_numeric_fields(value, depth + 1):
                yield f"{field.name}_{name}", values


def _iter_ros_fields(message, depth):
    for name in _ros_field_names(message):
        value = getattr(message, name)
        if _is_number(value):
            yield name, [float(value)]
        elif _is_numeric_sequence(value):
            yield name, [float(v) for v in value]
        elif _is_ros_message(value) and not _is_ros_time(value):
            for sub, values in _iter_numeric_fields(value, depth + 1):
                yield f"{name}_{sub}", values


def _ros_field_names(message):
    if dataclasses.is_dataclass(message):
        # rosbags carries the type name as a `__msgtype__` field; it's not data.
        return [f.name for f in dataclasses.fields(message) if f.name != "__msgtype__"]
    return getattr(message, "__slots__", ())


def _is_ros_message(value):
    return (
        dataclasses.is_dataclass(value) and not isinstance(value, type)
    ) or hasattr(value, "__slots__")


def _is_ros_time(value):
    # rosbags names the class `builtin_interfaces__msg__Time` and records
    # `builtin_interfaces/msg/Time` in `__msgtype__`; mcap-ros* names it `Time`.
    msgtype = getattr(value, "__msgtype__", type(value).__name__)
    return msgtype.rsplit("/", 1)[-1] in _ROS_TIME_TYPES


def _is_numeric_sequence(value):
    """True for a non-empty, flat sequence of numbers.

    rosbags decodes numeric ROS arrays to ``numpy`` arrays. Byte arrays
    (``uint8[]``/``int8[]``: image and point-cloud buffers) are excluded, as
    the ``mcap-ros*`` decoders already exclude them by returning ``bytes``.
    """
    if isinstance(value, np.ndarray):
        return (
            value.ndim == 1
            and value.size > 0
            and value.dtype.kind in "iuf"
            and value.dtype.itemsize > 1
        )
    return isinstance(value, (list, tuple)) and bool(value) and all(_is_number(v) for v in value)


def _is_number(value):
    """True for real numeric scalars. Bools are excluded despite being ints.

    A boolean channel (gripper open/closed, e-stop) is a state flag, not a
    measurement: differentiating it produces meaningless "speed" spikes at
    every toggle, and its two distinct values look like a pinned sensor to the
    clipping check.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_homogeneous_transform(values):
    """True iff a length-16 vector is a row-major 4x4 transform (last row [0, 0, 0, 1]).

    Detected by shape rather than by field name so it works on any producer.
    The bottom-row test is what distinguishes a pose from an arbitrary
    16-element array (a 16-DOF hand's joint vector, say) that must not be
    reinterpreted as a matrix.
    """
    if len(values) != 16:
        return False
    a, b, c, d = values[12:16]
    return (
        abs(a) < _TRANSFORM_TOLERANCE
        and abs(b) < _TRANSFORM_TOLERANCE
        and abs(c) < _TRANSFORM_TOLERANCE
        and abs(d - 1.0) < _TRANSFORM_TOLERANCE
    )


def _flatten_transform(name, values, out):
    """Emits translation (``_x/_y/_z``) and ZYX-euler orientation for a 4x4 transform.

    Six interpretable degrees of freedom instead of 16 matrix cells, so a
    pose channel differentiates into an actual velocity. Euler angles are
    accepted knowing they wrap and can gimbal-lock: the alternative
    (quaternions) has no meaningful scalar derivative either, and the
    smoothness metrics only need a locally-continuous signal, which euler
    gives everywhere except at the wrap.
    """
    out[f"{name}_x"] = values[3]
    out[f"{name}_y"] = values[7]
    out[f"{name}_z"] = values[11]
    out[f"{name}_yaw"] = math.atan2(values[4], values[0])
    out[f"{name}_pitch"] = math.atan2(-values[8], math.hypot(values[9], values[10]))
    out[f"{name}_roll"] = math.atan2(values[9], values[10])
