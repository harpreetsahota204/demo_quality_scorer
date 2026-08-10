"""Generic decode of MCAP channels into flat scalar time series.

Handles two channel kinds (see :mod:`.discovery`):

- ``telemetry`` (protobuf / ROS1 / ROS2 structured messages): every numeric
  field is walked generically off the message's own field descriptors (for
  protobuf) or ``__slots__`` (for the dynamically-generated ROS1/ROS2
  message classes) -- never off a hardcoded field name -- so this works on
  any structured schema.
- ``scalar_sidecar`` (flat JSON messages): decoded directly, keeping only
  numeric leaf values.

Messages with no numeric content (e.g. a string-only instructions message)
simply produce an empty record; callers skip those.
"""

import json
import math

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


def has_numeric_signal(filepath, channel):
    """Peeks a channel's first message to check whether it carries any numeric field.

    A channel can be structurally valid telemetry (protobuf/ROS, not
    camera/scalar_sidecar) yet carry no numeric field at all -- e.g.
    ABC-130k's ``/instruction`` channel, a protobuf-encoded natural-language
    label with zero numeric fields. Used to decide Motion-family
    eligibility: only channels with *some* numeric signal can produce a
    speed profile at all. Reads only the first message, bounding the cost
    of checking every discovered channel to one message each, not a full
    decode.

    Args:
        filepath: path to the MCAP file
        channel: a :class:`.discovery.ChannelInfo`

    Returns:
        True iff the channel's first message decodes to at least one
        numeric field
    """
    if channel.kind == SCALAR_SIDECAR:
        return bool(_first_json_fields(filepath, channel.topic))
    if channel.kind == TELEMETRY:
        return bool(_first_structured_fields(filepath, channel.topic))
    return False


def _first_json_fields(filepath, topic):
    """The numeric fields on a channel's first message, or ``{}`` if it has none."""
    with open(filepath, "rb") as f:
        for _, _, message in make_reader(f).iter_messages(topics=[topic]):
            return _json_numeric_fields(message.data)
    return {}


def _iter_structured_messages(filepath, topic):
    """Yields ``(message, decoded)`` for one protobuf/ROS channel, in log-time order.

    Stops quietly if the installed decoder can't handle the channel. MCAP files
    come from arbitrary producers, so a schema can be something the decoder
    refuses to build types for: a ROS 2 message definition with a lowercase
    constant name, a schema referencing a package it can't resolve, a truncated
    definition. That is a property of the file rather than a bug here, and it
    must not take down the episode -- one unreadable channel out of twenty
    should cost that channel and nothing else. Callers see the same empty
    result they'd get from a channel with no numeric content, and anything
    decoded before a mid-stream failure is kept.
    """
    with open(filepath, "rb") as f:
        reader = make_reader(f, decoder_factories=_decoder_factories())
        try:
            for _, _, message, decoded in reader.iter_decoded_messages(topics=[topic]):
                yield message, decoded
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


def _decoder_factories():
    """Whichever MCAP message decoders are installed, in encoding-priority order.

    All three are optional dependencies: a user scoring protobuf episodes
    shouldn't be forced to install the ROS stacks. Missing ones are skipped
    here; the operator form reports them explicitly via
    :func:`missing_decoder_package`, and decode of an affected channel
    produces empty records rather than an import error.
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
    return factories


# Maps each structured message encoding to the decoder module that handles it
# and the pip package that provides that module. ROS 2 recordings write their
# wire format's name (`cdr`) rather than a `ros2` label, so both map to the
# ROS 2 stack. JSON is absent deliberately: it decodes with the stdlib.
_ENCODING_DECODER_PACKAGES = {
    "protobuf": ("mcap_protobuf.decoder", "mcap-protobuf-support"),
    "ros1": ("mcap_ros1.decoder", "mcap-ros1-support"),
    "ros2": ("mcap_ros2.decoder", "mcap-ros2-support"),
    "cdr": ("mcap_ros2.decoder", "mcap-ros2-support"),
}


def missing_decoder_package(message_encoding):
    """The pip package needed to decode ``message_encoding``, or ``None`` if decodable now.

    Lets callers distinguish "this channel decoded to no numeric fields" (a
    property of the data) from "nothing on this machine can decode this
    channel at all" (a property of the environment). The two must not be
    conflated: the first justifies excluding a channel, the second is the
    user's call plus a warning naming the package to install -- learned on a
    FiftyOne Enterprise pod without ``mcap-protobuf-support``, where every
    protobuf channel silently read as "no numeric signal" and the whole
    Motion family shut itself off.

    Args:
        message_encoding: a channel's ``message_encoding`` string

    Returns:
        the pip package name to install, or ``None`` if the encoding is
        decodable right now (decoder installed, or JSON which needs none)
    """
    entry = _ENCODING_DECODER_PACKAGES.get(message_encoding)
    if entry is None:
        return None

    module_name, package = entry
    try:
        __import__(module_name)
    except ImportError:
        return package
    return None


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

    Supports both protobuf messages (via ``DESCRIPTOR.fields``) and the
    dynamically-generated ROS1/ROS2 message classes (via ``__slots__``).

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
        yield from _iter_slot_fields(message, depth)


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


def _iter_slot_fields(message, depth):
    for name in getattr(message, "__slots__", ()):
        value = getattr(message, name)
        if _is_number(value):
            yield name, [float(value)]
        elif isinstance(value, (list, tuple)) and value and all(_is_number(v) for v in value):
            yield name, [float(v) for v in value]
        elif hasattr(value, "__slots__") and type(value).__name__ not in _ROS_TIME_TYPES:
            for sub, values in _iter_numeric_fields(value, depth + 1):
                yield f"{name}_{sub}", values


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
    return abs(a) < 1e-6 and abs(b) < 1e-6 and abs(c) < 1e-6 and abs(d - 1.0) < 1e-6


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
