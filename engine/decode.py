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


def _first_structured_fields(filepath, topic):
    """As `_first_json_fields`, for protobuf/ROS channels (needs the decoders)."""
    with open(filepath, "rb") as f:
        reader = make_reader(f, decoder_factories=_decoder_factories())
        for _, _, _, decoded in reader.iter_decoded_messages(topics=[topic]):
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
    records = []
    with open(filepath, "rb") as f:
        reader = make_reader(f, decoder_factories=_decoder_factories())
        for _, _, message, decoded in reader.iter_decoded_messages(topics=[topic]):
            records.append((message.log_time, message.publish_time, _walk_fields(decoded)))
    return records


def _decoder_factories():
    """Whichever MCAP message decoders are installed, in encoding-priority order.

    All three are optional dependencies: a user scoring protobuf episodes
    shouldn't be forced to install the ROS stacks. Missing ones are skipped,
    which surfaces later as "this channel has no numeric signal" rather than
    an import error at plugin load.
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


def _iter_numeric_fields(message):
    """Yields ``(field_name, [float, ...])`` for every numeric field on a decoded message.

    Supports both protobuf messages (via ``DESCRIPTOR.fields``) and the
    dynamically-generated ROS1/ROS2 message classes (via ``__slots__``).
    """
    if hasattr(message, "DESCRIPTOR"):
        for field in message.DESCRIPTOR.fields:
            if field.cpp_type not in _PROTOBUF_NUMERIC_CPP_TYPES:
                continue
            value = getattr(message, field.name)
            yield field.name, [float(v) for v in value] if field.is_repeated else [float(value)]
        return

    for name in getattr(message, "__slots__", ()):
        value = getattr(message, name)
        if _is_number(value):
            yield name, [float(value)]
        elif isinstance(value, (list, tuple)) and value and all(_is_number(v) for v in value):
            yield name, [float(v) for v in value]


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
