"""Generic MCAP channel discovery.

Classifies channels by ``message_encoding`` and schema shape only, never by
dataset-specific topic or schema names, so the same code path works on any
MCAP producer (ROS1, ROS2, protobuf, or flat-JSON telemetry sidecars).
"""

from dataclasses import dataclass

from mcap.reader import make_reader

# Well-known video schema names across common MCAP producers. Topic strings
# are not used for this check, because a topic can be misleadingly named.
_VIDEO_SCHEMAS = {
    "foxglove.CompressedVideo",
    "foxglove.RawImage",
    "foxglove.CompressedImage",
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}

# MCAP's well-known message encodings for schema-carrying messages. ROS 2
# writes `cdr` (its wire format) rather than a "ros2" label, so omitting it
# makes every ROS 2 recording read as an undecodable `other` channel; `ros2`
# is kept for producers that use the label anyway.
_STRUCTURED_ENCODINGS = {"protobuf", "ros1", "ros2", "cdr"}

CAMERA = "camera"
TELEMETRY = "telemetry"
SCALAR_SIDECAR = "scalar_sidecar"
OTHER = "other"


@dataclass(frozen=True)
class ChannelInfo:
    """Metadata for one MCAP channel, cheaply read from the file summary."""

    topic: str
    schema_name: str
    message_encoding: str
    n_messages: int
    kind: str


def discover_channels(filepath):
    """Enumerates and classifies the channels in an MCAP file.

    Uses only the MCAP summary section (schemas, channels, message counts),
    so this does not scan message bodies.

    Args:
        filepath: path to a local ``.mcap``/``.bag``/``.rrd`` file

    Returns:
        a list of :class:`ChannelInfo`, sorted by topic
    """
    with open(filepath, "rb") as f:
        summary = make_reader(f).get_summary()

    if summary is None:
        return []

    counts = summary.statistics.channel_message_counts if summary.statistics else {}

    channels = []
    for cid, channel in summary.channels.items():
        schema = summary.schemas.get(channel.schema_id)
        channels.append(
            ChannelInfo(
                topic=channel.topic,
                schema_name=schema.name if schema else "unknown",
                message_encoding=channel.message_encoding,
                n_messages=counts.get(cid, 0),
                kind=_classify(channel.message_encoding, schema),
            )
        )
    return sorted(channels, key=lambda c: c.topic)


def _classify(message_encoding, schema):
    """Assigns a channel the kind that decides what can be computed on it.

    Order matters: cameras are protobuf/ROS-encoded too, so they must be
    ruled out by schema before encoding is consulted, or every image channel
    would be handed to the scalar decoder. `OTHER` is the deliberate
    catch-all for encodings with no decoder here (cbor, flatbuffer, custom) --
    they stay visible in discovery, just not scorable.
    """
    schema_name = schema.name if schema else None
    if schema_name in _VIDEO_SCHEMAS:
        return CAMERA
    if message_encoding == "json":
        return SCALAR_SIDECAR
    if message_encoding in _STRUCTURED_ENCODINGS:
        return TELEMETRY
    return OTHER


def summarize_presence(channels):
    """Reduces a list of :class:`ChannelInfo` to boolean presence flags.

    These are the flags an operator/panel input can use to auto-configure
    itself; they fall directly out of the generic classification above, so
    they cover any MCAP producer rather than a fixed sensor list.

    Args:
        channels: a list of :class:`ChannelInfo`

    Returns:
        a dict of ``has_<kind>`` -> bool
    """
    kinds = {c.kind for c in channels}
    return {f"has_{kind}": kind in kinds for kind in (CAMERA, TELEMETRY, SCALAR_SIDECAR)}
