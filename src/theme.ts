/** Colors tuned to the FiftyOne App's dark theme. */
export const theme = {
  accent: "#ff6d04", // FiftyOne orange
  card: "#1f1f1f",
  cardBorder: "#333",
  text: "#e0e0e0",
  textDim: "#9e9e9e",
  bar: "#5b9bd5",
  warn: "#f9a825",
  fail: "#e04f4f",
  pass: "#4caf50",
  unknown: "#757575",
  rowHover: "#2a2a2a",
  headerBg: "#262626",
};

// Stable per-channel series colors (first matches theme.bar)
export const channelColors = ["#5b9bd5", "#e8a33d", "#7cb47c", "#b07cc6", "#c66a6a", "#5bbcb8"];

export const channelColor = (channels: string[], channel: string): string =>
  channelColors[Math.max(0, channels.indexOf(channel)) % channelColors.length];

export const verdictColor: Record<string, string> = {
  pass: theme.pass,
  warn: theme.warn,
  fail: theme.fail,
  unknown: theme.unknown,
};

export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined ? "–" : v.toFixed(digits);
