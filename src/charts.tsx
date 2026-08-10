import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { channelColor, theme, verdictColor } from "./theme";
import { HistogramBar, MetricHistogramData, QualityRow } from "./types";

const axisStyle = { fontSize: 10, fill: theme.textDim };
const axisLabelStyle = { fill: theme.textDim, fontSize: 11 };

// Drops the padding zeros toPrecision adds ("1.50" -> "1.5"), only ever after
// a decimal point: the trailing zeros of an integer like "150" are its value,
// not padding, and stripping them renders it as "15".
function trimTrailingZeros(text: string): string {
  return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
}

// Compact tick text for large magnitudes ("7.5M", "12k") so y-axis labels
// don't collide with wide tick numbers
function fmtTick(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e6) return `${trimTrailingZeros((v / 1e6).toPrecision(3))}M`;
  if (abs >= 1e3) return `${trimTrailingZeros((v / 1e3).toPrecision(3))}k`;
  return trimTrailingZeros(Number(v).toPrecision(3));
}
const tooltipStyle: React.CSSProperties = {
  background: theme.headerBg,
  border: `1px solid ${theme.cardBorder}`,
  borderRadius: 6,
  fontSize: 12,
  color: theme.text,
  padding: "6px 10px",
};

function HistTooltip({ active, payload, channels, allChannels }: any) {
  if (!active || !payload?.length) return null;
  const bar: HistogramBar = payload[0].payload;
  return (
    <div style={tooltipStyle}>
      <div>
        {bar.x0.toPrecision(3)} to {bar.x1.toPrecision(3)}
      </div>
      {channels.map((channel: string) => (
        <div key={channel} style={{ color: channelColor(allChannels, channel) }}>
          {allChannels.length > 1 ? `${channel}: ` : ""}
          {bar.counts[channel] ?? 0} episode(s)
        </div>
      ))}
    </div>
  );
}

export function MetricHistogram(props: {
  data: MetricHistogramData;
  /** Full channel list across the tab -- keeps series colors stable while filtering */
  allChannels: string[];
  /** When set, render only this channel's series */
  filterChannel?: string | null;
  /** X-axis label (the metric's short name; values are in the metric's own units) */
  xLabel: string;
  height?: number;
  warnThresholds?: Record<string, number>;
  onBarClick?: (bar: HistogramBar, channel: string) => void;
}) {
  const { data, allChannels, filterChannel, xLabel, height = 200, warnThresholds, onBarClick } = props;
  const { bars } = data;
  const channels =
    filterChannel && data.channels.includes(filterChannel) ? [filterChannel] : data.channels;
  const multi = allChannels.length > 1;

  // The x-axis is categorical (one band per bin), so anchor each channel's
  // warn line to the bin whose range contains its threshold.
  const warnLines: { channel: string; x: number }[] = [];
  for (const channel of channels) {
    const threshold = warnThresholds?.[channel];
    if (threshold === undefined || bars.length === 0) continue;
    const hit =
      bars.find((b) => threshold >= b.x0 && threshold < b.x1) ??
      (threshold >= bars[bars.length - 1].x1 ? bars[bars.length - 1] : bars[0]);
    warnLines.push({ channel, x: hit.x });
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      {/* top margin leaves headroom for the "warn" reference-line label;
          bottom margin for the x-axis label */}
      <BarChart data={bars} margin={{ top: 18, right: 8, bottom: 14, left: 4 }} barGap={0}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="x"
          tick={axisStyle}
          tickFormatter={(v: number) => Number(v).toPrecision(2)}
          interval="preserveStartEnd"
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{ value: xLabel, position: "bottom", offset: 0, ...axisLabelStyle }}
        />
        <YAxis
          tick={axisStyle}
          allowDecimals={false}
          tickLine={false}
          axisLine={false}
          width={46}
          label={{
            value: "# of episodes",
            angle: -90,
            position: "insideLeft",
            style: { textAnchor: "middle" },
            ...axisLabelStyle,
          }}
        />
        <Tooltip
          content={<HistTooltip channels={channels} allChannels={allChannels} />}
          cursor={{ fill: "#ffffff10" }}
        />
        {channels.map((channel) => (
          <Bar
            key={channel}
            name={channel}
            dataKey={(bar: HistogramBar) => bar.counts[channel] ?? 0}
            fill={multi ? channelColor(allChannels, channel) : theme.bar}
            radius={[2, 2, 0, 0]}
            isAnimationActive={false}
            onClick={(entry: any) => onBarClick?.((entry?.payload ?? entry) as HistogramBar, channel)}
            style={{ cursor: onBarClick ? "pointer" : "default" }}
          />
        ))}
        {warnLines.map(({ channel, x }) => (
          <ReferenceLine
            key={channel}
            x={x}
            stroke={multi ? channelColor(allChannels, channel) : theme.warn}
            strokeDasharray="5 4"
            label={
              multi ? undefined : { value: "warn", position: "top", fill: theme.warn, fontSize: 10 }
            }
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function VerdictBar(props: {
  counts: Record<string, number>;
  onBarClick?: (verdict: string) => void;
}) {
  const data = Object.entries(props.counts)
    .filter(([verdict, count]) => verdict !== "unknown" || count > 0)
    .map(([verdict, count]) => ({ verdict, count }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 14, left: 4 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="verdict"
          tick={axisStyle}
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{ value: "verdict", position: "bottom", offset: 0, ...axisLabelStyle }}
        />
        <YAxis
          tick={axisStyle}
          allowDecimals={false}
          tickLine={false}
          axisLine={false}
          width={46}
          label={{
            value: "# of episodes",
            angle: -90,
            position: "insideLeft",
            style: { textAnchor: "middle" },
            ...axisLabelStyle,
          }}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={{ fill: "#ffffff10" }}
          formatter={(value: number) => [value, "episodes"]}
        />
        <Bar
          dataKey="count"
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
          onClick={(entry: any) => props.onBarClick?.(entry.verdict)}
          style={{ cursor: props.onBarClick ? "pointer" : "default" }}
        >
          {data.map((entry) => (
            <Cell key={entry.verdict} fill={verdictColor[entry.verdict] ?? theme.bar} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ScatterTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const row: QualityRow = payload[0].payload;
  return (
    <div style={tooltipStyle}>
      <div style={{ fontWeight: 600 }}>{row.episode}</div>
      <div style={{ color: theme.textDim }}>
        iforest {row.iforest_score?.toFixed(3)} · knn {row.knn_dist?.toFixed(3)}
      </div>
      {row.is_outlier && <div style={{ color: theme.fail }}>flagged outlier</div>}
    </div>
  );
}

export function OutlierScatter(props: { rows: QualityRow[]; onOpen: (id: string) => void }) {
  const points = props.rows.filter((r) => r.iforest_score !== null && r.knn_dist !== null);

  return (
    <ResponsiveContainer width="100%" height={380}>
      <ScatterChart margin={{ top: 12, right: 16, bottom: 14, left: 0 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" />
        <XAxis
          dataKey="iforest_score"
          type="number"
          name="iforest_score"
          domain={["auto", "auto"]}
          tick={axisStyle}
          tickFormatter={fmtTick}
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{
            value: "iforest_score (higher = more anomalous)",
            position: "bottom",
            offset: 0,
            ...axisLabelStyle,
          }}
        />
        <YAxis
          dataKey="knn_dist"
          type="number"
          name="knn_dist"
          domain={["auto", "auto"]}
          tick={axisStyle}
          tickFormatter={fmtTick}
          tickLine={false}
          axisLine={false}
          label={{
            value: "knn_dist (higher = more isolated)",
            angle: -90,
            position: "insideLeft",
            ...axisLabelStyle,
          }}
        />
        <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: "3 3" }} />
        <Scatter
          data={points}
          isAnimationActive={false}
          onClick={(point: any) => point?.id && props.onOpen(point.id)}
          style={{ cursor: "pointer" }}
        >
          {points.map((row) => (
            <Cell
              key={row.id}
              fill={row.is_outlier ? theme.fail : theme.bar}
              stroke={row.is_outlier ? theme.fail : "none"}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
