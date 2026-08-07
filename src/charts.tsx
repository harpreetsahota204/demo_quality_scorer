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
import { theme, verdictColor } from "./theme";
import { HistogramBar, QualityRow } from "./types";

const axisStyle = { fontSize: 10, fill: theme.textDim };
const tooltipStyle: React.CSSProperties = {
  background: theme.headerBg,
  border: `1px solid ${theme.cardBorder}`,
  borderRadius: 6,
  fontSize: 12,
  color: theme.text,
  padding: "6px 10px",
};

function HistTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const bar: HistogramBar = payload[0].payload;
  return (
    <div style={tooltipStyle}>
      <div>
        {bar.x0.toPrecision(3)} to {bar.x1.toPrecision(3)}
      </div>
      <div style={{ color: theme.textDim }}>{bar.count} episode(s)</div>
    </div>
  );
}

export function MetricHistogram(props: {
  bars: HistogramBar[];
  warnThreshold?: number;
  onBarClick?: (bar: HistogramBar) => void;
}) {
  const { bars, warnThreshold, onBarClick } = props;

  // The x-axis is categorical (one band per bin), so anchor the warn line
  // to the bin whose range contains the threshold; the exact value is shown
  // in the card subtitle.
  let warnX: number | undefined;
  if (warnThreshold !== undefined && bars.length > 0) {
    const hit =
      bars.find((b) => warnThreshold >= b.x0 && warnThreshold < b.x1) ??
      (warnThreshold >= bars[bars.length - 1].x1 ? bars[bars.length - 1] : bars[0]);
    warnX = hit.x;
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      {/* top margin leaves headroom for the "warn" reference-line label */}
      <BarChart data={bars} margin={{ top: 18, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="x"
          tick={axisStyle}
          tickFormatter={(v: number) => Number(v).toPrecision(2)}
          interval="preserveStartEnd"
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
        />
        <YAxis tick={axisStyle} allowDecimals={false} tickLine={false} axisLine={false} />
        <Tooltip content={<HistTooltip />} cursor={{ fill: "#ffffff10" }} />
        <Bar
          dataKey="count"
          fill={theme.bar}
          radius={[2, 2, 0, 0]}
          isAnimationActive={false}
          onClick={(entry: any) => onBarClick?.(entry as HistogramBar)}
          style={{ cursor: onBarClick ? "pointer" : "default" }}
        />
        {warnX !== undefined && (
          <ReferenceLine
            x={warnX}
            stroke={theme.warn}
            strokeDasharray="5 4"
            label={{ value: "warn", position: "top", fill: theme.warn, fontSize: 10 }}
          />
        )}
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
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke={theme.cardBorder} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="verdict" tick={axisStyle} tickLine={false} axisLine={{ stroke: theme.cardBorder }} />
        <YAxis tick={axisStyle} allowDecimals={false} tickLine={false} axisLine={false} />
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
          tickLine={false}
          axisLine={{ stroke: theme.cardBorder }}
          label={{
            value: "iforest_score (higher = more anomalous)",
            position: "bottom",
            offset: 0,
            fill: theme.textDim,
            fontSize: 11,
          }}
        />
        <YAxis
          dataKey="knn_dist"
          type="number"
          name="knn_dist"
          domain={["auto", "auto"]}
          tick={axisStyle}
          tickLine={false}
          axisLine={false}
          label={{
            value: "knn_dist (higher = more isolated)",
            angle: -90,
            position: "insideLeft",
            fill: theme.textDim,
            fontSize: 11,
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
