import React from "react";
import { MetricHistogram, OutlierScatter, VerdictBar } from "./charts";
import { fmt, theme, verdictColor } from "./theme";
import { Card, Chip, DataTable } from "./ui";
import {
  HistogramBar,
  METRIC_LABELS,
  MOTION_METRICS,
  MotionMetric,
  PanelData,
  QualityRow,
} from "./types";

export type ShowEpisodes = (ids: string[], description: string) => void;

// Matches numpy.histogram bin semantics: bins are half-open [x0, x1) except
// the last bin, which also includes its right edge.
function rowsInBin(
  rows: QualityRow[],
  metric: MotionMetric,
  bar: HistogramBar,
  isLast: boolean
): QualityRow[] {
  return rows.filter((r) => {
    const v = r[metric];
    if (v === null) return false;
    return v >= bar.x0 && (v < bar.x1 || (isLast && v <= bar.x1));
  });
}

// Conceptual explainers shown behind each chart's "i" icon. Directions must
// match engine/motion.py HIGHER_IS_WORSE (sparc/ldlj/psd_lf_hf: higher is
// smoother; jerk_rms: higher is worse).
const EXPLAINERS: Record<string, string> = {
  sparc:
    "Spectral Arc Length: motion smoothness measured from the shape of the speed profile's " +
    "frequency spectrum. Smooth, well-coordinated motion has a simple spectrum and a value " +
    "closer to 0; jerky, fragmented motion drags it more negative. Bars count episodes in the " +
    "current view; the dashed line marks the warn threshold. Click a bar to filter the samples " +
    "panel to the episodes in that range.",
  ldlj:
    "Log Dimensionless Jerk: total jerk (rate of change of acceleration) over the episode, " +
    "scaled to be comparable across motions of different speeds and durations, then " +
    "log-transformed. Values closer to 0 mean smoother motion; more negative means rougher.",
  jerk_rms:
    "Root-mean-square jerk: the raw average intensity of jerk (rate of change of acceleration) " +
    "in the motion. Higher values mean more abrupt, shaky movement. Unlike SPARC and LDLJ this " +
    "is not scale-invariant, so it is most meaningful when comparing similar tasks.",
  psd_lf_hf:
    "Low/high frequency power ratio: how much of the motion's energy lives in slow, deliberate " +
    "movement versus high-frequency roughness (tremor, vibration, sensor noise). Higher means " +
    "cleaner low-frequency-dominated motion; lower means more high-frequency energy.",
  ranking:
    "Episodes ranked worst-first. Overall is a weighted average of each metric's robust " +
    "z-score: how many robust standard deviations (median/MAD) worse than the dataset median " +
    "the episode is, so scores are comparable across metrics with different units. Flags " +
    "counts metrics at warn severity (z \u2265 2) or worse.",
  health_verdicts:
    "Sensor-health rollup per episode, computed from raw message timestamps and values: " +
    "dropouts, frame-rate stability, clock drift, cross-channel desync, and value clipping. " +
    "Each metric is z-scored against this dataset; an episode fails if any metric is \u2265 3 " +
    "robust standard deviations worse than the dataset median, and warns at \u2265 2. Click a " +
    "bar to filter the samples panel to the episodes with that verdict.",
  health_table:
    "Per-episode health verdicts. The worst-metric column names the health metric with the " +
    "highest z-score, i.e. the one that drove the verdict, so you know what to inspect first " +
    "when you open the episode.",
  outliers:
    "Each point is an episode, placed by two complementary outlier detectors fit on the " +
    "quality scalars. X: Isolation Forest score, how easily the episode is separated from the " +
    "rest (higher = more anomalous). Y: mean distance to its k nearest neighbors, how far it " +
    "sits from its most similar episodes. Top-right points are unusual by both measures; red " +
    "points crossed the warn threshold.",
};

function scoreColor(score: number | null): string {
  if (score === null) return theme.text;
  if (score >= 2) return theme.fail;
  if (score >= 1) return theme.warn;
  return theme.text;
}

export function FamilyNotScored(props: { family: string }) {
  return (
    <div style={{ padding: "48px 16px", textAlign: "center", color: theme.textDim }}>
      <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
        {props.family} wasn't scored in the last run
      </div>
      <div style={{ fontSize: 13 }}>
        Re-run <b>Compute episode quality</b> with the {props.family} family enabled to populate
        this tab.
      </div>
    </div>
  );
}

export function MotionTab(props: {
  data: PanelData;
  onOpen: (id: string) => void;
  onShow: ShowEpisodes;
}) {
  const { data, onShow } = props;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 12,
        }}
      >
        {MOTION_METRICS.map((metric) => {
          const bars = data.histograms[metric] ?? [];
          const threshold = data.warn_thresholds[metric];
          const arrow = data.smoother_direction[metric] === "left" ? "← smoother" : "smoother →";
          const base =
            threshold !== undefined
              ? `${arrow} · warn ${data.smoother_direction[metric] === "left" ? "≥" : "≤"} ${threshold.toPrecision(3)}`
              : arrow;
          return (
            <Card
              key={metric}
              title={METRIC_LABELS[metric].title}
              subtitle={`${base} · click a bar to filter`}
              info={EXPLAINERS[metric]}
            >
              <MetricHistogram
                bars={bars}
                warnThreshold={threshold}
                onBarClick={(bar) => {
                  const isLast = bars.length > 0 && bar.x1 === bars[bars.length - 1].x1;
                  const hits = rowsInBin(data.rows, metric, bar, isLast);
                  onShow(
                    hits.map((r) => r.id),
                    `${METRIC_LABELS[metric].short} in [${bar.x0.toPrecision(3)}, ${bar.x1.toPrecision(3)}]`
                  );
                }}
              />
            </Card>
          );
        })}
      </div>

      <Card
        title="Worst-first ranking"
        subtitle="Click a row to open the episode"
        info={EXPLAINERS.ranking}
      >
        <DataTable
          columns={[
            { key: "episode", label: "Episode" },
            { key: "overall_score", label: "Overall", align: "right" },
            { key: "n_flags", label: "Flags", align: "right" },
            ...MOTION_METRICS.map((m) => ({
              key: m,
              label: METRIC_LABELS[m].short,
              align: "right" as const,
            })),
          ]}
          rowKeys={data.rows.map((r) => r.id)}
          rows={data.rows.map((r) => ({
            episode: r.episode,
            overall_score: (
              <span style={{ color: scoreColor(r.overall_score), fontWeight: 600 }}>
                {fmt(r.overall_score)}
              </span>
            ),
            n_flags: r.n_flags,
            sparc: fmt(r.sparc),
            ldlj: fmt(r.ldlj),
            jerk_rms: fmt(r.jerk_rms),
            psd_lf_hf: fmt(r.psd_lf_hf),
          }))}
          onRowClick={props.onOpen}
        />
      </Card>
    </div>
  );
}

export function HealthTab(props: {
  data: PanelData;
  onOpen: (id: string) => void;
  onShow: ShowEpisodes;
}) {
  const { data, onShow } = props;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 12,
        }}
      >
        <Card
          title="Health verdicts"
          subtitle="fail: any metric z ≥ 3 · warn: any z ≥ 2 · click a bar to filter"
          info={EXPLAINERS.health_verdicts}
        >
          <VerdictBar
            counts={data.verdict_counts}
            onBarClick={(verdict) =>
              onShow(
                data.rows.filter((r) => r.health_verdict === verdict).map((r) => r.id),
                `health verdict '${verdict}'`
              )
            }
          />
        </Card>
      </div>

      <Card
        title="Per-episode verdicts"
        subtitle="Click a row to open the episode"
        info={EXPLAINERS.health_table}
      >
        <DataTable
          columns={[
            { key: "episode", label: "Episode" },
            { key: "verdict", label: "Verdict" },
            { key: "reason", label: "Worst metric" },
          ]}
          rowKeys={data.rows.map((r) => r.id)}
          rows={data.rows.map((r) => ({
            episode: r.episode,
            verdict: <Chip label={r.health_verdict} color={verdictColor[r.health_verdict]} />,
            reason: r.health_reason || "–",
          }))}
          onRowClick={props.onOpen}
        />
      </Card>
    </div>
  );
}

export function OutliersTab(props: { data: PanelData; onOpen: (id: string) => void }) {
  const outliers = props.data.rows.filter((r) => r.is_outlier).length;
  return (
    <Card
      title="Isolation-forest score vs kNN manifold distance"
      subtitle={`${outliers} episode(s) cleared the outlier warn threshold (red) · click a point to open it`}
      info={EXPLAINERS.outliers}
    >
      <OutlierScatter rows={props.data.rows} onOpen={props.onOpen} />
    </Card>
  );
}

export type { QualityRow };
