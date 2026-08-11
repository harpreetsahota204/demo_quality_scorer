import React from "react";
import { MetricHistogram, OutlierScatter, VerdictBar } from "./charts";
import { fmt, signalColor, theme, verdictColor } from "./theme";
import { Card, Chip, DataTable } from "./ui";
import {
  HistogramBar,
  METRIC_LABELS,
  MOTION_METRICS,
  MotionMetric,
  PanelData,
  QualityRow,
} from "./types";

type ShowEpisodes = (ids: string[], description: string) => void;

const OVERALL_WARN_SCORE = 1;
const OVERALL_FAIL_SCORE = 2;

// Matches numpy.histogram bin semantics: bins are half-open [x0, x1) except
// the last bin, which also includes its right edge. Values are looked up on
// the clicked signal's series.
function rowsInBin(
  rows: QualityRow[],
  metric: MotionMetric,
  signal: string,
  bar: HistogramBar,
  isLast: boolean
): QualityRow[] {
  return rows.filter((r) => {
    const v = r.by_signal?.[signal]?.[metric];
    if (v === null || v === undefined) return false;
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
    "current view, one colored series per selected signal (each signal is normalized against " +
    "its own dataset-wide stats); dashed lines mark warn thresholds. Click a bar to filter the " +
    "samples panel to the episodes in that range on that signal.",
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
    "z-score: how many robust standard deviations worse than the dataset median the episode " +
    "is, measured with a spread fit from the metric's own bad tail, so scores are comparable " +
    "across metrics with different units and skew. When " +
    "several signals are scored, each metric shows its worst signal's value (worst-of, not " +
    "averaged \u2014 a smooth arm never masks a jerky one); hover a value for the per-signal " +
    "breakdown. Isolating a signal in the legend re-ranks the table by that signal's " +
    "motion-only score and shows its values.",
  health_verdicts:
    "Sensor-health rollup per episode, computed from raw message timestamps and values: " +
    "dropouts, frame-rate stability, clock drift, cross-channel desync, and value clipping. " +
    "Each metric is z-scored against this dataset. Click a bar to filter the samples panel to " +
    "the episodes with that verdict.",
  health_table:
    "Per-episode health verdicts. The worst-metric column names the health metric with the " +
    "highest z-score, i.e. the one that drove the verdict, so you know what to inspect first " +
    "when you open the episode.",
  outliers:
    "Each point is an episode, placed by two complementary outlier detectors fit on the " +
    "quality scalars. X: Isolation Forest score, how easily the episode is separated from the " +
    "rest (higher = more anomalous). Y: mean distance to its k nearest neighbors, how far it " +
    "sits from its most similar episodes. Top-right points are unusual by both measures; red " +
    "points crossed the warn threshold. Outlier scores stay separate from Overall because " +
    "unusual episodes can be exceptionally clean rather than bad.",
};

// Ranking-table metric cell: worst signal's value, with a hover title
// breaking the value down per signal when several were scored
function metricCell(row: QualityRow, metric: MotionMetric): React.ReactNode {
  const signals = Object.keys(row.by_signal ?? {});
  if (signals.length < 2) return fmt(row[metric]);

  const worst = row.worst_signal?.[metric];
  const breakdown = signals
    .map((signal) => {
      const v = row.by_signal[signal]?.[metric];
      const marker = signal === worst ? " (worst, shown)" : "";
      return `${signal}: ${v === null || v === undefined ? "–" : v.toFixed(3)}${marker}`;
    })
    .join("\n");
  return <span title={breakdown}>{fmt(row[metric])}</span>;
}

// Thresholds are deliberately lower than the engine's per-metric warn/fail
// z-scores: this colors a weighted *average* of z-scores, which an episode
// reaches only by being broadly bad rather than by failing one metric.
function scoreColor(score: number | null): string {
  if (score === null) return theme.text;
  if (score >= OVERALL_FAIL_SCORE) return theme.fail;
  if (score >= OVERALL_WARN_SCORE) return theme.warn;
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

// One shared legend for every histogram on the tab. Clicking a signal
// isolates its series across all plots; clicking it again shows everything.
function SignalLegendBar(props: {
  signals: string[];
  active: string | null;
  onPick: (signal: string | null) => void;
}) {
  const { signals, active, onPick } = props;
  if (signals.length < 2) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
      {signals.map((signal) => {
        const isActive = active === signal;
        const dimmed = active !== null && !isActive;
        const color = signalColor(signals, signal);
        return (
          <button
            key={signal}
            onClick={() => onPick(isActive ? null : signal)}
            title={isActive ? "Show all signals" : `Show only ${signal}`}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 5,
              background: isActive ? `${color}22` : "transparent",
              border: `1px solid ${isActive ? color : theme.cardBorder}`,
              borderRadius: 12,
              padding: "2px 9px",
              fontSize: 11,
              color: dimmed ? theme.textDim : theme.text,
              opacity: dimmed ? 0.55 : 1,
              cursor: "pointer",
              transition: "opacity 120ms, border-color 120ms, background 120ms",
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: 2,
                background: color,
              }}
            />
            {signal}
          </button>
        );
      })}
    </div>
  );
}

// Expand/collapse toggle for one histogram card. Expanding swaps the 2x2
// grid for a single full-width card of about the same total height -- not
// a modal, just a re-layout within the tab's existing real estate.
function ExpandToggle(props: { expanded: boolean; onClick: () => void }) {
  return (
    <button
      onClick={props.onClick}
      title={props.expanded ? "Back to grid" : "Expand this chart"}
      style={{
        background: "none",
        border: `1px solid ${theme.cardBorder}`,
        borderRadius: 4,
        color: theme.textDim,
        width: 22,
        height: 22,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        padding: 0,
      }}
      onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = theme.text)}
      onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = theme.textDim)}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
        {props.expanded ? (
          // collapse: arrows pointing inward
          <>
            <polyline points="4 14 10 14 10 20" />
            <polyline points="20 10 14 10 14 4" />
          </>
        ) : (
          // expand: arrows pointing outward
          <>
            <polyline points="15 3 21 3 21 9" />
            <polyline points="9 21 3 21 3 15" />
          </>
        )}
      </svg>
    </button>
  );
}

export function MotionTab(props: {
  data: PanelData;
  onOpen: (id: string) => void;
  onShow: ShowEpisodes;
}) {
  const { data, onShow } = props;
  const [activeSignal, setActiveSignal] = React.useState<string | null>(null);
  const [expanded, setExpanded] = React.useState<MotionMetric | null>(null);
  const shownMetrics = expanded ? [expanded] : [...MOTION_METRICS];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {data.signals.length > 1 && (
        <div style={{ fontSize: 11, color: theme.textDim, marginBottom: -4 }}>
          {activeSignal
            ? `showing only ${activeSignal} · click its chip again to show all signals`
            : "click a signal chip to isolate it"}
        </div>
      )}
      <SignalLegendBar
        signals={data.signals}
        active={activeSignal}
        onPick={setActiveSignal}
      />
      <div style={{ fontSize: 11, color: theme.textDim }}>
        click any bar to filter the samples panel · dashed lines = warn thresholds
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: expanded ? "1fr" : "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 12,
        }}
      >
        {shownMetrics.map((metric) => {
          const hist = data.histograms[metric] ?? { signals: [], bars: [] };
          const thresholds = data.warn_thresholds[metric];
          const arrow = data.smoother_direction[metric] === "left" ? "← smoother" : "smoother →";
          const single = hist.signals.length === 1 ? thresholds?.[hist.signals[0]] : undefined;
          // Shared cues (bar-click filtering, dashed warn lines) live once in
          // the legend hint; the subtitle stays per-metric only
          const subtitle =
            single !== undefined
              ? `${arrow} · warn ${data.smoother_direction[metric] === "left" ? "≥" : "≤"} ${single.toPrecision(3)}`
              : arrow;
          return (
            <Card
              key={metric}
              title={METRIC_LABELS[metric].title}
              subtitle={hist.signals.length > 0 ? subtitle : undefined}
              info={EXPLAINERS[metric]}
              action={
                <ExpandToggle
                  expanded={expanded === metric}
                  onClick={() => setExpanded(expanded === metric ? null : metric)}
                />
              }
            >
              {hist.signals.length === 0 ? (
                <div
                  style={{
                    height: 180,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: theme.textDim,
                    fontSize: 12,
                  }}
                >
                  Not computed in the last run (deselected or no data)
                </div>
              ) : (
                <MetricHistogram
                  data={hist}
                  allSignals={data.signals}
                  filterSignal={activeSignal}
                  xLabel={METRIC_LABELS[metric].short}
                  height={expanded ? 480 : 200}
                  warnThresholds={thresholds}
                  onBarClick={(bar, signal) => {
                    const bars = hist.bars;
                    const isLast = bars.length > 0 && bar.x1 === bars[bars.length - 1].x1;
                    const hits = rowsInBin(data.rows, metric, signal, bar, isLast);
                    const signalNote = hist.signals.length > 1 ? ` on ${signal}` : "";
                    onShow(
                      hits.map((r) => r.id),
                      `${METRIC_LABELS[metric].short}${signalNote} in [${bar.x0.toPrecision(3)}, ${bar.x1.toPrecision(3)}]`
                    );
                  }}
                />
              )}
            </Card>
          );
        })}
      </div>

      <RankingTable
        rows={data.rows}
        activeSignal={activeSignal}
        warnZ={data.warn_z}
        onOpen={props.onOpen}
      />
    </div>
  );
}

// Worst-first ranking. With a signal isolated in the legend, both the
// values and the sort key switch to that signal: metric cells show the
// signal's own values and rows re-rank by its motion-only score (health/
// outlier metrics aren't per-signal, so they can't contribute there).
function RankingTable(props: {
  rows: QualityRow[];
  activeSignal: string | null;
  warnZ: number;
  onOpen: (id: string) => void;
}) {
  const { rows, activeSignal, warnZ, onOpen } = props;

  const signalScore = (r: QualityRow) =>
    activeSignal ? r.signal_scores?.[activeSignal]?.score ?? null : r.overall_score;
  const signalFlags = (r: QualityRow) =>
    activeSignal ? r.signal_scores?.[activeSignal]?.n_flags ?? 0 : r.n_flags;

  // Backend rows are pre-sorted by overall_score; re-rank client-side when
  // a signal is isolated (episodes without that signal sink to the bottom)
  const sorted = activeSignal
    ? [...rows].sort((a, b) => (signalScore(b) ?? -Infinity) - (signalScore(a) ?? -Infinity))
    : rows;

  return (
    <Card
      title="Worst-first ranking"
      subtitle={
        activeSignal
          ? `ranked by ${activeSignal} (motion metrics only) · click a row to open the episode`
          : "Click a row to open the episode"
      }
      info={`${EXPLAINERS.ranking} Flags count metrics at warn severity (z \u2265 ${warnZ}) or worse.`}
    >
      <DataTable
        columns={[
          { key: "episode", label: "Episode" },
          { key: "overall_score", label: activeSignal ? "Motion score" : "Overall", align: "right" },
          { key: "n_flags", label: "Flags", align: "right" },
          ...MOTION_METRICS.map((m) => ({
            key: m,
            label: METRIC_LABELS[m].short,
            align: "right" as const,
          })),
        ]}
        rowKeys={sorted.map((r) => r.id)}
        rows={sorted.map((r) => ({
          episode: r.episode,
          overall_score: (
            <span style={{ color: scoreColor(signalScore(r)), fontWeight: 600 }}>
              {fmt(signalScore(r))}
            </span>
          ),
          n_flags: signalFlags(r),
          ...Object.fromEntries(
            MOTION_METRICS.map((m) => [
              m,
              activeSignal ? fmt(r.by_signal?.[activeSignal]?.[m]) : metricCell(r, m),
            ])
          ),
        }))}
        onRowClick={onOpen}
      />
    </Card>
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
          subtitle={`fail: any metric z \u2265 ${data.fail_z} · warn: any z \u2265 ${data.warn_z} · click a bar to filter`}
          info={`${EXPLAINERS.health_verdicts} An episode fails if any metric is \u2265 ${data.fail_z} robust standard deviations worse than the dataset median, and warns at \u2265 ${data.warn_z}.`}
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
