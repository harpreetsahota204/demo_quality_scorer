export interface QualityRow {
  id: string;
  episode: string;
  overall_score: number | null;
  n_flags: number;
  sparc: number | null;
  ldlj: number | null;
  jerk_rms: number | null;
  psd_lf_hf: number | null;
  /** Per-channel motion metric values: {channel: {metric: value}} */
  by_channel: Record<string, Record<string, number | null>>;
  /** Per-channel motion-only score/flags (same z aggregation as overall_score) */
  channel_scores: Record<string, { score: number; n_flags: number }>;
  /** Which channel's z-score drove each top-level motion value */
  worst_channel: Record<string, string>;
  iforest_score: number | null;
  knn_dist: number | null;
  is_outlier: boolean;
  health_verdict: "pass" | "warn" | "fail" | "unknown";
  health_reason: string;
}

export interface HistogramBar {
  x: number;
  x0: number;
  x1: number;
  /** Episode count per channel for this bin (shared bin grid) */
  counts: Record<string, number>;
}

export interface MetricHistogramData {
  channels: string[];
  bars: HistogramBar[];
}

export interface PanelData {
  scored: boolean;
  rows: QualityRow[];
  channels: string[];
  histograms: Record<string, MetricHistogramData>;
  /** Per metric, per channel: warn threshold in the channel's own units */
  warn_thresholds: Record<string, Record<string, number>>;
  verdict_counts: Record<string, number>;
  smoother_direction: Record<string, "left" | "right">;
  motion_scored: boolean;
  health_scored: boolean;
  outliers_scored: boolean;
  config_version_mismatch: boolean;
}

export const MOTION_METRICS = ["sparc", "ldlj", "jerk_rms", "psd_lf_hf"] as const;
export type MotionMetric = (typeof MOTION_METRICS)[number];

// Human-readable chart titles and short column headers per motion metric
export const METRIC_LABELS: Record<MotionMetric, { title: string; short: string }> = {
  sparc: { title: "Smoothness (SPARC)", short: "SPARC" },
  ldlj: { title: "Normalized jerk (LDLJ)", short: "LDLJ" },
  jerk_rms: { title: "Jerk intensity (RMS)", short: "Jerk RMS" },
  psd_lf_hf: { title: "Low/high frequency ratio (PSD)", short: "PSD LF/HF" },
};

export const PLUGIN = "demo-quality-scorer";
