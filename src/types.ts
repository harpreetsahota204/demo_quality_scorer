export interface QualityRow {
  id: string;
  episode: string;
  overall_score: number | null;
  n_flags: number;
  sparc: number | null;
  ldlj: number | null;
  jerk_rms: number | null;
  psd_lf_hf: number | null;
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
  count: number;
}

export interface PanelData {
  scored: boolean;
  rows: QualityRow[];
  histograms: Record<string, HistogramBar[]>;
  warn_thresholds: Record<string, number>;
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
