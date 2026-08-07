# Demo Quality Scorer — User Guide

A FiftyOne plugin that ranks multimodal MCAP episodes worst-first on
motion-smoothness, sensor-health, and outlier metrics, then deep-links
straight into the multimodal viewer so a human can confirm every flag by eye.

**Core rule: this is triage, not autofilter.** Nothing here should ever be
used as an automatic accept/reject gate. Structural errors (the wrong action
at a key moment) are invisible to every metric in this plugin — only a human
watching the episode can catch those. Scores exist to tell you *where to
look first*, not to make the decision for you.

## Contents

- [Setup](#setup)
- [Running the scorer](#running-the-scorer)
- [The Motion tab](#the-motion-tab)
- [The Health tab](#the-health-tab)
- [The Outliers tab](#the-outliers-tab)
- [Bulk triage](#bulk-triage)
- [Which metrics matter for which dataset type](#which-metrics-matter-for-which-dataset-type)
- [Reading `quality_intervals` and the timeline](#reading-quality_intervals-and-the-timeline)
- [As-built deviations from the original PRD](#as-built-deviations-from-the-original-prd)
- [Known limitations](#known-limitations)

## Setup

```bash
pip install "fiftyone[multimodal]>=1.19.0" mcap mcap-protobuf-support mcap-ros1-support mcap-ros2-support numpy scipy scikit-learn
```

`VFF_MULTIMODAL=1` must be set as an environment variable **before**
`fiftyone` is imported, in every process that touches a multimodal dataset
(scoring script, `fiftyone app launch`, notebook, etc.) — this gates the
App's native MCAP viewer, independent of the plugin.

The Episode Quality panel is a React component and ships prebuilt
(`dist/index.umd.js`) — nothing extra to install to *use* it. To modify the
frontend (`src/`), rebuild with `npm install && npm run build` (or
`npm run dev` for watch mode) and hard-refresh the browser.

## Running the scorer

1. Open (or create) a multimodal dataset (`media_type="multimodal"`,
   `.mcap`/`.bag`/`.rrd` samples).
2. Run **Compute episode quality** from the Operator Browser, or click
   "Compute episode quality" from inside the (empty) Episode Quality panel.
3. The form is organized by **metric family**, one tab per family (Motion /
   Sensor health / Outliers). Each tab holds that family's enable checkbox
   and settings; selections persist when you switch tabs, and the
   validation line below the tabs always reflects all three families.
   Inside each family every individual metric is a checkbox row with a
   short description:
   - **Motion smoothness** — auto-checked when the first sample has at
     least one telemetry channel carrying a numeric (speed-derivable)
     signal, with a multi-select of which channels carry motion. The
     picker starts empty — add channels from the dropdown (each picked
     topic leaves the dropdown, so you can't add one twice; e.g. add both
     arms of a bimanual rig). Each selected channel is scored
     independently and the worst channel per metric drives the episode's
     score (see [the worst-of rationale](#why-worst-of-across-channels)).
     The four metric checkboxes (SPARC / LDLJ / Jerk RMS / PSD ratio,
     all on by default) carry validity notes from the literature — SPARC
     is the most noise-robust and validated; LDLJ and jerk are
     noise-sensitive, so deselect them on noisy telemetry. Deselected
     metrics are simply absent from `quality.*` and the overall score
     renormalizes over what's left. Windowing (window length / overlap,
     defaults 2s / 50%), `idle_alpha` (default `0.05`), and
     `jerk_cutoff_hz` (default `10.0`, shown only while Jerk RMS is
     selected) live inside this section because windows only affect
     motion — health uses full-episode timestamps and outliers use
     episode scalars. Auto-unchecked with a visible reason if no numeric
     channel exists (e.g. a camera-only episode).
   - **Sensor health** — on by default; same empty-start multi-select of
     the discovered channels, plus five metric checkboxes (dropout, rate
     stability, clock drift, clipping, cross-channel desync).
   - **Outliers** — on by default, with a channel picker of its own:
     since the models consume per-channel motion features, you can
     restrict which channels' features feed them (leave empty for all
     selected motion channels; health features are episode-wide and
     always contribute). With Motion disabled the picker disappears and
     a notice explains the models fall back to health features only.

   A validation line above **Run** explains what will and won't be
   computed and why (e.g. "Motion: skipped, no telemetry channel carries a
   numeric signal") — this is advisory, never a hard block; a health-only
   run (Motion and Outliers both unchecked) is entirely legitimate.
4. For views over 50 samples the operator forces delegated (background)
   execution automatically; smaller views run immediately with a progress
   bar.
5. Open the **Episode Quality** panel (New panel → Episode Quality). It
   refreshes automatically whenever the current view changes, so filtering
   the grid re-ranks the panel to match.

Re-running the operator (e.g. after adding episodes, or with a different
family/channel selection) re-fits normalization from scratch across the
current view and overwrites every sample's `quality*` fields — this is
expected; scores are always relative to whatever population you last
scored together.

Formula changes bump `quality.config_version`. If a view mixes samples
scored under different `config_version`s (e.g. you scored half the dataset
before a metric-formula update and half after), the panel shows a warning
banner instead of silently blending the two into one ranking — re-run the
scorer across the whole view to make it comparable again.

## The Motion tab

Four histograms — Smoothness (SPARC), Normalized jerk (LDLJ), Jerk intensity
(RMS), and Low/high frequency ratio (PSD) — and a worst-first ranking table
sorted by `quality.overall_score`. When several motion channels were scored,
each histogram draws one colored series per channel (legend on top), and
dashed lines mark each channel's own warn threshold (z >= 2 against the
batch) in that channel's units. Every chart carries an "i" icon with a
plain-language explainer of what the metric means and how to read the plot.

The charts are interactive:

- **Click a histogram bar** to filter the samples panel to the episodes in
  that bin on that channel (a toast confirms what's showing; clear the view
  bar to reset). Since the panel re-ranks to the current view, the
  histograms then re-bin to the filtered subset.
- **Click any row** in the ranking table to jump straight into that
  episode's multimodal viewer, with a toast surfacing the timecode of its
  worst flagged interval. Metric cells show the worst channel's value —
  hover one for the per-channel breakdown.

### Why worst-of across channels

Each selected motion channel is scored independently and normalized against
its own per-(metric, channel) dataset-wide stats — different channels can
carry different units (rad/s vs m/s), so pooling them into one stats fit
would be meaningless. The per-channel values are then combined
**worst-of**: the channel with the highest (worst) z-score per metric
drives the episode's top-level value, `overall_score`, and `n_flags`.

This deliberately diverges from the published choice in the closest
comparable work: RINSE ([arXiv:2604.23000](https://arxiv.org/abs/2604.23000))
*averages* smoothness over both arms when filtering bimanual training data,
and that averaging is validated against downstream policy success. Filtering
and triage optimize different objectives, though. Averaging asks "how good
is this episode overall as training signal"; triage asks "is there anything
here a human should see." A jerky right arm averaged with a smooth left arm
looks fine — which is exactly the flag a reviewer needs surfaced. Movement
science reports smoothness per limb and defines no standard cross-limb
aggregate at all, so per-channel scoring with an explicit, documented
rollup is the honest middle ground. The per-channel values are kept on
every sample (`quality.motion_by_channel`) and in the panel's hover
breakdowns, so nothing is hidden by the rollup.

The panel opens onto the first tab the last run actually scored — a
health-only run opens onto Health, and a tab whose family wasn't scored
shows an explanation instead of empty plots.

With multiple motion channels scored, each top-level metric field below
(`quality.sparc`, etc.) holds the **worst channel's** value; per-channel
values live in `quality.motion_by_channel`.

| Field | What it measures | Direction |
|---|---|---|
| `sparc` | Spectral arc length of the motion's speed profile | **Less negative (closer to 0) = smoother.** Very negative = erratic, jerky motion |
| `ldlj` | Log dimensionless jerk of the speed profile | **Less negative = smoother.** Very negative = rough motion |
| `jerk_rms` | RMS jerk of the speed profile, after a zero-phase low-pass filter (`jerk_cutoff_hz`, default 10 Hz) applied before differentiating | **Lower = smoother.** High values = abrupt corrections, kickbacks, or actuator issues. Down-weighted (0.3x) in `overall_score` since it tends to co-vary with `sparc`/`ldlj`/`psd_lf_hf` — all four summarize the same speed profile's roughness |
| `psd_lf_hf` | Ratio of low-frequency to high-frequency power (Welch PSD) in the speed profile — **our own metric**, not a reproduction of Sojib & Begum's PSD data-quality metric (arXiv:2605.01544), whose "PSD" is raw summed DFT power on 3D end-effector *position*, ranked ascending. The name overlap is coincidental and the numbers aren't comparable to that paper's | **Higher = smoother** (energy concentrated at low frequencies). Low values mean a lot of high-frequency noise/vibration |
| `motion_by_channel` | List of per-channel motion metric docs (`channel` + every metric above), one entry per scored channel | Raw values in each channel's own units; this is what the multi-series histograms and hover breakdowns read |
| `motion_worst_channel` | Which channel's z-score drove each top-level motion value (e.g. `{sparc: "/right-arm-state"}`) | Provenance for the worst-of rollup — tells you which side to watch first |
| `idle_frac` | Fraction of the motion channel's windows below an idle-speed threshold, computed as `idle_alpha * median(speed)` **for that episode's own channel** (not a fixed absolute speed) — this is what makes the same threshold meaningful whether a channel reports rad/s, m/s, or a normalized unit | Not shown in this tab (visible via the sidebar/`dataset.values()`) and not scored into `overall_score`. High = episode is mostly stationary |
| `saturation_frac` | Fraction of samples pinned at their own observed min/max (heuristic — see [Known limitations](#known-limitations)) | Not shown in this tab; not scored into `overall_score` |
| `overall_score` | Weighted mean of every *enabled* metric's robust z-score (motion + health + outlier), oriented so higher is always worse. Disabling a family removes its metrics and renormalizes the remaining weights, rather than diluting the average with zeros | The sort key. `0` = typical for this batch; `+2` and up starts to be genuinely unusual; large negative scores are your *cleanest* episodes |
| `n_flags` | Count of metrics whose z-score cleared the warn threshold (z >= 2) | Higher = more independent signals think this episode is off |

SPARC and LDLJ come from movement-smoothness literature on human point-to-point
reaching, where healthy reaches land around SPARC ~= -1.6 and LDLJ ~= -6
(Balasubramanian et al., IEEE TBME 2012). Robot/vehicle telemetry won't
literally match those numbers — what matters is the *relative* ranking within
your own batch, which is exactly what `overall_score` gives you.

Each motion metric is also computed at p95 across an episode's windows
(`quality.sparc_p95`, etc., visible via the sidebar/`dataset.values()`, not
in the table) — the median tells you the episode's typical smoothness, the
p95 catches a single bad stretch that a median would wash out.

## The Health tab

A pass/warn/fail bar chart plus a table of the same verdicts per episode —
each non-pass verdict names the worst offending metric in a "Worst metric"
column, so a "fail" tells you what to look at, not just that something's
wrong. Clicking a verdict bar filters the samples panel to the episodes
with that verdict; clicking a table row opens the episode. Everything is computed generically from message timestamps — none of this depends on
motion, and the formulas apply the same way regardless of channel kind. The
health channel picker only offers `telemetry` and `scalar_sidecar` channels
today, though: camera channels would work fine mathematically, but the
engine's decode path doesn't support them yet (see [Known
limitations](#known-limitations)).

| Field | What it measures | Direction |
|---|---|---|
| `health.dropout` | Fraction of gaps between messages more than 3x the channel's expected inter-arrival time | Higher = more dropped/missing messages |
| `health.desync_ms` | Worst nearest-neighbor timestamp offset between any two scored channels, in ms | Higher = channels drifting further apart in time |
| `health.clock_drift_ppm` | Trend of `log_time` vs `publish_time` over the episode | Larger magnitude = one clock running faster/slower than the other (sign is discarded — only magnitude is scored) |
| `health.rate_cov` | Coefficient of variation of inter-arrival times | Higher = less steady sampling rate |
| `health.clipping_frac` | Fraction of samples pinned at their own observed min/max | Higher = more values sitting at what looks like a sensor/actuator limit (heuristic — see [Known limitations](#known-limitations)) |

A verdict is **fail** if any health metric's z-score hits the fail threshold
(z >= 3), **warn** if any hits the warn threshold (z >= 2) but none fail,
else **pass**. Verdicts need at least one completed scoring run on the
dataset (`quality_panel` reads the cached normalization stats from the
`demo_quality_scorer` run) — until then every episode shows `unknown`.

## The Outliers tab

A scatter of `iforest_score` (x) vs `knn_dist` (y), one point per episode —
points that cleared the outlier warn threshold (`is_outlier`) render red.
Click any point for the same open-episode-plus-notify deep-link as the
Motion/Health tables.

| Field | What it measures | Direction |
|---|---|---|
| `iforest_score` | Isolation-forest anomaly score over every computed metric (motion + health), fit across the current batch | Higher = more anomalous relative to the rest of the batch |
| `knn_dist` | Mean distance to the 5 nearest neighboring episodes in that same metric space | Higher = sits further from everything else — the one signal in the plugin that reacts to an episode's overall state, not just its motion smoothness or timestamps |
| `is_outlier` | `True` if either z-score above clears the warn threshold | A coarse "worth a second look" flag, not a verdict |

Both models need a real corpus to fit against (`n < 2` returns all-NaN) and
get more meaningful with more episodes — a handful of samples will produce
noisy, low-confidence outlier scores.

**Outlier channel selection is real.** Both models are fit on the batch's
already-computed quality scalars — each selected motion channel's
per-channel values plus the health metrics. The form's outlier channel
picker restricts which channels' motion feature columns feed the models
(leave it empty for all selected motion channels); health features are
episode-wide medians, so they always contribute. Normalization stats and
z-scores are never affected by this filter — it only changes what the
outlier models see. The **Outliers** toggle itself skips fitting both
models entirely and removes them from `overall_score`.

## Bulk triage

The two buttons at the bottom of the panel tag episodes `review` or
`exclude-candidate`. If you have samples selected in the grid, only those
get tagged; with nothing selected, the *entire current view* gets tagged.
The button labels state the scope explicitly ("Tag 3 selected" vs "Tag all
40 in view") — read them before clicking, and filter down to what you
actually mean to tag first. Tags are just FiftyOne
sample tags: nothing is deleted or hidden, `exclude-candidate` is a view you
build later (`dataset.match_tags("exclude-candidate", bool=False)`), not an
automatic filter.

## Which metrics matter for which dataset type

The engine is fully generic — it never hardcodes a topic or schema name —
but how much signal each tab gives you depends on what's in your channels:

| Domain | Motion tab | Health tab | Outliers tab |
|---|---|---|---|
| **Manipulation / teleop** (joint positions, end-effector pose, gripper state) | Strong signal. Catches jerky corrections, operator-fatigue jitter, kickback after contact | Strong for cross-arm/cross-sensor desync between telemetry channels (multi-camera desync isn't checked today — see [Known limitations](#known-limitations)) | Strong once you have enough episodes *of the same task* (see caveat below) |
| **Autonomous vehicles** (pose, velocity, steering telemetry) | Strong — harsh braking/steering corrections and sensor-glitch-induced motion noise both show up as high jerk / low SPARC | Strong across telemetry channels (LiDAR/IMU/GPS) — dropout and desync are common real failure modes; camera channels aren't checked yet (see [Known limitations](#known-limitations)) | Strong |
| **UAV / drone** (flight-controller position/velocity/attitude) | Strong — wind-gust artifacts and control-loop oscillation both surface as roughness | Strong | Strong |
| **Rover** (wheel odometry, pose) | Strong — stuck-and-slip and terrain jitter both surface as roughness | Strong | Strong |
| **Egocentric / wearable** (head/body pose, IMU) | Weaker discriminator — head turns are usually *intentional*, not a quality problem, so high jerk doesn't always mean "bad." Still useful for catching literal camera shake / sensor glitches | Strong — still just timestamps | Moderate |
| **Camera-only, no telemetry channels** | No signal — Motion auto-unchecks itself (no numeric channel to pick), so `quality.sparc` etc. won't be written | No signal today — rate/dropout only need message timestamps and would work just as well on a camera channel, but the health picker currently only offers `telemetry`/`scalar_sidecar` channels (see [Known limitations](#known-limitations)) | Falls back to whatever health metrics exist; weak without motion features |

**Important caveat on Outliers and cross-task comparison:** isolation forest
and kNN manifold distance compare episodes to *each other* in the current
batch. If your batch mixes many different tasks (as in a diversity-sampled
dataset), an episode can rank as an outlier simply because its task
inherently involves more or faster motion than the rest of the batch — not
because it's a bad demonstration. In validation against `Voxel51/ABC-130k`
(40 episodes across 40 different tasks), the two top-ranked "worst" episodes
were folding tasks (folding a t-shirt, folding a paper airplane) — plausibly
just more dynamic bimanual tasks, not necessarily poor teleoperation. Run the
scorer over episodes of the *same task* when you want a clean apples-to-apples
anomaly ranking, or treat cross-task outlier flags as "worth a look" rather
than "probably bad," per the triage philosophy at the top of this doc.

## Reading `quality_intervals` and the timeline

Every window whose metric clears the warn/fail threshold against the
*window-level* corpus statistics (not the episode-level ones used for
`overall_score` — window values are naturally noisier, so they're normalized
separately) becomes an interval: `{channel, metric, start, end, value,
severity}`, stored in `quality_intervals` (queryable data) **and** written as
a multimodal *temporal tag* on that sample, labeled
`"<channel>:<group>.<metric>:<severity>"` (e.g.
`"/left-arm-state:position.jerk_rms:fail"`) — this is what puts a colored,
clickable span on the multimodal player's timeline so you can see exactly
where a flag is and scrub straight to it.

It's easy to get this wrong: FiftyOne has a `TemporalDetections` label type
that looks like the obvious tool for "a labeled time interval," and it does
show up in the sidebar and as grid-thumbnail overlay text — but the
multimodal player's timeline builds its tracks *exclusively* from the
temporal-tags collection (a separate store from sample fields). A
`TemporalDetections` field is never read by the multimodal timeline code at
all, so it can look correctly populated everywhere except the one place you
want it. Temporal tags are the only mechanism confirmed to render on the
multimodal player's timeline. They live in FiftyOne's `fiftyone.core.tags`
module — undocumented/internal today, not the same programmatic-creation API
a person's manual shift-drag on the timeline uses to end up in the same
place, but it works.

Every temporal tag this plugin writes carries an internal `anchor` of
`"demo_quality_scorer"`, so re-running the operator can find-and-clear
exactly its own tags first (samples in the run's view only) without ever
touching a tag you drew by hand on the timeline. Re-running with a smaller
view or a different channel selection only re-tags the samples actually in
that run — everything else keeps whatever it was last scored with.

Expect many small intervals/tags on a long, dynamic episode — a metric like
`jerk_rms` naturally has bursts of roughness around every grasp/release or
sharp turn, so a handful of short flagged spans scattered through an episode
is normal, not a bug. What's worth investigating is a *sustained*
`fail`-severity tag, or an episode with a high `n_flags` across multiple
unrelated metrics at once.

## As-built deviations from the original PRD

The plugin's original design doc (the "Demo Quality Scorer" PRD) predates
some real decisions made while building and validating it. Rather than
bending the code back toward stale spec text, this section documents where
the as-built plugin diverges and why — the code is the source of truth:

- **Channel discovery** classifies channels by shape (`camera` /
  `telemetry` / `scalar_sidecar` / `other`, from `message_encoding` +
  schema structure), never by a fixed table of expected schema/topic names.
  A schema-name roles table was tried and rejected during validation: it
  classified zero channels on `Voxel51/ABC-130k` (a protobuf-encoded
  dataset), while shape-based discovery worked immediately and generalizes
  to any MCAP producer. Shape beats name.
- **Interval flags** are real multimodal *temporal tags*
  (`fiftyone.core.tags`), not a `TemporalDetections` label field. An
  earlier iteration used `TemporalDetections`, which does show up in the
  sidebar and grid-overlay text but is **never read by the multimodal
  player's timeline** (confirmed against FiftyOne's own frontend/backend
  source) — so it looked correct everywhere except the one place a
  "scrub to this flag" feature needs it.
- **`psd_lf_hf`** is this plugin's own Welch band-ratio-on-speed metric, an
  analog of — not a reproduction of — Sojib & Begum's PSD data-quality
  metric (arXiv:2605.01544). See [The Motion tab](#the-motion-tab).
- **`resolve_input`** is a family-first form (Motion / Sensor health /
  Outliers, each independently toggleable), replacing an earlier flat
  "pick every channel to score" list. The rescope keeps a health-only run
  (camera-only episodes, or anyone who just wants sensor-health checks)
  a first-class, clearly-labeled path instead of an implicit side effect
  of unchecking everything else.
- **Motion channels are multi-select, scored per channel, rolled up
  worst-of** (`config_version` 3). The PRD-era single channel picker made a
  bimanual rig half-blind: score one arm and a jerky other arm ranks clean.
  Channels are normalized per (metric, channel) — units differ across
  channels — and the worst z per metric drives the episode. See [Why
  worst-of across channels](#why-worst-of-across-channels) for why this
  intentionally diverges from RINSE's cross-arm averaging.
- **Idle threshold and jerk RMS** were tuned after initial validation: the
  idle-speed threshold is relative to each episode's own median speed
  (`idle_alpha * median(speed)`) rather than one fixed absolute value
  (which doesn't generalize across a channel's unit scale — rad/s vs m/s
  vs normalized), and `jerk_rms` low-pass filters the speed profile before
  differentiating (differentiation amplifies sensor noise by omega²,
  so an unfiltered RMS jerk on real telemetry is mostly noise). Both
  changes shift already-computed scores, tracked via
  `quality.config_version` (see [Running the
  scorer](#running-the-scorer)).

## Known limitations

- **No programmatic playhead seek.** Clicking a row/point opens the episode
  and shows a toast with the flagged timecode; you still have to scrub the
  timeline yourself.
- **`clipping_frac` is a heuristic**, not hardware-calibrated: it flags
  values pinned at the channel's own *observed* min/max, since there's no
  generic way to know a real actuator or sensor's true limits.
- **Remote/cloud `.mcap` files** are read with a plain local `open()`;
  efficient byte-range reads over cloud storage are not implemented.
- **Temporal tags are written via an undocumented, internal FiftyOne API**
  (`fiftyone.core.tags`), not the public SDK — it works in the version this
  plugin was built against but carries no stability guarantee, and a future
  FiftyOne release could change or remove it without notice.
- Re-running the operator **overwrites** `quality*` fields and the cached
  normalization stats for the run key `demo_quality_scorer`; there's no
  versioning across runs.
- **`config_version` mismatches are a warning, not a hard block.** The
  panel flags a view that mixes samples scored under different metric
  formulas, but it still renders the ranking/histograms below the
  banner — treat them as not-yet-comparable until you re-score.
- **The Sensor health channel picker only offers `telemetry` and
  `scalar_sidecar` channels**, not `camera`, even though rate/dropout only
  need message timestamps and would work fine on a camera channel too. The
  engine's decode path doesn't support camera channels yet (it would need a
  lightweight timestamp-only read instead of a full frame decode), so
  they're left out of the picker rather than offered and quietly no-op'd.
- **Outlier feature selection is channel-level only** — you can pick which
  channels feed the models, but not individual features (metrics); see
  [The Outliers tab](#the-outliers-tab).
