# Demo Quality Scorer

A FiftyOne plugin that ranks multimodal MCAP episodes worst-first on motion
smoothness, sensor health, and outlier metrics, then deep-links into the
multimodal viewer so you can confirm every flag by eye.

**Use it to decide where to look first. Do not use it as an automatic
accept/reject gate.** Structural errors, such as the wrong action at a key
moment, are invisible to every metric here. A demo can be perfectly smooth and
still be a failed demonstration. Only a human watching the episode catches
that, so the scores exist to order your queue, not to make the call.

## Contents

- [Install](#install)
- [Score a dataset](#score-a-dataset)
- [The Motion tab](#the-motion-tab)
- [The Health tab](#the-health-tab)
- [The Outliers tab](#the-outliers-tab)
- [How a score is computed](#how-a-score-is-computed)
- [Flagged intervals and the timeline](#flagged-intervals-and-the-timeline)
- [Tagging for review](#tagging-for-review)
- [What signal to expect from your data](#what-signal-to-expect-from-your-data)
- [Limits](#limits)

## Install

```bash
pip install "fiftyone[multimodal]>=1.19.0" mcap mcap-protobuf-support \
    mcap-ros1-support mcap-ros2-support numpy scipy scikit-learn
```

Set `VFF_MULTIMODAL=1` as an environment variable **before** `fiftyone` is
imported, in every process that touches a multimodal dataset (scoring script,
`fiftyone app launch`, notebook). This gates the App's native MCAP viewer and
is independent of the plugin.

The Episode Quality panel is a React component and ships prebuilt in
`dist/index.umd.js`, so there is nothing extra to install to use it. To change
the frontend in `src/`, run `npm install && npm run build` (or `npm run dev`
for watch mode) and hard-refresh the browser.

## Score a dataset

1. Open a multimodal dataset (`media_type="multimodal"`, with
   `.mcap`/`.bag`/`.rrd` samples).
2. Run **Compute episode quality** from the Operator Browser, or click the
   **Compute episode quality** button in the empty Episode Quality panel.
3. Fill in the form (below), then Run. Views over 50 samples are pushed to
   delegated background execution automatically; smaller views run immediately
   with a progress bar.
4. Open the panel: **New panel → Episode Quality**. It refreshes whenever the
   current view changes, so filtering the grid re-ranks the panel to match.

The form has one tab per metric family. Each tab holds that family's on/off
switch, its per-metric checkboxes, and a channel picker. Your selections
persist when you switch tabs, and the advisory line above **Run** always
reflects all three families.

**Motion smoothness.** Enabled when the first sample has at least one
telemetry channel carrying a numeric signal a speed profile can be derived
from. If none does (a camera-only episode, for example), the family switches
itself off and says why. Pick which channels carry motion; the dropdown drops
each topic you add so you can't add one twice, which makes selecting both arms
of a bimanual rig straightforward. Each channel is scored on its own and the
worst channel per metric drives the episode's score. The four metric
checkboxes (SPARC, LDLJ, Jerk RMS, PSD ratio) are all on by default and carry
short validity notes: SPARC is the most validated of the four and the least
affected by sensor noise, while LDLJ and jerk RMS are noise-sensitive, so
consider unchecking those two on noisy telemetry. Anything you uncheck is absent from `quality.*`
and the overall score renormalizes over what is left.

Windowing settings live in this tab because windows only affect motion (health
reads full-episode timestamps, outliers read episode scalars):

| Setting | Default | What it does |
|---|---|---|
| Window length | `2.0` s | Shorter windows localize flags more precisely and get noisier. Minimum 0.5 s |
| Window overlap | `0.5` | Fraction of overlap between consecutive windows, up to 0.9 |
| Idle threshold | `0.05` | Multiplier on each episode's own moving speed, below which a sample counts as idle |
| Jerk pre-filter cutoff | `10.0` Hz | Low-pass cutoff applied before differentiating for jerk RMS. Shown only while Jerk RMS is checked |

**Sensor health.** On by default, with its own channel picker and five metric
checkboxes (dropout, rate stability, clock drift, clipping, cross-channel
desync). Everything in this family comes from message timestamps and raw
values, so it needs no motion channel and works identically on any channel
kind the plugin can decode.

**Outliers.** On by default. The models consume per-channel motion features,
so you can restrict which channels' features reach them; leave the picker
empty to use every selected motion channel. Health features are episode-wide
and always contribute. With Motion off, the picker is replaced by a notice
that the models will run on health features alone.

The advisory line above **Run** spells out what will and won't be computed,
for example `Motion: skipped, no telemetry channel carries a numeric signal`.
It never blocks the run. A health-only run, with Motion and Outliers both off,
is a legitimate way to use this plugin.

Re-running rewrites every `quality*` field on the samples in the current view
and re-fits normalization from scratch across that view. Expect scores to move
when you re-run over a different population, because every score is relative
to the batch it was fit against.

## The Motion tab

Four histograms and a worst-first ranking table:

| Chart | Metric |
|---|---|
| Smoothness (SPARC) | `quality.sparc` |
| Normalized jerk (LDLJ) | `quality.ldlj` |
| Jerk intensity (RMS) | `quality.jerk_rms` |
| Low/high frequency ratio (PSD) | `quality.psd_lf_hf` |

Each chart carries an "i" icon with a plain-language explainer, an expand
button that swaps the 2x2 grid for one full-width plot, a dashed line at the
warn threshold in that channel's own units, and an arrow showing which
direction is smoother. A metric you didn't compute shows "Not computed in the
last run" instead of an empty plot.

Ways to drive it:

- **Click a histogram bar** to filter the samples panel to the episodes in
  that bin on that channel. A toast confirms what's showing; clear the view
  bar to reset. The panel re-ranks to the current view, so the histograms then
  re-bin to the filtered subset.
- **Click a channel chip** in the legend (shown when more than one channel was
  scored) to isolate that channel across all four plots. The ranking table
  follows: its values switch to that channel's own, the Overall column becomes
  **Motion score**, and rows re-rank by that channel's motion-only score.
  Click the chip again to show everything.
- **Click a table row** to open that episode in the multimodal viewer, with a
  toast naming the timecode of its worst flagged interval. Metric cells show
  the worst channel's value; hover one for the per-channel breakdown.

With several channels scored, each top-level field below holds the worst
channel's value.

| Field | What it measures | How to read it |
|---|---|---|
| `sparc` | Spectral arc length of the speed profile | Closer to 0 is smoother. Very negative means erratic, fragmented motion |
| `ldlj` | Log dimensionless jerk of the speed profile | Closer to 0 is smoother. Very negative means rough motion |
| `jerk_rms` | RMS jerk of the speed profile, low-pass filtered before differentiating | Lower is smoother. High values mean abrupt corrections, kickback, or actuator trouble. Weighted 0.3x in `overall_score` because it covaries with the other three |
| `psd_lf_hf` | Log ratio of low- to high-frequency power (Welch PSD) of the speed profile | Higher is smoother, meaning energy sits in slow deliberate movement. Low values mean high-frequency noise, tremor, or vibration |
| `sparc_worst`, `ldlj_worst`, `jerk_rms_worst`, `psd_lf_hf_worst` | The episode's single worst window for that metric, where the metric itself is the median over windows | Sidebar or `dataset.values()` only, not in the table. The median says what the episode was typically like; the worst window catches one bad stretch the median washes out |
| `idle_frac` | Fraction of windows below the idle-speed threshold | Not scored into `overall_score`. High means the episode was mostly stationary, which can be a legitimate pause as easily as a stall |
| `saturation_frac` | Fraction of samples pinned at their own observed min/max | Not scored into `overall_score`. A heuristic; see [Limits](#limits) |
| `motion_by_channel` | One embedded doc per scored channel, holding `channel` plus every metric above | Raw values in each channel's own units. This is what the multi-series histograms and hover breakdowns read |
| `motion_worst_channel` | Which channel drove each top-level value, e.g. `{sparc: "/right-arm-state"}` | Tells you which side to watch first |
| `overall_score` | Weighted mean of every enabled metric's z-score, oriented so higher is always worse | The sort key. `0` is typical for this batch, `+2` and up is genuinely unusual, and large negative scores are your cleanest episodes |
| `n_flags` | How many metrics cleared the warn threshold | Higher means more independent signals agree something is off |

`psd_lf_hf` is this plugin's own Welch band-ratio metric on the speed profile.
It is an analog of, not a reproduction of, the PSD data-quality metric in
Sojib & Begum ([arXiv:2605.01544](https://arxiv.org/abs/2605.01544)), which
sums raw DFT power on 3D end-effector position. The name overlap is
coincidental and the numbers are not comparable to that paper's.

SPARC and LDLJ come from movement-smoothness work on human point-to-point
reaching, where healthy reaches land near SPARC -1.6 and LDLJ -6
(Balasubramanian et al., IEEE TBME 2012). Robot and vehicle telemetry won't
match those numbers literally. What matters is the ranking within your own
batch, which is what `overall_score` gives you.

## The Health tab

A pass/warn/fail bar chart and a per-episode verdict table. Each non-pass
verdict names the metric with the highest z-score in a **Worst metric**
column, so a "fail" tells you what to inspect. Click a verdict bar to filter
the samples panel to the episodes with that verdict; click a table row to open
the episode.

| Field | What it measures | How to read it |
|---|---|---|
| `health.dropout` | Estimated fraction of expected messages lost to gaps longer than 3x the channel's expected inter-arrival time, weighted by how many messages each gap swallowed | Higher means more missing messages. A 100-message gap and a 4-message gap score differently |
| `health.desync_ms` | Best-case nearest-neighbor timestamp offset between the worst-aligned pair of scored channels, in milliseconds | Higher means two channels drifting apart in time. Single-digit ms is normal for most rigs; tens of ms between camera and proprioception misaligns image/action pairs |
| `health.clock_drift_ppm` | Magnitude of the `log_time` vs `publish_time` trend over the episode, in parts per million | Hardware crystals drift by tens of ppm. Hundreds or worse means a clock is misbehaving. Reported as unavailable, not 0.0, when there is only one clock to read, which is the common case: many writers set `publish_time` equal to `log_time`. A constant nonzero offset is two real clocks with no relative drift and does report 0.0 |
| `health.rate_cov` | MAD-based coefficient of variation of inter-arrival times | Near 0 is metronomic. Higher means jittery timing, often a loaded compute box or lossy transport |
| `health.clipping_frac` | Fraction of samples pinned at their own observed min/max, over the dimensions where that means anything | Higher means more values sitting at what looks like a sensor or actuator limit. A heuristic; see [Limits](#limits) |

Each metric is reduced to one episode-wide number, the median across whatever
contributed to it (channels, field groups, or channel pairs). Desync is the
exception, since it is a property of a channel pair rather than a channel: the
episode reports its worst pair, because a single badly skewed pair is enough to
misalign the data.

A verdict is **fail** if any health metric hits z >= 3, **warn** if any hits
z >= 2 without failing, otherwise **pass**. Verdicts need one completed scoring
run on the dataset, because they read the normalization stats cached under the
`demo_quality_scorer` run key. Until then every episode reads `unknown`, which
means unmeasured rather than healthy.

## The Outliers tab

A scatter of `iforest_score` (x) against `knn_dist` (y), one point per episode.
Points that cleared the warn threshold render red. Top-right points are
unusual by both measures. Click any point to open that episode.

| Field | What it measures | How to read it |
|---|---|---|
| `iforest_score` | Isolation-forest anomaly score over every computed metric, fit across the batch | Higher is more anomalous relative to the rest of the batch |
| `knn_dist` | Mean distance to the 5 nearest neighboring episodes in that same space | Higher means it sits further from everything else. This is the one signal here that reacts to an episode's overall character rather than its smoothness or its timestamps alone |
| `is_outlier` | True if either score clears z >= 2 | A coarse "worth a second look" flag |

Both models are fit on the batch's oriented z-scores rather than raw values,
so a `jerk_rms` in the thousands can't drown out a `dropout` in `[0, 1]` in the
Euclidean distance. The `_worst` tail columns are held out of the feature
matrix, since each is a percentile of the same per-window distribution its
median twin already summarizes. Both scores are weighted 0.5x in
`overall_score`, because they are functions of every other metric already in
the sum.

Both need a real corpus to fit against. Fewer than two episodes returns NaN,
and a handful of episodes produces noisy, low-confidence scores.

These models flag *unusual*, not *bad*. Your one exceptionally clean demo is
an outlier too, and so is the episode recorded in a different room. See the
[cross-task caveat](#what-signal-to-expect-from-your-data) before reading much
into a ranking over mixed tasks.

## How a score is computed

**Windows.** Motion metrics are computed on 2-second windows at 50% overlap,
on each channel's own timestamps, and windows need at least 5 samples to be
scored. Every window spans the full window length: several of these metrics
are duration-dependent (LDLJ's dimensionless jerk scales as roughly duration
to the fourth), so a short trailing window is not comparable to the full ones
it would be normalized against. An episode summarizes its windows twice, as a
median and as a worst window.

**Per-channel normalization.** Every metric is z-scored against the batch's
own distribution for that exact (metric, channel) pair, never pooled across
channels, because a gripper's units and a shoulder joint's aren't comparable.
Episode-level and window-level values are also fit separately: per-window
values spread out far more than medians-over-windows, and z-scoring one
against the other's stats would flag almost every above-average window.

**The scale.** Z-scores are median-based, so a few gross outliers don't
inflate the denominator. The scale is fit from the bad side of the median only,
as the semi-interquartile range on that side (`p75 - p50` where higher is
worse, `p50 - p25` where lower is worse). On a symmetric distribution that is
identical to an ordinary MAD. Real per-window telemetry is strongly skewed,
though, and a two-sided MAD averages in the narrow good side, which understates
how much room the bad side actually has.

Metrics that read exactly zero across most of a clean corpus (`dropout`,
`clock_drift_ppm`, `desync_ms`) have no spread to measure, so their scale is
set such that a typical member of the nonzero tail lands on the warn
threshold. Being in the tail at all earns a glance, and being further out earns
proportionally more. Z-scores are clipped at +/-10 as a backstop for a corpus
with no variation whatsoever.

**Worst-of across channels.** Per-channel z-scores are combined by taking the
worst: the highest z per metric drives the episode's top-level value,
`overall_score`, and `n_flags`. RINSE
([arXiv:2604.23000](https://arxiv.org/abs/2604.23000)) averages smoothness
across both arms when filtering bimanual training data, validated against
downstream policy success, and averaging is the right call for that job.
Filtering asks how good an episode is as training signal overall. Triage asks
whether anything here deserves a human's attention, and a jerky right arm
averaged against a smooth left arm looks fine. Movement science reports
smoothness per limb and defines no standard cross-limb aggregate, so
per-channel scoring with an explicit rollup is the honest middle ground. Every
per-channel value stays on the sample in `quality.motion_by_channel` and in the
panel's hover breakdowns, so the rollup hides nothing.

**Weights.** `overall_score` is a weighted mean over the metrics that were
actually computed. Disabling a family removes its metrics and renormalizes the
remaining weights rather than diluting the average with zeros. `jerk_rms` is
weighted 0.3x and the two outlier scores 0.5x; everything else is 1.0.
`idle_frac` and `saturation_frac` are reported for context and are not scored,
because neither is inherently bad at any level.

**`quality.config_version`.** Written on every sample, and bumped whenever a
formula change would move already-written scores. If a view mixes samples
carrying different values (you scored half the dataset, then updated the
plugin, then scored the rest), the panel shows a banner rather than blending
incomparable rankings. Re-run over the whole view to fix it.

## Flagged intervals and the timeline

A window whose metric clears the warn or fail threshold against the
window-level statistics becomes an interval, written two ways:

- `quality_intervals` on the sample, as `{channel, metric, start, end, value,
  severity}`, where `metric` is `<field group>.<metric>`. This is the
  queryable copy.
- A multimodal temporal tag labeled `<channel>:<group>.<metric>:<severity>`,
  for example `/left-arm-state:position.jerk_rms:fail`. This is what puts a
  colored, clickable span on the multimodal player's timeline.

Windows overlap, so one real event flags two or more consecutive windows;
touching flags of the same series are merged into a single span that reports
the worst of what it absorbed.

Temporal tags are the only mechanism confirmed to render on the multimodal
player's timeline. FiftyOne's `TemporalDetections` label type looks like the
obvious tool for a labeled time interval and does appear in the sidebar and as
grid-thumbnail overlay text, but the multimodal timeline builds its tracks
exclusively from the temporal-tags collection, a separate store from sample
fields. A `TemporalDetections` field can look correctly populated everywhere
except the one place you want it.

Every temporal tag this plugin writes carries the internal anchor
`demo_quality_scorer`, so a re-run clears exactly its own tags on the samples
in that run and never touches a tag you drew by hand on the timeline.
Re-running over a smaller view leaves every other sample's tags alone.

Expect many short spans on a long, dynamic episode. Jerk RMS has bursts of
roughness around every grasp, release, and sharp turn, and there is one
independent flag stream per (channel, field group, metric). Four channels with
four field groups each, scored on four metrics, is 64 streams, and each one
contributes its own worst few percent of windows. On `Voxel51/ABC-130k` that
works out to a median of about 60 flagged spans on an 87-second episode. What
deserves investigation is a sustained `fail` span, or an episode with a high
`n_flags` across several unrelated metrics at once, rather than a raw span
count.

## Tagging for review

Two buttons at the bottom of the panel tag episodes `review` or
`exclude-candidate`. The button labels state their scope before you click:
`Tag 3 selected: review` when you have samples selected in the grid, or
`Tag all 40 in view: review` when you don't. With nothing selected, the entire
current view gets tagged, so filter down to what you mean first.

These are ordinary FiftyOne sample tags. Nothing is deleted or hidden, and
`exclude-candidate` is a view you build later, not an automatic filter:

```python
keep = dataset.match_tags("exclude-candidate", bool=False)
```

## What signal to expect from your data

The engine never hardcodes a topic or schema name, so it runs on any MCAP
producer: protobuf, ROS 1, ROS 2 (`cdr`), and flat-JSON sidecars. Channels are
classified by encoding and schema shape, and numeric fields are found by
walking each message's own descriptors, including nested submessages, so a
`foxglove.Odometry`'s velocity and a ROS `sensor_msgs/msg/Imu`'s angular rate
are picked up without naming either.

Two things are skipped on purpose. Timestamp and duration submessages are
excluded by type, because a clock is a monotonic ramp and differentiating it
manufactures a smooth constant-velocity signal that says nothing about the
robot. Field groups whose speed never varies across the episode are skipped
too, which covers covariance blocks, point-cloud dimensions and strides, camera
intrinsics, and static transforms. Those channels still appear in the pickers;
they just contribute nothing if you select one.

A channel whose schema the installed decoder can't build types for (a ROS 2
definition with a lowercase constant, a schema referencing an unresolvable
package) is skipped like an empty channel rather than failing the run.

How much each tab tells you depends on what your channels carry.

| Domain | Motion tab | Health tab | Outliers tab |
|---|---|---|---|
| **Manipulation / teleop** (joint positions, end-effector pose, gripper state) | Strong. Catches jerky corrections, operator-fatigue jitter, kickback after contact | Strong for cross-arm and cross-sensor desync between telemetry channels | Strong once you have enough episodes of the same task |
| **Autonomous vehicles** (pose, velocity, steering) | Strong. Harsh braking and steering corrections show up as high jerk and low SPARC | Strong across LiDAR/IMU/GPS telemetry, where dropout and desync are common real failure modes | Strong |
| **UAV / drone** (flight-controller position, velocity, attitude) | Strong. Wind-gust artifacts and control-loop oscillation both surface as roughness | Strong | Strong |
| **Rover** (wheel odometry, pose) | Strong. Stuck-and-slip and terrain jitter both surface as roughness | Strong | Strong |
| **Egocentric / wearable** (head or body pose, IMU) | Weaker. Head turns are usually intentional, so high jerk doesn't always mean bad. Still catches literal camera shake and sensor glitches | Strong, since it only needs timestamps | Moderate |
| **Camera-only, no telemetry** | No signal. Motion switches itself off and no `quality.sparc` is written | No signal today; the health picker only offers decodable channels, so camera channels are left out (see [Limits](#limits)) | Falls back to health metrics, and is weak without motion features |

The motion picker offers any telemetry channel with a numeric field, which is
broader than the set worth scoring. A point cloud or a log channel can carry
varying numeric metadata and will be offered even though its "motion" means
nothing. Pick the channels that actually hold kinematics.

**Cross-task comparison caveat.** Isolation forest and kNN distance compare
episodes to each other within the batch you scored. If that batch mixes many
tasks, an episode can rank as an outlier because its task involves more or
faster motion than the rest, not because it was performed badly. Scoring
`Voxel51/ABC-130k` (40 episodes across 40 different tasks) puts two folding
tasks at the top, which are plausibly just the most dynamic bimanual tasks in
the set. Score episodes of the same task when you want a clean apples-to-apples
anomaly ranking, and treat cross-task outlier flags as worth a look rather than
probably bad.

## Limits

- **No programmatic playhead seek.** Clicking a row or point opens the episode
  and toasts the flagged timecode. You still scrub the timeline yourself.
- **Motion needs at least 5 samples per window**, so a channel publishing
  slower than about 2.5 Hz yields no motion windows at the default 2 s length
  and contributes nothing. Nothing in the form tells you that's why the tab came
  back empty. Raise **Window length** for slow channels.
- **`clock_drift_ppm` needs two real clocks** and is usually unavailable,
  because most writers set `publish_time` equal to `log_time`.
- **`clipping_frac` and `saturation_frac` are heuristics**, not
  hardware-calibrated. They flag values pinned at a channel's own observed
  min/max, since nothing in an MCAP file exposes a real actuator or sensor
  limit. Dimensions with fewer than three distinct values are skipped, so a
  real saturation event captured at only two levels is missed. The alternative
  reports every boolean status bit as permanently saturated. Note also that
  these can't read exactly 0: a clean signal still has one argmin and one
  argmax.
- **`desync_ms` can't resolve skew that is a whole multiple of the denser
  channel's sample interval.** A channel shifted by exactly one or two of the
  other's periods lands back on its timestamps and reads as aligned. This is
  inherent to nearest-neighbor timestamp matching without a shared reference
  event. It also reports best-case alignment, so a pair aligned for a small
  part of an episode and adrift for the rest reads as aligned.
- **Warn and fail thresholds (z >= 2, z >= 3) are calibrated for a normal
  distribution, and window-level metrics have heavier tails than that.** The
  warn band lands about where a normal predicts (~2.2% of windows), but
  fail-severity windows run around 4% against a normal's 0.13%. Combined with
  one flag stream per (channel, field group, metric), a long dynamic episode
  accumulates many short spans.
- **The health channel picker only offers `telemetry` and `scalar_sidecar`
  channels**, not `camera`. Rate and dropout need nothing but message
  timestamps and would work fine on a camera channel, but the decode path
  doesn't support camera channels yet, so they are left out of the picker
  rather than offered and silently ignored.
- **Outlier feature selection is channel-level only.** You can choose which
  channels feed the models, not which individual metrics.
- **Remote and cloud `.mcap` files** are read with a plain local `open()`.
  Byte-range reads over cloud storage are not implemented.
- **Temporal tags are written through `fiftyone.core.tags`**, an internal,
  undocumented API rather than the public SDK. It works in the version this
  plugin was built against and carries no stability guarantee.
- **Re-running overwrites** the `quality*` fields and the cached normalization
  stats under the `demo_quality_scorer` run key. There is no versioning across
  runs.
- **A `config_version` mismatch is a warning, not a hard block.** The panel
  flags a mixed view and still renders the ranking below the banner. Treat
  those scores as not yet comparable.
