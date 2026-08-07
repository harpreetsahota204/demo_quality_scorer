"""Unlisted operators backing the React Episode Quality panel.

The panel is a JS component (``src/``) with no Python Panel class; these
operators are its backend. All are ``unlisted`` -- they exist to be called
via ``useOperatorExecutor``, not from the operator browser.
"""

import fiftyone.operators as foo

from . import debug
from .panel_data import build_panel_data, worst_interval_message


class GetQualityPanelData(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(name="get_quality_panel_data", unlisted=True)

    def execute(self, ctx):
        data = build_panel_data(ctx.dataset, ctx.view)
        debug.log(
            ctx,
            "get_quality_panel_data",
            "execute",
            scored=data.get("scored"),
            n_rows=len(data.get("rows", [])),
        )
        return data


class OpenQualityEpisode(foo.Operator):
    """Deep-links into one episode's viewer, with a worst-flag toast.

    The toast is the PRD's own workaround for FiftyOne having no
    programmatic playhead-seek yet -- we can open the episode but not
    scrub it, so we tell the user where to scrub.
    """

    @property
    def config(self):
        return foo.OperatorConfig(name="open_quality_episode", unlisted=True)

    def execute(self, ctx):
        sample_id = ctx.params.get("sample_id")
        if not sample_id:
            return {}

        ctx.ops.open_sample(sample_id)
        message = worst_interval_message(ctx.dataset[sample_id])
        if message:
            ctx.ops.notify(message, variant="warning")
        else:
            ctx.ops.notify("No flagged intervals for this episode.")
        return {}


class TagQualityEpisodes(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(name="tag_quality_episodes", unlisted=True)

    def execute(self, ctx):
        tag = ctx.params["tag"]
        ids = ctx.params.get("sample_ids") or []
        target = ctx.dataset.select(ids) if ids else ctx.view
        target.tag_samples(tag)
        ctx.ops.notify(f"Tagged {len(target)} episode(s) as '{tag}'.")
        return {"tagged": len(target)}


class ShowQualityEpisodes(foo.Operator):
    """Filters the samples panel to episodes picked in a panel chart.

    The frontend already holds every row's metric values, so chart clicks
    (histogram bins, verdict bars) resolve to sample IDs client-side and
    this operator just selects them -- one code path regardless of whether
    the clicked dimension is a real sample field (motion metrics) or a
    panel-computed one (health verdicts).
    """

    @property
    def config(self):
        return foo.OperatorConfig(name="show_quality_episodes", unlisted=True)

    def execute(self, ctx):
        ids = ctx.params.get("sample_ids") or []
        description = ctx.params.get("description") or "selection"
        if not ids:
            ctx.ops.notify(f"No episodes match {description}.")
            return {}

        ctx.ops.set_view(view=ctx.dataset.select(ids, ordered=True))
        ctx.ops.notify(
            f"Showing {len(ids)} episode(s): {description}. "
            "Clear the view bar to see everything again."
        )
        return {}


class PromptQualityScorer(foo.Operator):
    """Opens the scorer from the panel's empty state."""

    @property
    def config(self):
        return foo.OperatorConfig(name="prompt_quality_scorer", unlisted=True)

    def execute(self, ctx):
        ctx.trigger("demo-quality-scorer/compute_episode_quality")
        return {}
