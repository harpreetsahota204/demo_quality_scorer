"""Unlisted operators backing the React Episode Quality panel.

The panel is a JS component (``src/``) with no Python Panel class; these
operators are its backend. All are ``unlisted`` -- they exist to be called
via ``useOperatorExecutor``, not from the operator browser.
"""

import fiftyone.operators as foo

from .panel_data import build_panel_data, worst_interval_message


class GetQualityPanelData(foo.Operator):
    """Serves the panel its entire dataset in one call.

    Deliberately one round trip: the frontend needs every episode's metrics to
    draw distributions and to resolve chart clicks to sample IDs client-side,
    so paginating here would just mean the panel re-requesting everything.
    Called with both `ctx.dataset` and `ctx.view` so the payload can report
    corpus-wide context while respecting what the user is currently filtered to.
    """

    @property
    def config(self):
        return foo.OperatorConfig(name="get_quality_panel_data", unlisted=True)

    def execute(self, ctx):
        return build_panel_data(ctx.dataset, ctx.view)


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
    """Records a reviewer's verdict as a sample tag.

    Tags, not a new field: the whole point of the scorer is that a human
    decides, and a tag is the artifact the rest of FiftyOne (views, exports,
    other plugins) already understands, so a verdict is usable outside this
    panel.

    An empty `sample_ids` falls back to the current view, which is what makes
    "tag everything I'm looking at" work after the user filters to a bad
    cohort -- but it also means a caller that meant "tag nothing" would tag
    the view, so the frontend is responsible for not calling with an empty
    selection it didn't intend.
    """

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
