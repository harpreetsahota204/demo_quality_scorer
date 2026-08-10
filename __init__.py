"""Demo Quality Scorer: episode/interval quality triage for multimodal MCAP datasets."""

from .operators import ComputeEpisodeQuality
from .panel_ops import (
    GetQualityPanelData,
    OpenQualityEpisode,
    PromptQualityScorer,
    ShowQualityEpisodes,
    TagQualityEpisodes,
)


def register(p):
    """FiftyOne's plugin entry point.

    Only `ComputeEpisodeQuality` is listed in the operator browser; the rest
    are the React panel's unlisted backend (see `panel_ops`) and must still be
    registered here to be callable from it.
    """
    p.register(ComputeEpisodeQuality)
    p.register(GetQualityPanelData)
    p.register(OpenQualityEpisode)
    p.register(TagQualityEpisodes)
    p.register(ShowQualityEpisodes)
    p.register(PromptQualityScorer)
