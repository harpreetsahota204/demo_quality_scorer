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
    p.register(ComputeEpisodeQuality)
    p.register(GetQualityPanelData)
    p.register(OpenQualityEpisode)
    p.register(TagQualityEpisodes)
    p.register(ShowQualityEpisodes)
    p.register(PromptQualityScorer)
