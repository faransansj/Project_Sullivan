"""Research integrity utilities for speaker-independent AAI."""

from .contours import ArticulatorContour, ContourSample, JsonContourLoader, resample_ordered_contour
from .normalization import NormalizationArtifact
from .split_manifest import SampleAssignment, SplitManifest, build_manifest

__all__ = [
    "ArticulatorContour",
    "ContourSample",
    "JsonContourLoader",
    "NormalizationArtifact",
    "SampleAssignment",
    "SplitManifest",
    "build_manifest",
    "resample_ordered_contour",
]
