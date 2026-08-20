"""Deterministic offline analysis for closed experiment runs.

The package is deliberately separate from provider and runtime code.  Importing
it performs no network calls, and analyzing a run never modifies the four raw
records from which the result is derived.
"""

from detecting_an_insider_llm.analysis.offline import (
    ANALYZER_ID,
    ANALYZER_VERSION,
    RESULTS_FILENAME,
    AnalysisError,
    AnalysisInputError,
    AnalysisWriteError,
    AnalysisWriteResult,
    OfflineAnalyzer,
)

__all__ = [
    "ANALYZER_ID",
    "ANALYZER_VERSION",
    "RESULTS_FILENAME",
    "AnalysisError",
    "AnalysisInputError",
    "AnalysisWriteError",
    "AnalysisWriteResult",
    "OfflineAnalyzer",
]
