"""Pipeline stages for the DAM405 serverless breast-cancer classifier."""


class QualityGateError(RuntimeError):
    """Raised when data or model quality falls below the agreed thresholds."""
