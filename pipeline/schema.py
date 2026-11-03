"""Data contract for the raw extract and the features the model is served with."""

TARGET = "target"
CLASS_NAMES = ["malignant", "benign"]  # sklearn encodes malignant=0, benign=1

RAW_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error",
    "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension",
]


def to_feature_name(raw_name: str) -> str:
    """'mean concave points' -> 'mean_concave_points' (the name used in API requests)."""
    return raw_name.strip().lower().replace(" ", "_")


FEATURES = [to_feature_name(c) for c in RAW_FEATURES]
