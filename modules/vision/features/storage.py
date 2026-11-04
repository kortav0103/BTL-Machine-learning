"""Feature serialization and associated sample metadata."""


def save_features(features, manifest, config):
    """Save features with ordered image IDs, labels, splits, and extraction settings."""
    raise NotImplementedError("save_features is not implemented")


def load_features(path):
    """Load features and metadata, validating row alignment."""
    raise NotImplementedError("load_features is not implemented")
