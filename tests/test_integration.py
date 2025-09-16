from unittest.mock import MagicMock

import numpy as np


def test_full_analysis_pipeline():
    """Test the complete analysis pipeline with mock data."""
    # Create synthetic test data
    test_image = np.random.rand(50, 100, 100).astype(np.float32) * 100

    # Add some hot spots
    test_image[20:30, 40:50, 40:50] = 1000.0  # Hot sphere

    # Mock configuration
    cfg = MagicMock()
    cfg.ACTIVITY.HOT = 4000.0
    cfg.ACTIVITY.BACKGROUND = 1000.0
    cfg.ROIS.CENTRAL_SLICE = 25
    cfg.ROIS.SPACING = 2.0644
    cfg.ROIS.BACKGROUND_OFFSET_YX = [(-10, -10), (10, 10)]

    # Test the pipeline
    # This would test the integration of multiple modules
    pass
