import numpy as np
import pytest

pytest.importorskip("cv2")

from soccer_tactics.cv.calibration import estimate_calibration, project_points
from soccer_tactics.cv.video_types import PitchLandmarks


def _landmarks(image_xy, pitch_xy_m, confidence=None):
    if confidence is None:
        confidence = np.ones(len(image_xy), dtype=float)
    return PitchLandmarks(
        image_xy=np.asarray(image_xy, dtype=float),
        pitch_xy_m=np.asarray(pitch_xy_m, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
    )


def test_estimates_known_projective_transform_and_projects_points():
    pitch = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34], [0, 0]], dtype=float)
    image = np.array([[100, 80], [900, 100], [850, 620], [130, 600], [510, 350]], dtype=float)
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is True
    assert result.inlier_count == 5
    np.testing.assert_allclose(project_points(image, result), pitch, atol=1e-5)


def test_rejects_one_outlier_with_ransac():
    pitch = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34], [0, 0], [20, -10]], dtype=float)
    image = np.array([[100, 80], [900, 100], [850, 620], [130, 600], [510, 350], [640, 280]], dtype=float)
    pitch[-1] = [300, 250]
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is True
    assert result.inlier_count == 5
    np.testing.assert_allclose(project_points(image[:5], result), pitch[:5], atol=1e-4)


@pytest.mark.parametrize("image,pitch", [
    (np.array([[0, 0], [1, 0], [1, 1]], dtype=float), np.array([[0, 0], [1, 0], [1, 1]], dtype=float)),
    (np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float), np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)),
    (np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float), np.array([[0, 0], [1, 0], [1, 1], [np.nan, 1]], dtype=float)),
])
def test_rejects_insufficient_degenerate_or_nonfinite_correspondences(image, pitch):
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is False
    assert result.reason


def test_rejects_median_reprojection_error_over_configured_limit(monkeypatch):
    import cv2

    image = np.array([[100, 80], [900, 100], [850, 620], [130, 600]], dtype=float)
    pitch = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def inaccurate_homography(*args, **kwargs):
        assert kwargs["ransacReprojThreshold"] == 12.0
        return np.eye(3), np.ones((4, 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "findHomography", inaccurate_homography)
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is False
    assert result.reason
