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


def _project(points, homography):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ homography.T
    return projected[:, :2] / projected[:, 2:3]


def test_estimates_known_projective_transform_and_projects_points():
    pitch = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34], [0, 0]], dtype=float)
    pitch_to_image = np.array([[8.0, 0.25, 500.0], [0.1, 7.0, 350.0], [0.0003, -0.0002, 1.0]])
    image = _project(pitch, pitch_to_image)
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is True
    assert result.inlier_count == 5
    np.testing.assert_allclose(project_points(image, result), pitch, atol=1e-5)


def test_rejects_one_outlier_with_ransac():
    pitch = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34], [0, 0], [20, -10]], dtype=float)
    pitch_to_image = np.array([[8.0, 0.25, 500.0], [0.1, 7.0, 350.0], [0.0003, -0.0002, 1.0]])
    image = _project(pitch, pitch_to_image)
    image[-1] += [180.0, -130.0]
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is True
    assert result.inlier_count == 5
    np.testing.assert_allclose(project_points(image[:5], result), pitch[:5], atol=1e-4)


def test_invertible_calibration_is_kept_when_inverse_bottom_right_is_zero(monkeypatch):
    import cv2

    pitch = np.array([[0, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    pitch_to_image = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    image = _project(pitch, pitch_to_image)

    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *args, **kwargs: (pitch_to_image.copy(), np.ones((4, 1), dtype=np.uint8)),
    )
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is True
    assert result.image_to_pitch[2, 2] == 0
    np.testing.assert_allclose(project_points(image, result), pitch)


def test_calibration_is_invariant_to_homography_scale(monkeypatch):
    import cv2

    pitch = np.array([[-52.5, -34], [52.5, -34], [52.5, 34], [-52.5, 34], [0, 0]], dtype=float)
    pitch_to_image = np.array([[8.0, 0.25, 500.0], [0.1, 7.0, 350.0], [0.0003, -0.0002, 1.0]])
    image = _project(pitch, pitch_to_image)
    scaled_homography = pitch_to_image * 1e-6

    monkeypatch.setattr(
        cv2,
        "findHomography",
        lambda *args, **kwargs: (scaled_homography.copy(), np.ones((len(pitch), 1), dtype=np.uint8)),
    )
    result = estimate_calibration(_landmarks(image, pitch))

    assert result.valid is True
    np.testing.assert_allclose(project_points(image, result), pitch, atol=1e-5)


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
