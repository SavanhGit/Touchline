"""Per-frame planar calibration from pitch landmarks."""

import numpy as np

from soccer_tactics.cv.video_types import CalibrationResult, PitchLandmarks


def _invalid(reason: str, inlier_count: int = 0, median_error_px: float | None = None) -> CalibrationResult:
    return CalibrationResult(False, None, inlier_count, median_error_px, reason)


def _is_collinear(points: np.ndarray) -> bool:
    centered = points - points.mean(axis=0)
    return np.linalg.matrix_rank(centered) < 2


def estimate_calibration(
    landmarks: PitchLandmarks,
    *,
    min_inliers: int = 4,
    max_reprojection_error_px: float = 12.0,
) -> CalibrationResult:
    """Estimate the image-to-pitch homography using OpenCV RANSAC.

    Errors are measured in source-image pixels by mapping inlier pitch points
    back through the inverse homography and comparing them to their landmarks.
    """
    image_xy = np.asarray(landmarks.image_xy, dtype=np.float64)
    pitch_xy = np.asarray(landmarks.pitch_xy_m, dtype=np.float64)
    confidence = np.asarray(landmarks.confidence)

    if image_xy.ndim != 2 or image_xy.shape[1:] != (2,) or pitch_xy.shape != image_xy.shape:
        return _invalid("landmark coordinates must be matching Nx2 arrays")
    if confidence.ndim != 1 or len(confidence) != len(image_xy):
        return _invalid("landmark confidence must have one value per correspondence")
    if not np.isfinite(image_xy).all() or not np.isfinite(pitch_xy).all():
        return _invalid("landmark coordinates must be finite")
    if len(image_xy) < 4:
        return _invalid("at least four correspondences are required")
    if _is_collinear(image_xy) or _is_collinear(pitch_xy):
        return _invalid("landmark correspondences must span two dimensions")
    if min_inliers < 4:
        return _invalid("min_inliers must be at least four")
    if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
        return _invalid("max_reprojection_error_px must be positive and finite")

    # OpenCV stays optional for users of the core analytics package.
    try:
        import cv2
    except ImportError as exc:
        return _invalid(f"OpenCV is required for calibration: {exc}")

    pitch_to_image, mask = cv2.findHomography(
        pitch_xy.astype(np.float64),
        image_xy.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=max_reprojection_error_px,
    )
    if pitch_to_image is None or mask is None:
        return _invalid("homography estimation failed")
    if not np.isfinite(pitch_to_image).all() or abs(float(np.linalg.det(pitch_to_image))) < 1e-12:
        return _invalid("estimated homography is noninvertible")

    inliers = mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    if inlier_count < min_inliers:
        return _invalid(f"fewer than {min_inliers} inliers", inlier_count)

    try:
        image_to_pitch = np.linalg.inv(pitch_to_image)
    except np.linalg.LinAlgError:
        return _invalid("estimated homography is noninvertible", inlier_count)
    predicted_image = _apply_homography(pitch_xy[inliers], pitch_to_image)
    errors = np.linalg.norm(predicted_image - image_xy[inliers], axis=1)
    median_error_px = float(np.median(errors))
    if not np.isfinite(median_error_px) or median_error_px > max_reprojection_error_px:
        return _invalid("median reprojection error exceeds limit", inlier_count, median_error_px)

    return CalibrationResult(True, image_to_pitch / image_to_pitch[2, 2], inlier_count, median_error_px, None)


def _apply_homography(points_xy: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = homogeneous @ transform.T
    denominator = projected[:, 2]
    if np.any(np.isclose(denominator, 0)):
        raise ValueError("homography maps a point to infinity")
    return projected[:, :2] / denominator[:, None]


def project_points(points_xy: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Project source-image points to pitch metres using a valid calibration."""
    if not calibration.valid or calibration.image_to_pitch is None:
        raise ValueError("cannot project points with an invalid calibration")
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("points_xy must be a finite Nx2 array")
    return _apply_homography(points, calibration.image_to_pitch)
