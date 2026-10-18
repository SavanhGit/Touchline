"""Phase 2: video -> tracking.

Wrap roboflow/sports (pitch keypoints, player/ball detection, tracking, homography to
pitch meters) and emit frames matching `soccer_tactics.schema.COLUMNS`, so everything
downstream runs unchanged.
"""
