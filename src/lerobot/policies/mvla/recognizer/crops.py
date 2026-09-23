"""Turning a wrist frame into the input the recognizer is validated on.

The crop is not preprocessing detail, it is the experiment. Feeding whole
frames to a frozen encoder scores 88.8% on the mat-state task and, worse,
reaches 94.9% from a strip of bench that contains no object at all -- the
features encode which recording session a frame came from, and that cue
inverts on unseen sessions (45.5%, below chance). Replacing the background
with flat grey removes the cue outright and lifts the honest score to 97.9%.

So each task gets the crop it was measured with:

  mat_state       the part silhouette, background masked to grey.
                  What matters is which face is up; the bench is noise.

  tray_placement  a fixed window under the gripper at the moment of
                  release. What matters is the part *relative to* its
                  pocket, so the local mould has to stay -- masking it away
                  costs AUC (0.983 -> 0.958).

`placebo_crop` is the control that caught the artifact and is kept beside
the real crops on purpose: any change here is re-validated by checking that
a region with no object still scores near chance.
"""

from __future__ import annotations

import cv2
import numpy as np

# Pixels darker than this are candidate object. The part is black plastic on
# a green mat / white mould, so a value threshold separates it cleanly --
# 702/702 and 765/765 detections on the validation frames.
DARK_VALUE_MAX = 70
# Blobs smaller than this are speckle: cable shadows, bolt heads, mat scuffs.
MIN_BLOB_AREA = 800
MIN_BLOB_AREA_TRAY = 1500

# Window under the gripper holding the part just released. Fixed rather than
# detected because the wrist camera is rigid to the gripper, so the part it
# just let go of lands in the same place in frame every time.
TRAY_WINDOW = (160, 80, 500, 420)  # x0, y0, x1, y1 in a 640x480 frame
# Corner far from that window, used only as the no-object control.
PLACEBO_WINDOW = (0, 0, 180, 180)

GREY = 128
OUT_SIZE = 224


def _largest_dark_blob(
    bgr: np.ndarray, min_area: int, prefer_center: bool
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Mask and bbox of the object blob, or None if nothing qualifies.

    `prefer_center` picks the blob nearest the image centre instead of the
    biggest one: inside the tray several placed parts touch and the biggest
    component spans the whole kit, while the one under the gripper -- the
    only one this placement is about -- is the central one.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < DARK_VALUE_MAX).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
    if count < 2:
        return None

    best, best_key = None, None
    h, w = bgr.shape[:2]
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if prefer_center:
            key = (centroids[i][0] - w / 2) ** 2 + (centroids[i][1] - h / 2) ** 2
            better = best_key is None or key < best_key
        else:
            key = -area
            better = best_key is None or key < best_key
        if better:
            best, best_key = i, key
    if best is None:
        return None
    x, y, bw, bh = (int(v) for v in stats[best, :4])
    return (labels == best).astype(np.uint8), (x, y, bw, bh)


def _square_around(
    bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Square crop centred on `bbox`, grey-padded, optionally background-masked.

    Square and bbox-scaled so the object fills the frame at a consistent
    size regardless of where it sits on the mat or how far the wrist is.
    """
    x, y, w, h = bbox
    side = max(w, h) + 2 * int(margin * max(w, h))
    src = bgr if mask is None else np.where(mask[:, :, None] > 0, bgr, np.uint8(GREY))
    pad = cv2.copyMakeBorder(src, side, side, side, side, cv2.BORDER_CONSTANT, value=(GREY,) * 3)
    x0, y0 = x + w // 2 - side // 2, y + h // 2 - side // 2
    return pad[y0 + side : y0 + 2 * side, x0 + side : x0 + 2 * side]


def mat_crop(bgr: np.ndarray, size: int = OUT_SIZE) -> tuple[np.ndarray, bool]:
    """Object silhouette on flat grey, for `mat_state`.

    Returns (crop, detected). On a miss the whole frame is returned resized,
    which keeps batch shapes stable; callers that care read the flag.
    """
    found = _largest_dark_blob(bgr, MIN_BLOB_AREA, prefer_center=False)
    if found is None:
        return cv2.resize(bgr, (size, size)), False
    mask, bbox = found
    return cv2.resize(_square_around(bgr, bbox, 0.25, mask), (size, size)), True


def tray_crop(bgr: np.ndarray, size: int = OUT_SIZE) -> tuple[np.ndarray, bool]:
    """Fixed window under the gripper, for `tray_placement`.

    Unmasked: whether the part is seated is a statement about the part and
    the pocket together, so the mould has to be visible.
    """
    x0, y0, x1, y1 = TRAY_WINDOW
    return cv2.resize(bgr[y0:y1, x0:x1], (size, size)), True


def tray_masked_crop(bgr: np.ndarray, size: int = OUT_SIZE) -> tuple[np.ndarray, bool]:
    """Part-only variant of `tray_crop`, background removed.

    Scores lower than `tray_crop` (AUC 0.958 vs 0.983) because it discards
    the pocket, but it is the variant with no background channel at all, so
    it is the number that transfers to a re-arranged cell.
    """
    found = _largest_dark_blob(bgr, MIN_BLOB_AREA_TRAY, prefer_center=True)
    if found is None:
        return tray_crop(bgr, size)
    mask, bbox = found
    return cv2.resize(_square_around(bgr, bbox, 0.30, mask), (size, size)), True


def placebo_crop(bgr: np.ndarray, size: int = OUT_SIZE) -> tuple[np.ndarray, bool]:
    """Control input: a region the object is never in.

    A recognizer that scores well on this is reading the room, not the part.
    """
    x0, y0, x1, y1 = PLACEBO_WINDOW
    return cv2.resize(bgr[y0:y1, x0:x1], (size, size)), True


CROPPERS = {
    "mat": mat_crop,
    "tray": tray_crop,
    "tray_masked": tray_masked_crop,
    "placebo": placebo_crop,
}
