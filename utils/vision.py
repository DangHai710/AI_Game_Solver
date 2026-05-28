import cv2
import numpy as np


WARP_SIZE = 450
GRID_SIZE = 9


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None and img.ndim == 3 else img


def order_points(points):
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1)
    return np.array(
        [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
        dtype=np.float32,
    )


def preprocess_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)


def find_grid_contour(img, max_dim=1200):
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img
    bw = preprocess_grid(small)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = bw.shape[0] * bw.shape[1]
    best, best_score = None, 0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(cnt)
        if area < image_area * 0.03:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in (0.02, 0.04, 0.06):
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = order_points(approx)
                quad_area = cv2.contourArea(rect)
                width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
                height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
                aspect = width / max(height, 1)
                if 0.65 <= aspect <= 1.45 and quad_area > best_score:
                    best, best_score = rect, quad_area
                break

    if best is None:
        return None
    return (best / scale).astype(np.float32) if scale < 1 else best.astype(np.float32)


def warp_grid(img, contour, size=WARP_SIZE):
    dst = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    matrix = cv2.getPerspectiveTransform(order_points(contour), dst)
    return cv2.warpPerspective(img, matrix, (size, size)), matrix


def _center_crop(img, ratio=0.8):
    h, w = img.shape[:2]
    nh, nw = int(h * ratio), int(w * ratio)
    y, x = (h - nh) // 2, (w - nw) // 2
    return img[y : y + nh, x : x + nw]


def _clean_digit_mask(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    h, w = binary.shape[:2]
    mask = np.zeros_like(binary)
    total = h * w
    kept_components = 0
    largest_area = 0

    for label in range(1, n):
        x, y, bw, bh, area = stats[label]
        if not (total * 0.008 <= area <= total * 0.28):
            continue
        if bw > w * 0.78 or bh > h * 0.9:
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        if not (0.15 <= bw / max(bh, 1) <= 1.5):
            continue
        if area / max(bw * bh, 1) > 0.82:
            continue
        mask[labels == label] = 255
        kept_components += 1
        largest_area = max(largest_area, area)

    return mask, cv2.countNonZero(mask), kept_components, largest_area


def preprocess_cell(cell, size=28):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell.copy()
    gray = _center_crop(gray, 0.8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    candidates = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(otsu)
    candidates.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8))
    candidates.append(cv2.threshold(gray, min(180, int(np.mean(gray) * 0.9)), 255, cv2.THRESH_BINARY_INV)[1])

    best_mask = np.zeros_like(gray)
    best_area = 0
    best_components = 0
    best_largest_area = 0
    for binary in candidates:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        mask, area, components, largest_area = _clean_digit_mask(binary)
        score = area + largest_area * 0.5 - max(0, components - 1) * 0.05 * gray.size
        best_score = best_area + best_largest_area * 0.5 - max(0, best_components - 1) * 0.05 * gray.size
        if score > best_score:
            best_mask = mask
            best_area = area
            best_components = components
            best_largest_area = largest_area

    ink_ratio = best_area / float(gray.size)
    dominant_ratio = best_largest_area / float(gray.size)
    # Use slightly stricter thresholds to avoid false positives on noisy/blurred cells
    is_empty = ink_ratio < 0.02 or dominant_ratio < 0.01 or (best_components >= 3 and ink_ratio < 0.06)
    output = np.full_like(gray, 255) if is_empty else cv2.bitwise_not(best_mask)
    output = cv2.resize(output, (size, size), interpolation=cv2.INTER_AREA)
    return output[..., None].astype("float32") / 255.0, output, is_empty, ink_ratio, best_components, dominant_ratio


def split_cells(warped):
    cells, side = [], warped.shape[0] // GRID_SIZE
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            cell = warped[r * side : (r + 1) * side, c * side : (c + 1) * side]
            model_img, display, is_empty, ink_ratio, component_count, dominant_ratio = preprocess_cell(cell)
            row.append({
                "model": model_img,
                "display": display,
                "is_empty": is_empty,
                "ink_ratio": ink_ratio,
                "component_count": component_count,
                "dominant_ratio": dominant_ratio,
            })
        cells.append(row)
    return cells


def overlay_solution_on_original(img, contour, givens, solved):
    result = img.copy()
    src = np.float32([[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]])
    inv = cv2.getPerspectiveTransform(src, order_points(contour))
    step = WARP_SIZE / 9.0

    for r in range(9):
        for c in range(9):
            if givens[r][c] != 0 or solved[r][c] == 0:
                continue
            pt = np.float32([[[c * step + step / 2, r * step + step * 0.65]]])
            x, y = cv2.perspectiveTransform(pt, inv)[0][0].astype(int)
            cv2.putText(result, str(int(solved[r][c])), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 80, 0), 2, cv2.LINE_AA)
    return result
