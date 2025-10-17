import mahotas
import numpy as np

import mask_enhance.imkit as imk


def adjust_text_line_coordinates(
    coords,
    width_expansion_percentage: int,
    height_expansion_percentage: int,
    img: np.ndarray,
):
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = coords
    im_h, im_w, _ = img.shape

    # Calculate width, height, and respective expansion offsets
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    width_expansion_offset = int(((width * width_expansion_percentage) / 100) / 2)
    height_expansion_offset = int(((height * height_expansion_percentage) / 100) / 2)

    # Define the rectangle origin points (bottom left, top right) with expansion/contraction
    new_x1 = max(top_left_x - width_expansion_offset, 0)
    new_y1 = max(top_left_y - height_expansion_offset, 0)
    new_x2 = min(bottom_right_x + width_expansion_offset, im_w)
    new_y2 = min(bottom_right_y + height_expansion_offset, im_h)

    return new_x1, new_y1, new_x2, new_y2


def _process_stats_vectorized(
    stats: np.ndarray, image_shape: tuple[int, int], min_area: int, margin: int = 0
) -> np.ndarray:
    """
    Helper to filter stats using NumPy vectorization, with a border margin.

    Args:
        stats (np.ndarray): The stats array from connectedComponentsWithStats.
        image_shape (tuple): The (height, width) of the image.
        min_area (int): Minimum area for a component to be considered.
        margin (int): The number of pixels to exclude from each border.
                      Components touching this margin will be removed.
    """
    # The stats array includes the background at index 0. We slice from [1:] to exclude it.
    stats_no_bg = stats[1:]

    if stats_no_bg.shape[0] == 0:
        return np.empty((0, 4), dtype=int)

    height, width = image_shape

    # 1. Create a boolean mask for all conditions

    # Condition 1: Area is greater than min_area
    area_mask = stats_no_bg[:, imk.CC_STAT_AREA] > min_area

    # Condition 2: Bounding box is NOT within the margin of the border.
    x1 = stats_no_bg[:, imk.CC_STAT_LEFT]
    y1 = stats_no_bg[:, imk.CC_STAT_TOP]
    w = stats_no_bg[:, imk.CC_STAT_WIDTH]
    h = stats_no_bg[:, imk.CC_STAT_HEIGHT]

    border_mask = (
        (x1 >= margin)
        & (y1 >= margin)
        & ((x1 + w) <= width - margin)
        & ((y1 + h) <= height - margin)
    )

    # Combine all masks
    final_mask = area_mask & border_mask

    # Apply the mask to get only the valid components
    valid_stats = stats_no_bg[final_mask]

    if valid_stats.shape[0] == 0:
        return np.empty((0, 4), dtype=int)

    # Convert the filtered stats to [x1, y1, x2, y2] format in one go
    x1_f = valid_stats[:, imk.CC_STAT_LEFT]
    y1_f = valid_stats[:, imk.CC_STAT_TOP]
    x2_f = x1_f + valid_stats[:, imk.CC_STAT_WIDTH]
    y2_f = y1_f + valid_stats[:, imk.CC_STAT_HEIGHT]

    content_bboxes = np.stack([x1_f, y1_f, x2_f, y2_f], axis=1)

    return content_bboxes.astype(int)


def detect_content_in_bbox(
    image: np.ndarray, min_area: int = 10, margin: int = 1
) -> np.ndarray:
    """
    Detects content using mahotas stats and fast NumPy vectorized filtering.

    Args:
        image (np.ndarray): Input cropped image.
        min_area (int): Minimum area of detected text box.
        margin (int): The number of pixels to exclude from each border.

    Returns:
        np.ndarray: Bounding boxes of shape (N, 4) in [x1, y1, x2, y2] format.
    """
    if image is None or image.size == 0:
        return np.empty((0, 4), dtype=int)

    gray = imk.to_gray(image)
    threshold = mahotas.thresholding.otsu(gray)

    binary_black_text = gray < threshold
    binary_white_text = gray > threshold

    # Convert boolean to uint8 for connectedComponentsWithStats
    _, _, stats_white, _ = imk.connected_components_with_stats(
        binary_white_text.astype(np.uint8), connectivity=8
    )
    _, _, stats_black, _ = imk.connected_components_with_stats(
        binary_black_text.astype(np.uint8), connectivity=8
    )

    image_shape = image.shape[:2]

    # Pass the new margin parameter to the helper function
    bboxes_white = _process_stats_vectorized(stats_white, image_shape, min_area, margin)
    bboxes_black = _process_stats_vectorized(stats_black, image_shape, min_area, margin)

    # Combine the results
    if bboxes_white.shape[0] > 0 and bboxes_black.shape[0] > 0:
        return np.vstack([bboxes_white, bboxes_black])
    elif bboxes_white.shape[0] > 0:
        return bboxes_white
    else:
        return bboxes_black


def get_inpaint_bboxes(text_bbox: list[float], image: np.ndarray) -> list[list[int]]:
    """
    Get inpaint bounding boxes for a text region.

    Args:
        text_bbox: Text bounding box [x1, y1, x2, y2]
        image: Full image

    Returns:
        list of inpaint bounding boxes
    """
    x1, y1, x2, y2 = adjust_text_line_coordinates(text_bbox, 0, 10, image)

    # Crop the image to the text bounding box
    crop = image[y1:y2, x1:x2]

    # Detect content in the cropped bubble
    content_bboxes = detect_content_in_bbox(crop)

    # Adjusting coordinates to the full image
    adjusted_bboxes = []
    for bbox in content_bboxes:
        lx1, ly1, lx2, ly2 = bbox
        adjusted_bbox = [x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2]
        adjusted_bboxes.append(adjusted_bbox)

    return adjusted_bboxes
