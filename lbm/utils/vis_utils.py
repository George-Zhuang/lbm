from typing import Dict, List, Optional, Tuple, Union
import os
import torch
import random
import imageio
import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw

# --- Constants ---
MAX_IDS: int = 256
RADIUS_DEFAULT: int = 4
BOX_LINE_WIDTH: int = 2
TEXT_BG_ALPHA: int = 128
TEXT_MARGIN: int = 5

# --- Color Generation ---
colormap = cm.get_cmap("gist_rainbow")

# Generate normalized colors
normalized_colors = [colormap(i / (MAX_IDS - 1))[:3] for i in range(MAX_IDS)]
# Convert to RGB tuples (0-255)
colors_rgb: List[Tuple[int, int, int]] = [
    tuple(int(c * 255) for c in norm_color) for norm_color in normalized_colors
]
# Shuffle colors for better visual distinction
random.shuffle(colors_rgb)
# Create a mapping from ID to color
ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {
    i: colors_rgb[i] for i in range(MAX_IDS)
}
DEFAULT_COLOR: Tuple[int, int, int] = (255, 0, 0) # Default color if ID exceeds MAX_IDS


def _get_numpy(tensor: Union[torch.Tensor, np.ndarray, List, Tuple]) -> np.ndarray:
    """Convert input to a NumPy array on CPU."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Unsupported type: {type(tensor)}")


def draw_points(
    rgb: np.ndarray,
    coords: Optional[torch.Tensor],
    visibility: Optional[Union[torch.Tensor, List, Tuple]],
    radius: int = RADIUS_DEFAULT,
) -> np.ndarray:
    """
    Draws points on an image using Pillow (PIL).

    Args:
        rgb: Input image (H, W, 3) in RGB order.
        coords: Point coordinates (N, 2). If None, returns original image.
        visibility: Visibility flags for each point (N,). If None, assumes all visible.
        radius: Radius of the points to draw.

    Returns:
        Image with points drawn.
    """
    if coords is None:
        return rgb.copy() # Return a copy to avoid modifying the original

    # Ensure the image data is in uint8 format for Pillow
    rgb_uint8 = rgb.copy().astype(np.uint8)
    img = Image.fromarray(rgb_uint8) # Work on a copy
    draw = ImageDraw.Draw(img)

    coords_np = _get_numpy(coords)
    visibility_np = _get_numpy(visibility) if visibility is not None else np.ones(len(coords_np), dtype=bool)

    height, _ = rgb.shape[:2]

    for (x, y), vis in zip(coords_np, visibility_np):
        x_int, y_int = int(x), int(y)
        if x_int == 0 and y_int == 0: # Skip origin points if they mean invalid
            continue

        # Color based on vertical position
        color_normalized = colormap(y_int / height)[:3]
        color = tuple(int(c * 255) for c in color_normalized)

        # Define the bounding box for the circle
        bbox = [x_int - radius, y_int - radius, x_int + radius, y_int + radius]

        if vis:
            draw.ellipse(bbox, fill=color) # Filled circle for visible
        else:
            draw.ellipse(bbox, outline=color, width=1) # Outline for non-visible

    return np.array(img)


def draw_boxes(
    rgb: np.ndarray,
    track_instances: Dict[str, torch.Tensor]
) -> np.ndarray:
    """
    Draws bounding boxes, IDs, labels, and scores on an image.

    Args:
        rgb: Input image (H, W, 3) in RGB order.
        track_instances: Dictionary with 'bboxes', 'instances_id',
    Returns:
        Image with boxes and labels drawn.
    """
    # Ensure the image data is in uint8 format for Pillow
    rgb_uint8 = rgb.copy().astype(np.uint8)
    img = Image.fromarray(rgb_uint8) # Work on a copy
    draw = ImageDraw.Draw(img)

    # Ensure all data is on CPU as numpy arrays

    # Ensure all data is on CPU as numpy arrays
    boxes = _get_numpy(track_instances['bboxes'])
    ids = _get_numpy(track_instances['instances_id'])
    scores = _get_numpy(track_instances['scores'])
    labels = _get_numpy(track_instances['labels']) # Assuming labels are numerical or string convertible

    for box, score, label, id_num in zip(boxes, scores, labels, ids):
        x1, y1, x2, y2 = map(int, box)
        instance_id = int(id_num)
        color = ID_TO_COLOR.get(instance_id % MAX_IDS, DEFAULT_COLOR)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_LINE_WIDTH)

        # Prepare text
        text = f"ID:{instance_id}, {label}: {score:.2f}"

        # Calculate text size and position using textbbox for better accuracy
        try:
            # textbbox requires xy argument, (0,0) is fine for size calculation
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Fallback for older Pillow versions
             text_width, text_height = draw.textsize(text)


        # Position text box above the bounding box
        text_x = x1
        text_y = y1 - text_height - TEXT_MARGIN # Position text above box
        bg_y1 = max(0, text_y) # Ensure background doesn't go offscreen top
        bg_y2 = y1

        # Draw text background rectangle with transparency
        bg_fill_color = color + (TEXT_BG_ALPHA,)
        draw.rectangle([x1, bg_y1, x1 + text_width, bg_y2], fill=bg_fill_color)

        # Draw text
        draw.text((text_x, text_y), text, fill=(255, 255, 255)) # White text

    return np.array(img)

def draw_trajectory(
    rgb: np.ndarray,
    trajectories: Union[torch.Tensor, np.ndarray],
    vis: Union[torch.Tensor, np.ndarray] = None,
    rho: Union[torch.Tensor, np.ndarray] = None,
    max_len: int = 30,
    radius: int = RADIUS_DEFAULT,
    line_width: int = 4,
    vis_threshold: float = 0.8,
    rho_threshold: float = 0.1,
    reset_masked_trajectories: bool = True,
) -> Tuple[np.ndarray, Union[torch.Tensor, np.ndarray]]:
    """
    Draws multiple trajectories on an image with fading lines and final points.

    Each trajectory's line segments fade (increase transparency) further back in time.
    The color of each trajectory and its end point is determined by the
    vertical position of the last point in that specific trajectory.

    Args:
        rgb: Input image (H, W, 3) in RGB order.
        trajectories: Point coordinates for the trajectories (N, T, 2), where N
                      is the number of trajectories and T is the number of points
                      per trajectory. Points are ordered chronologically.
        vis: Visibility scores for each trajectory (N,). If provided, trajectories
             with visibility < vis_threshold will be filtered out.
        rho: Confidence scores for each trajectory (N,). If provided, trajectories
             with confidence > rho_threshold will be filtered out.
        max_len: Maximum number of points to draw from the end of each trajectory.
        radius: Radius of the final points.
        line_width: Width of the trajectory lines.
        vis_threshold: Minimum visibility threshold for drawing trajectories.
        rho_threshold: Maximum confidence threshold for drawing trajectories.
        reset_masked_trajectories: If True, reset filtered trajectories to zeros
                                  to prevent them from being drawn in future visualizations.

    Returns:
        Tuple containing:
        - Image (np.ndarray) with the trajectories drawn.
        - Trajectories array: modified with masked trajectories reset to zeros if reset_masked_trajectories=True
    """
    trajectories_np = _get_numpy(trajectories)
    
    # Create a copy for potential modification
    trajectories_modified = trajectories_np.copy()

    # If no trajectories, return the original image
    if trajectories_np.shape[0] == 0:
        return rgb.copy(), trajectories

    # Apply combined visibility and confidence mask
    combined_mask = None
    
    if vis is not None:
        vis_np = _get_numpy(vis)
        vis_mask = vis_np >= vis_threshold
        combined_mask = vis_mask
    
    if rho is not None:
        rho_np = _get_numpy(rho)
        rho_mask = rho_np <= rho_threshold
        if combined_mask is not None:
            # Both vis and rho are provided - use AND operation
            combined_mask = combined_mask & rho_mask
        else:
            # Only rho is provided
            combined_mask = rho_mask
    
    # Reset masked trajectories if requested
    if reset_masked_trajectories and combined_mask is not None:
        # Reset trajectories that don't pass the mask to zeros
        masked_indices = ~combined_mask
        trajectories_modified[masked_indices] = 0
    
    # Apply the combined mask if any mask was created
    if combined_mask is not None:
        trajectories_np = trajectories_np[combined_mask]
        
        # If no trajectories left after filtering, return the original image
        if trajectories_np.shape[0] == 0:
            return rgb.copy(), torch.from_numpy(trajectories_modified)

    # Convert background image to RGBA PIL Image for alpha compositing
    rgb_uint8 = rgb.copy().astype(np.uint8)
    img = Image.fromarray(rgb_uint8).convert('RGBA')
    # Create a transparent overlay layer to draw the trajectories
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    height, _ = rgb.shape[:2]

    # Iterate through each trajectory
    for traj_idx in range(trajectories_np.shape[0]):
        traj_np = trajectories_np[traj_idx] # Get single trajectory (T, 2)

        # Filter out points that are (0, 0), often used as padding/invalid
        valid_points_mask = (traj_np[:, 0] != 0) | (traj_np[:, 1] != 0)
        traj_np = traj_np[valid_points_mask]

        # If no valid points after filtering, skip this trajectory
        if traj_np.shape[0] == 0:
            continue

        # Keep only the last max_len points
        if traj_np.shape[0] > max_len:
            traj_np = traj_np[-max_len:]

        num_points = traj_np.shape[0]

        # If no points left after length check, skip
        if num_points == 0:
            continue

        # Get the last point to determine color and draw the end marker
        x_last, y_last = traj_np[-1]
        x_last_int, y_last_int = int(x_last), int(y_last)

        # Determine base color from the last point's y-coordinate
        # Clamp y_last_int to avoid division by zero or index out of bounds if height is 0 or 1
        clamped_y = max(0, min(y_last_int, height - 1))
        color_normalized = colormap(clamped_y / height if height > 1 else 0.5)[:3]
        base_color = tuple(int(c * 255) for c in color_normalized)

        # Draw trajectory lines if there are at least 2 points
        if num_points >= 2:
            num_segments = num_points - 1
            for i in range(num_segments):
                # i = 0 is the oldest segment, i = num_segments - 1 is the newest
                p1 = traj_np[i]
                p2 = traj_np[i+1]
                x1_int, y1_int = int(p1[0]), int(p1[1])
                x2_int, y2_int = int(p2[0]), int(p2[1])

                # Calculate transparency: newest segment (index num_segments - 1) has transparency 0.
                # Transparency increases for older segments. Adjust factor as needed.
                segment_index_from_newest = (num_segments - 1) - i
                # Make fade more pronounced, e.g., scale alpha from 50 to 255
                min_alpha = 50
                max_alpha = 255
                alpha_range = max_alpha - min_alpha
                # Alpha is higher for newer segments
                alpha = int(min_alpha + (alpha_range * (i + 1) / num_segments))
                alpha = max(0, min(255, alpha)) # Clamp alpha between 0 and 255


                line_color_rgba = base_color + (alpha,)
                draw_overlay.line([(x1_int, y1_int), (x2_int, y2_int)], fill=line_color_rgba, width=line_width)

        # Draw the final point (fully opaque)
        # Check again if the last point is valid (might be the only point)
        if x_last_int != 0 or y_last_int != 0:
            bbox = [x_last_int - radius, y_last_int - radius, x_last_int + radius, y_last_int + radius]
            # Use base color with full alpha (255)
            end_point_color_rgba = base_color + (255,)
            draw_overlay.ellipse(bbox, fill=end_point_color_rgba)

    # Composite the trajectory overlay onto the original image
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB NumPy array
    result_image = np.array(img.convert('RGB'))
    
    return result_image, torch.from_numpy(trajectories_modified)


def save_visualization(frame, pred_track_instances, coord, vis, save_path, model):
    """Saves the visualization of a single frame."""
    frame_numpy = frame[0].cpu().numpy().transpose(1, 2, 0) # Convert to HWC format
    vis_frame = draw_boxes(frame_numpy, pred_track_instances)
    if coord is None:
        # Use default query points if no specific coordinates are available
        coord = model.query_points
        vis = torch.ones_like(coord[:, 0], dtype=torch.float32)
    vis_frame = draw_points(vis_frame, coord, vis)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(str(save_path), vis_frame)
