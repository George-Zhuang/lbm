from typing import Dict, List, Optional, Tuple, Union
import os
import torch
import pickle
import random
import imageio
import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import mediapy as media
import cv2
import time
import open3d as o3d
import matplotlib

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
        bg_y2 = max(y1, bg_y1 + text_height + TEXT_MARGIN) # Ensure background doesn't go offscreen bottom

        # Draw text background rectangle with transparency
        bg_fill_color = color + (TEXT_BG_ALPHA,)
        draw.rectangle([x1, bg_y1, x1 + text_width, bg_y2], fill=bg_fill_color)

        # Draw text
        draw.text((text_x, text_y), text, fill=(255, 255, 255)) # White text

    return np.array(img)

def draw_trajectory(
    rgb: np.ndarray,
    trajectories: Union[torch.Tensor, np.ndarray],
    max_len: int = 30,
    radius: int = RADIUS_DEFAULT,
    line_width: int = 4,
) -> np.ndarray:
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
        max_len: Maximum number of points to draw from the end of each trajectory.
        radius: Radius of the final points.
        line_width: Width of the trajectory lines.

    Returns:
        Image (np.ndarray) with the trajectories drawn.
    """
    trajectories_np = _get_numpy(trajectories)

    # If no trajectories, return the original image
    if trajectories_np.shape[0] == 0:
        return rgb.copy()

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
    return np.array(img.convert('RGB'))


def draw_trajectory_3d(
    trajectories: Union[torch.Tensor, np.ndarray], # n t 3
    max_len: int = 30,
    radius: int = RADIUS_DEFAULT,
    line_width: int = 2,
    elev: float = -45.0,  # elevation angle in degrees
    azim: float = 170.0,  # azimuth angle in degrees
    roll: float = -80.0,  # roll angle in degrees
    dist: float = 3.0,  # distance from the center
    show_axes: bool = False,  # whether to show coordinate axes
    show_grid: bool = False,  # whether to show grid lines
    axis_labels: bool = False,  # whether to show axis labels
    show_plot: bool = True,  # whether to show the plot interactively
    ax: Optional[plt.Axes] = None,  # existing axes to plot on
    clear_ax: bool = True,  # whether to clear the axes before plotting
) -> Tuple[np.ndarray, plt.Axes]:
    """
    Draw trajectories in 3D coordinate system.

    Args:
        trajectories: Point coordinates for trajectories (N, T, 3), where N is the number of trajectories,
                     T is the number of points per trajectory, and 3 represents xyz coordinates.
        max_len: Maximum number of points to draw from the end of each trajectory.
        radius: Radius of the end point markers.
        line_width: Width of the trajectory lines.
        elev: Elevation angle in degrees (vertical rotation).
        azim: Azimuth angle in degrees (horizontal rotation).
        roll: Roll angle in degrees (horizontal rotation).
        dist: Distance from the center of the plot.
        show_axes: Whether to show coordinate axes.
        show_grid: Whether to show grid lines.
        axis_labels: Whether to show axis labels.
        show_plot: Whether to show the plot interactively.
        ax: Existing axes to plot on. If None, creates new axes.
        clear_ax: Whether to clear the axes before plotting.

    Returns:
        Tuple of (image array containing the 3D trajectory plot, axes object)
    """
    trajectories_np = _get_numpy(trajectories)

    # Return blank image if no trajectories
    if trajectories_np.shape[0] == 0:
        return ax

    # Create or use existing figure and axes
    if ax is None:
        fig = plt.figure(figsize=(10, 10), facecolor='none')
        ax = fig.add_subplot(111, projection='3d', facecolor='none')
    elif clear_ax:
        ax.clear()

    # Set fixed view angle
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.dist = dist

    # Set fixed axis limits
    ax.set_xlim([10, -10])
    ax.set_ylim([0, 6])
    ax.set_zlim([0, 60])

    # Draw ground plane (y = 0)
    x = np.linspace(10, -10, 2)
    z = np.linspace(0, 60, 2)
    X, Z = np.meshgrid(x, z)
    Y = np.ones_like(X) * 0
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.3)

    # Generate colors for each trajectory
    num_trajectories = trajectories_np.shape[0]
    colors = [colormap(i / num_trajectories)[:3] for i in range(num_trajectories)]

    # Draw each trajectory
    for traj_idx in range(num_trajectories):
        traj = trajectories_np[traj_idx]
        
        # Filter out invalid points (0,0,0)
        valid_points_mask = ~np.all(traj == 0, axis=1)
        traj = traj[valid_points_mask]

        if traj.shape[0] == 0:
            continue

        # Keep only the most recent max_len points
        if traj.shape[0] > max_len:
            traj = traj[-max_len:]

        # Draw trajectory lines
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]
        
        # Draw trajectory lines with color gradient over time
        for i in range(len(x)-1):
            alpha = 0.3 + 0.7 * (i / (len(x)-1))  # Transparency increases with time
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 
                   color=colors[traj_idx], alpha=alpha, linewidth=line_width)

        # Draw end point
        ax.scatter(x[-1], y[-1], z[-1], color=colors[traj_idx], s=radius)

        # Draw ground projection
        # Project trajectory onto ground plane (y=0)
        x_proj = x
        y_proj = np.zeros_like(x)  # Set y to 0 for ground projection
        z_proj = z

        # Draw projected trajectory lines with light gray color
        for i in range(len(x_proj)-1):
            alpha = 0.3 + 0.7 * (i / (len(x_proj)-1))  # Transparency increases with time
            ax.plot([x_proj[i], x_proj[i+1]], [y_proj[i], y_proj[i+1]], [z_proj[i], z_proj[i+1]], 
                   color='lightgray', alpha=alpha, linewidth=line_width)

        # Draw projected end point with dark gray color
        ax.scatter(x_proj[-1], y_proj[-1], z_proj[-1], color='darkgray', s=radius)

    # Configure axes and grid visibility
    if not show_axes:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
        if axis_labels:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    # Configure grid visibility
    ax.grid(show_grid)

    # Convert figure to image array if needed
    if show_plot:
        plt.draw()
        plt.pause(0.001)
        
    # Get the RGBA buffer from the figure
    fig = ax.get_figure()
    fig.canvas.draw()

    return ax


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


# --- 3D Visualization ---
# Adapted from DELTA https://github.com/snap-research/DELTA_densetrack3d

def read_data(args):
    with open(args.filepath, "rb") as handle:
        trajs_3d_dict = pickle.load(handle)

    coords = trajs_3d_dict["coords"].astype(np.float32)  # T N 3
    colors = trajs_3d_dict["colors"].astype(np.float32) # N 3, 0->255
    vis = trajs_3d_dict["vis"].astype(np.float32)  # T N

    return coords, vis, colors

def save_viewpoint(vis):
    global global_vidname

    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    save_dir = "./results/viewpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{global_vidname}.json")
    
    o3d.io.write_pinhole_camera_parameters(save_path, camera_params)
    print("Viewpoint saved to", save_path)
    return False  # Returning False ensures the visualizer continues to run



def get_viewpoint(traj3d: np.ndarray, traj2d: np.ndarray, vis: np.ndarray):
    """
    Get visualization viewpoint for 3D trajectories.

    Args:
        traj3d: 3D trajectory data (T, N, 3)
        traj2d: 2D trajectory data (T, N, 2)
        vis: Visibility flags (T, N)
    """
    # Get height from 2D trajectory for color mapping
    height = traj2d.shape[1]  # Use height from 2D trajectory
    
    # Generate colors based on y-coordinate
    cmap = plt.get_cmap('gist_rainbow')
    colors = np.zeros((traj3d.shape[1], 3))
    
    # Generate color for each point based on y-coordinate
    for i in range(traj3d.shape[1]):
        y_coord = traj2d[0, i, 1]  # Use y-coordinate from first timestep
        color_normalized = cmap(y_coord / height)[:3]
        colors[i] = color_normalized

    # Create point cloud geometry
    list_geometry = []
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(traj3d[0])
    geometry.colors = o3d.utility.Vector3dVector(colors)
    list_geometry.append(geometry)

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # Add geometries
    for geo in list_geometry:
        vis.add_geometry(geo)

    # Register callback for saving viewpoint
    vis.register_key_callback(ord("S"), save_viewpoint)

    # Set rendering options
    vis.poll_events()
    vis.get_render_option().point_size = 6
    vis.run()
    vis.destroy_window()

def load_viewpoint(vis, filename="default.json"):
    # Load the camera parameters from the JSON file
    camera_params = o3d.io.read_pinhole_camera_parameters(filename)
    
    # Apply the loaded camera parameters to the view control of the visualizer
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    
    print(f"Viewpoint loaded from {filename}")



def capture(args):

    camera_params = o3d.io.read_pinhole_camera_parameters(f"./results/viewpoints/{args.video_name}.json")
    intrinsic = camera_params.intrinsic
    window_h, window_w = intrinsic.height, intrinsic.width
    
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(width=window_w, height=window_h)
    visualizer.get_render_option().point_size = 6

    view_control = visualizer.get_view_control()

    visualizer.poll_events()
    visualizer.update_renderer()

    saved_folder = os.path.join("results/open3d_capture", args.video_name)
    os.makedirs(saved_folder, exist_ok=True)
    os.makedirs(os.path.join(saved_folder, "color"), exist_ok=True)
    

    traj, vis, color = read_data(args)
    T, N = traj.shape[:2]
    
    try:
        binary_mask_ori = cv2.imread(args.fg_mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = cv2.resize(binary_mask_ori, (512, 384), cv2.INTER_NEAREST).astype(bool) # NOTE resize to the same number of tracking points, in this case I track every pixels of reso 384x512, thus resize to this
        binary_mask = binary_mask.reshape(-1)
    except:
        binary_mask = np.ones((384,512)).astype(bool).reshape(-1)

    foreground_mask = binary_mask
    background_mask = ~binary_mask

    caro_mask = np.zeros((384, 512)).astype(bool) # NOTE subsample points to draw trajectories, otherwise drawing all trajectires are very hard to see
    caro_mask[::4,::4] = 1 
    binary_mask2 = foreground_mask & caro_mask.reshape(-1)

    caro_mask = np.zeros((384, 512)).astype(bool)
    caro_mask[::10,::10] = 1
    binary_mask3 = background_mask & caro_mask.reshape(-1)

    binary_mask4 = binary_mask2 | binary_mask3
    
    # # NOTE optional, blending color to highlight moving obj, does not necessary
    # cmap = plt.get_cmap('gist_rainbow')
    # traj_len = binary_mask.sum()
    # norm =  matplotlib.colors.Normalize(vmin=-traj_len*0.1, vmax=traj_len*1.1)
    # rainbow_colors = np.asarray(cmap(norm(np.arange(traj_len)))[:, :3]) * 255.0
    # blend_w = 0.3
    # color[binary_mask,:] = color[binary_mask,:] * blend_w + (1-blend_w) * rainbow_colors
    # #############################!SECTION

    list_geometry = []
    for t in range(T):
        if len(list_geometry) > 0:# NOTE remove visualization from previous frame
            for g in list_geometry: 
                visualizer.remove_geometry(g)
            list_geometry = []

        # NOTE draw point cloud
        vis_pc = o3d.geometry.PointCloud()
        vis_pc.points =  o3d.utility.Vector3dVector(traj[t])
        vis_pc.colors = o3d.utility.Vector3dVector(color /255.0)
        list_geometry.append(vis_pc)

        # NOTE draw trajectories
        diff_track = (traj[:t, background_mask] - traj[t:t+1, background_mask]).mean(1) # T , 3 # NOTE compensate for camera motion
        for i in range(max(1, 1), t):
            p1 = traj[i-1, binary_mask4] - diff_track[i-1] # - delta * (65-i) #   + np.array([2/40 * (i-1 + 1), 0, 0])[None]
            p2 = traj[i, binary_mask4]  - diff_track[i] # - delta * (65-i+1) # + np.array([2/40 * (i + 1), 0, 0])[None]

            n_pts = p1.shape[0]
            vertices = np.concatenate([p1, p2], 0)
            lines = np.stack([np.arange(n_pts), np.arange(n_pts)+n_pts], 1)

            cmap = plt.get_cmap('gist_rainbow')

            traj_len = len(p1)
            norm =  matplotlib.colors.Normalize(vmin=-traj_len*0.1, vmax=traj_len*1.1)
            line_colors = np.asarray(cmap(norm(np.arange(traj_len)))[:, :3])
            lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertices),
                lines=o3d.utility.Vector2iVector(lines)
            )
            lineset.colors = o3d.utility.Vector3dVector(line_colors)

            list_geometry.append(lineset)

        for g in list_geometry:
            visualizer.add_geometry(g)

        view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True )
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(os.path.join(saved_folder, "color", f'{t:05}.png'))
        time.sleep(0.01)

    time.sleep(2.0)

    video = []
    for t in range(T):
        img = cv2.imread(os.path.join(saved_folder, "color", f'{t:05}.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.append(img)
    video = np.stack(video, axis=0)

    save_video_path = os.path.join(saved_folder, f"video.mp4")
    media.write_video(save_video_path, video, fps=10)
    print("Video saved to", save_video_path)

    visualizer.destroy_window()
