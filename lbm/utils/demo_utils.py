import cv2
import torch
import os
import glob
import numpy as np


def load_video(video_path, click_query=False, grid_query=False, grid_size=10):
    frames = []
    first_frame = None
    query_points = []

    if os.path.isdir(video_path):
        # Handle directory of images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(video_path, ext)))
        
        if not image_paths:
            raise IOError(f"No image files found in directory: {video_path}")
            
        image_paths.sort() # Ensure correct order

        # Read the first frame for query selection
        first_frame = cv2.imread(image_paths[0])
        if first_frame is None:
            raise IOError(f"Unable to read the first image: {image_paths[0]}")

        # Load all frames
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Unable to read image {img_path}. Skipping.")
                continue
            frames.append(frame)
        
        if not frames:
             raise IOError(f"Could not read any valid images from directory: {video_path}")

    elif os.path.isfile(video_path):
        # Handle video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")

        # Read the first frame for query selection
        ret, first_frame_vid = cap.read()
        if not ret:
            cap.release()
            raise IOError(f"Unable to read the first frame from video: {video_path}")
        first_frame = first_frame_vid.copy() # Keep a copy for display
        cap.release() # Release after reading the first frame

        # Reopen the video and load all frames
        cap_full = cv2.VideoCapture(video_path)
        if not cap_full.isOpened():
             raise IOError(f"Unable to reopen video file: {video_path}")
        while True:
            ret, frame = cap_full.read()
            if not ret:
                break
            frames.append(frame)
        cap_full.release()
        
        if not frames:
             raise IOError(f"Could not read any frames from video: {video_path}")

    else:
        raise FileNotFoundError(f"Path is not a valid file or directory: {video_path}")

    # --- Query Point Selection (Common Logic) ---
    if click_query and first_frame is not None:
        display_frame = first_frame.copy() # Work on a copy for drawing
        def click_event(event, x, y, flags, param):
            nonlocal query_points, display_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                query_points.append((x, y))
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("First Frame - Click to add points, Press 'q' to finish", display_frame)

        cv2.imshow("First Frame - Click to add points, Press 'q' to finish", display_frame)
        cv2.setMouseCallback("First Frame - Click to add points, Press 'q' to finish", click_event)
        print("Click on the frame to add query points. Press 'q' when finished.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # Add a small delay to prevent high CPU usage if needed
            # cv2.waitKey(20) 

        cv2.destroyWindow("First Frame - Click to add points, Press 'q' to finish")
    # --- End Query Point Selection ---

    # --- Grid Query Point Selection ---
    if grid_query and first_frame is not None:
        height, width = first_frame.shape[:2]
        
        step_x = width / (grid_size + 1)
        step_y = height / (grid_size + 1)
        
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = int(i * step_x)
                y = int(j * step_y)
                query_points.append((x, y))
        
        print(f"Generated {len(query_points)} grid query points ({grid_size}x{grid_size})")

    # Convert query_points to a tensor.
    query_points_tensor = torch.tensor(query_points, dtype=torch.float32) if query_points else torch.zeros((0, 2), dtype=torch.float32)

    # Convert frames list (numpy arrays BGR HWC) to video tensor (1 T C H W)
    # Ensure all frames have the same shape
    if not frames:
        # Handle case where no frames were loaded (e.g., empty dir or bad video)
        # Returning empty tensors might be appropriate depending on downstream usage
        print("Warning: No frames loaded.")
        return [], torch.empty((0, 0, 0, 0, 0)), query_points_tensor

    # Check shape consistency
    first_shape = frames[0].shape
    for i, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(f"Inconsistent frame shapes detected. Frame 0: {first_shape}, Frame {i}: {frame.shape}")

    # Convert BGR to RGB and stack
    video_np = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], axis=0) # T H W C
    video = torch.from_numpy(video_np).float()
    video = video.permute(0, 3, 1, 2) # T C H W
    video = video.unsqueeze(0) # 1 T C H W

    # Return original numpy frames (BGR), processed video tensor (RGB), and query points tensor
    return frames, video, query_points_tensor
