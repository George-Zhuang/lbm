from typing import Optional, Tuple, List, Dict, Any
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import imageio.v3 as iio

from torch.utils.data import Dataset


class ObjTrackDataset(Dataset):
    """
    Dataset class for object tracking benchmarks like BFT.

    Loads video frames and associated metadata.
    """
    def __init__(
        self,
        eval_dataset: str = "bft",
        root_dir: str = 'data/bft',
        anno_file: str = 'annotations/test_v1.6.json',
        data_dir: str = 'test',
        mode: str = 'eval',
        detres_root: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initializes the dataset.

        Args:
            eval_dataset: Name of the dataset (e.g., "BFT").
            root_dir: Root directory of the dataset.
            anno_file: Path to the annotation file relative to root_dir.
            data_dir: Directory containing image data relative to root_dir.
            mode: Dataset mode (e.g., 'train', 'eval').
            detres_root: Root directory for detection results (optional).
            size: Target size (H, W) to resize images to. If None, original size is kept.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, data_dir)
        self.size = size
        self.detres_root = detres_root # Keep track, though not used in current methods
        self.mode = mode # Keep track, though not used in current methods

        anno_path = os.path.join(root_dir, anno_file)
        print(f'Loading {eval_dataset.upper()} dataset annotations from {anno_path}.')
        try:
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Annotation file not found at {anno_path}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {anno_path}")
            raise

        self.videos = anno_data.get('videos', [])
        self.images_all = anno_data.get('images', []) # Keep original list temporarily

        video_num = len(self.videos)
        image_num = len(self.images_all)
        print(f"Loaded {eval_dataset.upper()} dataset with {video_num} videos and {image_num} images.")

        if image_num > 0:
            # Use a placeholder if images_all is empty to avoid index error
            first_image_info = self.images_all[0] if self.images_all else {'file_name': 'N/A'}
            sample_image_path = os.path.join(self.image_dir, first_image_info['file_name'])
            print(f"Example image path: {sample_image_path}")
            # Check if the sample image exists
            if first_image_info['file_name'] != 'N/A' and not os.path.exists(sample_image_path):
                 print(f"WARNING: Sample image path does not exist: {sample_image_path}")
        else:
            print("WARNING: No images found in the annotation data.")


        # Group images by video_id for efficient lookup
        self.images_by_video: Dict[int, List[Dict[str, Any]]] = {}
        for image_info in self.images_all:
            video_id = image_info.get('video_id')
            if video_id is not None:
                if video_id not in self.images_by_video:
                    self.images_by_video[video_id] = []
                self.images_by_video[video_id].append(image_info)
            else:
                print(f"WARNING: Image missing 'video_id': {image_info.get('file_name', 'N/A')}")

        # Sort images within each video by frame_id or index if available
        for video_id in self.images_by_video:
             # Check if the list is not empty before accessing the first element
             if self.images_by_video[video_id]:
                 first_img = self.images_by_video[video_id][0]
                 # Prefer 'frame_id', fallback to 'index', otherwise keep original order
                 sort_key = 'frame_id' if 'frame_id' in first_img else \
                            ('index' if 'index' in first_img else None)
                 if sort_key:
                     self.images_by_video[video_id].sort(key=lambda img: img.get(sort_key, float('inf')))
             else:
                 print(f"WARNING: Video ID {video_id} has an empty image list after grouping.")


    def __len__(self) -> int:
        """Returns the number of videos in the dataset."""
        return len(self.videos)


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves data for a single video.

        Args:
            idx: Index of the video.

        Returns:
            A dictionary containing video information and loaded images.
        """
        video_meta = self.videos[idx]
        video_id = video_meta['id']

        # Default frame_range to 1 if not specified
        frame_range = video_meta.get('frame_range', 1)

        video_info = {
            'id': video_id,
            'width': video_meta['width'],
            'height': video_meta['height'],
            'neg_category_ids': video_meta.get('neg_category_ids', []),
            'not_exhaustive_category_ids': video_meta.get('not_exhaustive_category_ids', []),
            'name': video_meta.get('name', f'video_{video_id}'),
            'frame_range': frame_range,
        }

        images_info = self.images_by_video.get(video_id, [])
        if not images_info:
             print(f"WARNING: No images found for video ID {video_id}")
             # Return minimal info or raise error depending on desired behavior
             return {'video_info': video_info, 'images_info': [], 'images': torch.empty(0)}


        # Load images using imageio
        image_paths = [os.path.join(self.image_dir, img['file_name']) for img in images_info]
        images_list = []
        try:
            # imageio reads images in RGB order by default
            for path in image_paths:
                 try:
                     images_list.append(iio.imread(path))
                 except FileNotFoundError:
                     print(f"ERROR: Image file not found: {path}. Skipping this image for video ID {video_id}.")
                 except Exception as e:
                     print(f"ERROR: Error loading image {path}: {e}. Skipping this image for video ID {video_id}.")

        except Exception as e:
            # Catch potential errors during path joining or list comprehension itself
            print(f"ERROR: An unexpected error occurred during image path processing for video ID {video_id}: {e}")
            raise # Reraise critical errors

        if not images_list:
             print(f"WARNING: Could not load any valid images for video ID {video_id}")
             images_tensor = torch.empty(0)
        else:
            try:
                # Stack images into a single NumPy array (T, H, W, C)
                images_np = np.stack(images_list, axis=0)
                # Convert to PyTorch tensor and permute to (T, C, H, W)
                images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float()

                # Resize if necessary
                if self.size:
                    # Ensure tensor is float for interpolation
                    images_tensor = F.interpolate(
                        images_tensor,
                        size=self.size,
                        mode='bilinear',
                        align_corners=False
                    )
            except Exception as e:
                 print(f"ERROR: Error processing images for video ID {video_id} after loading (stacking, converting, resizing): {e}")
                 # Decide how to handle: return empty, raise error, etc.
                 # Returning empty tensor for now to avoid crashing the whole process if one video fails
                 images_tensor = torch.empty(0)
                 images_info = [] # Clear corresponding info if tensor is invalid


        data_info = {
            'video_info': video_info,
            'images_info': images_info, # Metadata for each image
            'images': images_tensor,    # Tensor of image data (T, C, H, W)
        }

        return data_info
