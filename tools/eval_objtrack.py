import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import pickle
import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from lbm.datasets.objtrack_dataset import ObjTrackDataset
from lbm.models.lbm_online import LBM_online
from lbm.utils.train_utils import fix_random_seeds
from lbm.utils.eval_utils import (
    evaluate_results, 
    load_config, 
    bboxes_resize, 
    get_paths
)
from lbm.utils.vis_utils import save_visualization


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate an object tracking model.')
    parser.add_argument('--config_path', type=str, default='lbm/configs/objtrack_bft.yaml', help='Path to the configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt', help='Path to the checkpoint file.')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the results.')
    return parser.parse_args()

@torch.no_grad()
def evaluate_video(data, model, config, device, vis_dir):
    """Processes a single video sequence for tracking evaluation."""
    video_info = data['video_info']
    video_name = video_info['name'][0]
    print(f"Processing video: {video_name}")

    video_id = int(video_info['id'][0])
    image_ids = [int(img['id']) for img in data['images_info']]
    det_res_dir = Path(config.public_det_path) / video_name
    video = data['images']
    video_size = (int(video_info['height']), int(video_info['width']))
    input_size = config.input_size

    video_results = []
    det_res_files = sorted(os.listdir(det_res_dir)) # Ensure correct frame order

    if len(det_res_files) != video.shape[1]:
         print(f"WARNING: Mismatch in frame count for {video_name}: "
                         f"{len(det_res_files)} detection files vs {video.shape[1]} frames.")
         # Adjust loop range if necessary, or handle error
         num_frames_to_process = min(len(det_res_files), video.shape[1])
    else:
         num_frames_to_process = video.shape[1]


    for frame_idx in range(num_frames_to_process):
        frame = video[:, frame_idx].to(device)
        image_id = image_ids[frame_idx]
        det_res_file = det_res_files[frame_idx]
        det_res_path = det_res_dir / det_res_file

        try:
            with open(det_res_path, 'rb') as f:
                det_res = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: Detection file not found: {det_res_path}")
            continue
        except Exception as e:
            print(f"ERROR: Error loading detection file {det_res_path}: {e}")
            continue

        # Prepare detections for the model
        bboxes_orig = det_res['det_bboxes'][:, :4].cpu().numpy()
        bboxes_resized = bboxes_resize(bboxes_orig, video_size, input_size)
        bboxes_w = bboxes_resized[:, 2] - bboxes_resized[:, 0]
        bboxes_h = bboxes_resized[:, 3] - bboxes_resized[:, 1]
        bboxes_area = bboxes_w * bboxes_h
        bboxes_aspect_ratio = bboxes_w / (bboxes_h + 1e-6)
        valid_mask = bboxes_area > 50 & (bboxes_aspect_ratio < 6) & (bboxes_aspect_ratio > 0.167) 
        bboxes_resized = bboxes_resized[valid_mask]
        scores = det_res['det_bboxes'][:, 4][valid_mask].cpu().numpy()
        labels = det_res['det_labels'][valid_mask].cpu().numpy()

        # Run model inference for the frame
        coord, vis, pred_track_instances = model.online_forward_obj(frame, bboxes_resized, scores, labels)

        # Process tracking results
        bboxes_pred = pred_track_instances['bboxes'].cpu().numpy()
        bboxes_pred_orig = bboxes_resize(bboxes_pred, input_size, video_size)
        scores_pred = pred_track_instances['scores'].cpu().numpy()
        labels_pred = pred_track_instances['labels'].cpu().numpy()
        instances_id = pred_track_instances['instances_id'].cpu().numpy()

        # Optionally visualize
        if config.visualize:
            vis_save_path = vis_dir / video_name / f"{frame_idx:06d}.jpg"
            save_visualization(frame, pred_track_instances, coord, vis, vis_save_path, model)

        # Format results for evaluation
        for box, score, label, track_id in zip(bboxes_pred_orig, scores_pred, labels_pred, instances_id):
            if score >= config.obj_score_thr:
                # Convert bbox from xyxy to xywh
                box[2] -= box[0]
                box[3] -= box[1]
                video_results.append({
                    'video_id': video_id,
                    'image_id': image_id,
                    'bbox': [float(b) for b in box.tolist()],
                    'score': float(score),
                    'category_id': int(label) + 1,
                    'track_id': int(track_id)
                })

    return video_results

def main():
    """Main function to run the evaluation."""
    fix_random_seeds(42)  # Set random seed for reproducibility
    args = parse_arguments()
    config = load_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir, vis_dir, results_file, suffix = get_paths(config)

    # Create output directories
    results_dir.mkdir(parents=True, exist_ok=True)
    if config.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # --- Inference Phase ---
    if not results_file.exists():
        print("Results file not found. Starting inference...")

        # Initialize dataset and dataloader
        dataset = ObjTrackDataset(
            eval_dataset=config.eval_dataset,
            root_dir=config.root_dir,
            anno_file=config.anno_file,
            data_dir=config.data_dir,
            size=config.input_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

        # Initialize tracking model
        model = LBM_online(config).to(device).eval()
        try:
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=True)
            print(f"Loaded model weights from: {config.checkpoint_path}")
        except FileNotFoundError:
            print(f"ERROR: Checkpoint file not found: {config.checkpoint_path}")
            return
        except Exception as e:
            print(f"ERROR: Error loading model weights: {e}")
            return


        all_results = []
        processed_count = 0
        for data in dataloader:
            video_results = evaluate_video(data, model, config, device, vis_dir)
            all_results.extend(video_results)
            processed_count += len(data['video_info']['id']) # Assumes batch_size=1 for videos
            print(f"Progress: {processed_count}/{len(dataset)} videos processed.")

        print(f"Inference finished. Saving results to {results_file}...")
        try:
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=4) # Add indent for readability
            print("Results saved successfully.")
        except IOError as e:
            print(f"ERROR: Error saving results to {results_file}: {e}")
            return

    else:
        print(f"Found existing results file: {results_file}. Skipping inference.")

    # --- Evaluation Phase ---
    print(f"Starting evaluation using results file: {results_file}")
    gt_file = Path(config.root_dir) / config.anno_file
    if not gt_file.exists():
        print(f"ERROR: Ground truth annotation file not found: {gt_file}")
        return

    evaluate_results(
        res_file=str(results_file),
        gt_file=str(gt_file),
        track_name=suffix,
        metric_list=config.metric_list,
        save_folder=str(results_dir), # Save evaluation metrics in results dir
        subset=config.subset,
    )
    print("Evaluation finished.")

if __name__ == "__main__":
    main()