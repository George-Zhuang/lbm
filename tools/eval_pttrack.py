import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import time

from lbm.models import LBM
from lbm.utils.train_utils import fix_random_seeds, get_dataloaders
from lbm.utils.eval_utils import load_config, Evaluator, compute_tapvid_metrics


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate an point tracking model.')
    parser.add_argument('--config_path', type=str, default='lbm/configs/lbm.yaml', help='Path to the configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt', help='Path to the checkpoint file.')
    parser.add_argument('--eval_dataset', type=str, default='davis', help='Evaluation dataset.', choices=['davis', 'kinetics', 'robotap', 'dynamicreplica', 'pointodyssey'])
    parser.add_argument('--tapvid_root', type=str, default='data/tapvid_davis/tapvid_davis.pkl', help='Path to the tapvid root file.')
    parser.add_argument('--save_dir', type=str, default='output/lbm_evaluate', help='Path to save the evaluation results.')
    parser.add_argument('--validation', type=bool, default=True, help='Whether to use validation set.')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the results.')
    return parser.parse_args()


def evaluate_with_logging(val_dataloader, model, config, args):
    """Evaluate with detailed logging of individual video results"""
    model.eval()
    model.extend_queries = False
    model.transformer.random_mask_ratio = 0

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0
    video_results = []

    print("Starting evaluation...")
    
    for j, (video, trajectory, visibility, query_points_i) in enumerate(val_dataloader):
        # Timer start
        start_time = time.time()
        total_frames += video.shape[1]

        query_points_i = query_points_i.cuda(non_blocking=True)      # (1, N, 3)
        trajectory = trajectory.cuda(non_blocking=True)              # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)              # (1, T, N)
        video = video.cuda(non_blocking=True)                        # (1, T, 3, H, W)
        B, T, N, _ = trajectory.shape
        _, _, _, H, W = video.shape
        device = video.device

        # Change (t, y, x) to (t, x, y)
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)

        out = model.inference(video, queries)
        pred_trajectory = out["points"]                # (1, T, N, 2)
        pred_visibility = out["visibility"]            # (1, T, N)

        # Timer end
        total_time += time.time() - start_time

        # Convert to numpy for metric computation
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy() 
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.clone().permute(0, 2, 1, 3).cpu().numpy() # (1, N, T, 2)

        # Compute metrics
        if args.eval_dataset in ["davis", "kinetics", "robotap"]:
            out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
            
            # Get video name if available
            video_name = f"video_{j}" if hasattr(val_dataloader.dataset, 'video_names') and j < len(val_dataloader.dataset.video_names) else f"video_{j}"
            
            # Calculate individual video metrics
            delta_avg = out_metrics['average_pts_within_thresh'][0] * 100
            aj = out_metrics['average_jaccard'][0] * 100
            oa = out_metrics['occlusion_accuracy'][0] * 100
            
            # Log individual video result
            print(f"Video {j}/{len(val_dataloader)} ({video_name}): "
                f"delta_avg={delta_avg:.2f}, AJ={aj:.2f}, OA={oa:.2f}")
            
            video_results.append({
                'video_name': video_name,
                'video_idx': j,
                'delta_avg': delta_avg,
                'aj': aj,
                'oa': oa,
                'evaluation_time': time.time() - start_time
            })
            
            evaluator.update(out_metrics)

        elif args.eval_dataset in ["dynamicreplica", "pointodyssey"]:
            out_metrics = compute_dynamicreplica_metrics(visibility, trajectory, pred_trajectory, video)
            video_name = f"video_{j}" if hasattr(val_dataloader.dataset, 'video_names') and j < len(val_dataloader.dataset.video_names) else f"video_{j}"
            survival = out_metrics['survival']
            accuracy = out_metrics['accuracy']
            accuracy_vis = out_metrics['accuracy_vis']
            accuracy_occ = out_metrics['accuracy_occ']
            
            print(f"Video {j}/{len(val_dataloader)} ({video_name}): "
                f"survival={survival:.2f}, accuracy={accuracy:.2f}, accuracy_vis={accuracy_vis:.2f}, accuracy_occ={accuracy_occ:.2f}")
            
            video_results.append({
                'video_name': video_name,
                'video_idx': j,
                'survival': survival,
                'accuracy': accuracy,
                'accuracy_vis': accuracy_vis,
                'accuracy_occ': accuracy_occ,
                'evaluation_time': time.time() - start_time
            })
            evaluator.update(out_metrics)

        
    fps = total_frames / (total_time + 1e-6)
    print(f"Evaluation FPS: {fps:.2f}")

    results = evaluator.get_results()
    if args.eval_dataset in ["davis", "kinetics", "robotap"]:
        delta_avg = results["delta_avg"]
        aj = results["aj"]
        oa = results["oa"]

        print("=" * 50)
        print("FINAL EVALUATION RESULTS:")
        print(f"delta_avg: {delta_avg:.2f}")
        print(f"delta_1: {results['delta_1']:.2f}")
        print(f"delta_2: {results['delta_2']:.2f}")
        print(f"delta_4: {results['delta_4']:.2f}")
        print(f"delta_8: {results['delta_8']:.2f}")
        print(f"delta_16: {results['delta_16']:.2f}")
        print(f"AJ: {aj:.2f}")
        print(f"OA: {oa:.2f}")
        print(f"Total evaluation time: {total_time:.2f}s")
        print(f"Total videos evaluated: {len(video_results)}")
        print("=" * 50)

    elif args.eval_dataset in ["dynamicreplica", "pointodyssey"]:
        survival = results["survival"]
        accuracy = results["accuracy"]
        accuracy_vis = results["accuracy_vis"]
        accuracy_occ = results["accuracy_occ"]

        print("=" * 50)
        print("FINAL EVALUATION RESULTS:")
        print(f"survival: {survival:.2f}")
        print(f"accuracy: {accuracy:.2f}")
        print(f"accuracy_vis: {accuracy_vis:.2f}")
        print(f"accuracy_occ: {accuracy_occ:.2f}")
        print(f"Total evaluation time: {total_time:.2f}s")
        print(f"Total videos evaluated: {len(video_results)}")
        print("=" * 50)

    return results, video_results, total_time


def main():
    args = parse_arguments()
    fix_random_seeds(42)
    config = load_config(args)

    print("Starting single-process evaluation")
    print(f"Arguments: {vars(args)}")

    # Initialize tracking model
    print("Initializing model...")
    model = LBM(config).cuda().eval()
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=True)

    # Get evaluation dataset
    print("Loading dataset...")
    _, val_dataloader = get_dataloaders(config)
    config.val_vis_delta = config.dataset_settings[args.eval_dataset]['val_vis_delta']
    config.val_memory_size = config.dataset_settings[args.eval_dataset]['val_memory_size']
    
    print(f"Dataset size: {len(val_dataloader.dataset)} videos")
    
    # Set model parameters
    model.visibility_threshold = config.val_vis_delta
    model.set_memory_size(config.val_memory_size)

    # Evaluate with logging
    start_time = time.time()
    with torch.no_grad():
        results, video_results, eval_time = evaluate_with_logging(val_dataloader, model, config, args)
    total_time = time.time() - start_time

    # Save results to file
    results_file = f"{args.save_dir}/{args.eval_dataset}/eval_results.log"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        if args.eval_dataset in ["davis", "kinetics", "robotap"]:
            delta_avg = results["delta_avg"]
            aj = results["aj"]
            oa = results["oa"]
            f.write("FINAL EVALUATION RESULTS:\n")
            f.write(f"delta_avg: {delta_avg:.2f}\n")
            f.write(f"AJ: {aj:.2f}\n")
            f.write(f"OA: {oa:.2f}\n")
            f.write(f"Total evaluation time: {total_time:.2f}s\n")
            f.write(f"Total videos evaluated: {len(video_results)}\n")
            f.write(f"Number of workers: 1 (single-process)\n")
            f.write(f"Dataset: {args.eval_dataset}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write("\nINDIVIDUAL VIDEO RESULTS:\n")
            for video_result in video_results:
                f.write(f"{video_result['video_name']}: "
                    f"delta_avg={video_result['delta_avg']:.2f}, "
                    f"AJ={video_result['aj']:.2f}, "
                    f"OA={video_result['oa']:.2f}\n")
        
        elif args.eval_dataset in ["dynamicreplica", "pointodyssey"]:
            survival = results["survival"]
            accuracy = results["accuracy"]
            accuracy_vis = results["accuracy_vis"]
            accuracy_occ = results["accuracy_occ"]
            
            f.write("FINAL EVALUATION RESULTS:\n")
            f.write(f"survival: {survival:.2f}\n")
            f.write(f"accuracy: {accuracy:.2f}\n")
            f.write(f"accuracy_vis: {accuracy_vis:.2f}\n")
            f.write(f"accuracy_occ: {accuracy_occ:.2f}\n")
            f.write(f"Total evaluation time: {total_time:.2f}s\n")
            f.write(f"Total videos evaluated: {len(video_results)}\n")
            f.write(f"Number of workers: 1 (single-process)\n")
            f.write(f"Dataset: {args.eval_dataset}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write("\nINDIVIDUAL VIDEO RESULTS:\n")
            for video_result in video_results:
                f.write(f"{video_result['video_name']}: "
                    f"survival={video_result['survival']:.2f}, "
                    f"accuracy={video_result['accuracy']:.2f}, "
                    f"accuracy_vis={video_result['accuracy_vis']:.2f}, "
                    f"accuracy_occ={video_result['accuracy_occ']:.2f}\n")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
