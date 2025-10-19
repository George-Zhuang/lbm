import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
import ray
import time

from lbm.models import LBM
from lbm.utils.train_utils import fix_random_seeds, get_dataloaders
from lbm.utils.eval_utils import load_config, Evaluator, compute_tapvid_metrics


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate an point tracking model with multi-process support.')
    parser.add_argument('--config_path', type=str, default='lbm/configs/lbm.yaml', help='Path to the configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt', help='Path to the checkpoint file.')
    parser.add_argument('--eval_dataset', type=str, default='kinetics', help='Evaluation dataset.', choices=['davis', 'kinetics', 'robotap'])
    parser.add_argument('--tapvid_root', type=str, default='data/tapvid_kinetics', help='Path to the tapvid root file.')
    parser.add_argument('--save_dir', type=str, default='output/lbm_evaluate', help='Path to save the evaluation results.')
    parser.add_argument('--validation', type=bool, default=True, help='Whether to use validation set.')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the results.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes. Defaults to number of available GPUs.')
    return parser.parse_args()


def split_chunks(lst, M):
    """
    Split the list lst into M sublists as evenly as possible.
    """
    n = len(lst)
    avg = n // M
    remainder = n % M
    chunks = []
    start = 0
    for i in range(M):
        end = start + avg + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


@ray.remote(num_gpus=1)
class EvaluatorWorker:
    def __init__(self, worker_id, config):
        self.worker_id = worker_id
        self.config = config
        
        # Initialize model
        self.setup_model()
        
    def setup_model(self):
        """Initialize model and data loader"""
        try:
            # Set random seed
            fix_random_seeds(42 + self.worker_id)
            
            # Initialize model
            self.model = LBM(self.config).cuda().eval()
            checkpoint = torch.load(self.config.checkpoint_path, map_location='cuda')
            self.model.load_state_dict(checkpoint['model'], strict=True)
            
            # Get data loader
            _, self.val_dataloader = get_dataloaders(self.config)
            self.config.val_vis_delta = self.config.dataset_settings[self.config.eval_dataset]['val_vis_delta']
            self.config.val_memory_size = self.config.dataset_settings[self.config.eval_dataset]['val_memory_size']
            
        except Exception as e:
            raise
    
    def evaluate_chunk(self, chunk_indices):
        """Evaluate data chunk"""
        try:
            # Create sub data loader
            chunk_dataloader = self.create_chunk_dataloader(chunk_indices)
            
            # Set model parameters
            self.model.visibility_threshold = self.config.val_vis_delta
            self.model.set_memory_size(self.config.val_memory_size)
            
            # Evaluate each video individually and collect results
            video_results = []
            total_delta_avg = 0
            total_aj = 0
            total_oa = 0
            
            for video_idx, (video, trajectory, visibility, query_points_i) in enumerate(chunk_dataloader):
                # Get video name if available
                video_name = f"video_{chunk_indices[video_idx]}" if hasattr(chunk_dataloader.dataset, 'video_names') else f"video_{chunk_indices[video_idx]}"
                
                # Evaluate single video
                start_time = time.time()
                with torch.no_grad():
                    single_result = self.evaluate_single_video(video, trajectory, visibility, query_points_i)
                end_time = time.time()
                evaluation_time = end_time - start_time
                
                video_results.append({
                    'video_name': video_name,
                    'video_idx': chunk_indices[video_idx],
                    'delta_avg': single_result['delta_avg'],
                    'aj': single_result['aj'],
                    'oa': single_result['oa'],
                    'evaluation_time': evaluation_time
                })
                
                total_delta_avg += single_result['delta_avg']
                total_aj += single_result['aj']
                total_oa += single_result['oa']
            
            # Calculate average results
            num_videos = len(video_results)
            avg_delta_avg = total_delta_avg / num_videos
            avg_aj = total_aj / num_videos
            avg_oa = total_oa / num_videos
            
            return {
                'worker_id': self.worker_id,
                'delta_avg': avg_delta_avg,
                'aj': avg_aj,
                'oa': avg_oa,
                'num_videos': num_videos,
                'video_results': video_results
            }
            
        except Exception as e:
            raise
    
    def evaluate_single_video(self, video, trajectory, visibility, query_points_i):
        """Evaluate a single video"""
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

        out = self.model.inference(video, queries)
        pred_trajectory = out["points"]                # (1, T, N, 2)
        pred_visibility = out["visibility"]            # (1, T, N)

        # Convert to numpy for metric computation
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

        # Compute metrics
        from lbm.utils.eval_utils import compute_tapvid_metrics
        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
        
        delta_avg = out_metrics['average_pts_within_thresh'][0] * 100
        aj = out_metrics['average_jaccard'][0] * 100
        oa = out_metrics['occlusion_accuracy'][0] * 100
        
        return {
            'delta_avg': delta_avg,
            'aj': aj,
            'oa': oa
        }
    
    def create_chunk_dataloader(self, chunk_indices):
        """Create data loader for chunk"""
        # Create sub dataset
        chunk_dataset = torch.utils.data.Subset(self.val_dataloader.dataset, chunk_indices)
        
        # Create new data loader
        chunk_dataloader = torch.utils.data.DataLoader(
            chunk_dataset,
            batch_size=self.val_dataloader.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 in Ray worker to avoid nested multiprocessing
            pin_memory=False,  # Set to False in Ray worker
            drop_last=False
        )
        
        return chunk_dataloader


def main():
    args = parse_arguments()
    
    print("Starting multi-process evaluation")
    print(f"Arguments: {vars(args)}")
    
    # Determine number of workers
    if args.num_workers is None:
        num_workers = min(8, torch.cuda.device_count())
    else:
        num_workers = min(args.num_workers, torch.cuda.device_count())
    
    print(f"Using {num_workers} workers with {torch.cuda.device_count()} available GPUs")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    print("Ray initialized")
    
    # Get dataset size
    config = load_config(args)
    _, val_dataloader = get_dataloaders(config)
    dataset_size = len(val_dataloader.dataset)
    print(f"Dataset size: {dataset_size} videos")
    
    # Create data index chunks
    all_indices = list(range(dataset_size))
    chunks = split_chunks(all_indices, num_workers)
    print(f"Split dataset into {len(chunks)} chunks: {[len(c) for c in chunks]}")
    
    # Create workers
    workers = [EvaluatorWorker.remote(i, config) for i in range(num_workers)]
    print(f"Created {num_workers} workers")
    
    # Submit tasks
    start_time = time.time()
    tasks = [worker.evaluate_chunk.remote(chunk) for worker, chunk in zip(workers, chunks)]
    print("Submitted all evaluation tasks")
    
    # Wait for all tasks to complete
    results = ray.get(tasks)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"All tasks completed in {total_time:.2f}s")
    
    # Aggregate results
    total_delta_avg = 0
    total_aj = 0
    total_oa = 0
    total_videos = 0
    all_video_results = []
    
    print("Evaluation results by worker:")
    for result in results:
        print(f"Worker {result['worker_id']}: delta_avg={result['delta_avg']:.2f}, "
              f"AJ={result['aj']:.2f}, OA={result['oa']:.2f}, "
              f"videos={result['num_videos']}")
        
        # Weighted average (based on video count)
        weight = result['num_videos']
        total_delta_avg += result['delta_avg'] * weight
        total_aj += result['aj'] * weight
        total_oa += result['oa'] * weight
        total_videos += weight
        
        # Collect individual video results
        all_video_results.extend(result['video_results'])
    
    # Calculate final results
    final_delta_avg = total_delta_avg / total_videos
    final_aj = total_aj / total_videos
    final_oa = total_oa / total_videos
    
    print("=" * 50)
    print("FINAL EVALUATION RESULTS:")
    print(f"delta_avg: {final_delta_avg:.2f}")
    print(f"AJ: {final_aj:.2f}")
    print(f"OA: {final_oa:.2f}")
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Total videos evaluated: {total_videos}")
    print("=" * 50)
    
    # Save final results to file
    results_file = f"{args.save_dir}/{args.eval_dataset}/eval_results.log"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        f.write("FINAL EVALUATION RESULTS:\n")
        f.write(f"delta_avg: {final_delta_avg:.2f}\n")
        f.write(f"AJ: {final_aj:.2f}\n")
        f.write(f"OA: {final_oa:.2f}\n")
        f.write(f"Total evaluation time: {total_time:.2f}s\n")
        f.write(f"Total videos evaluated: {total_videos}\n")
        f.write(f"Number of workers: {num_workers}\n")
        f.write(f"Dataset: {args.eval_dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write("\nINDIVIDUAL VIDEO RESULTS:\n")
        for video_result in all_video_results:
            f.write(f"{video_result['video_name']}: "
                   f"delta_avg={video_result['delta_avg']:.2f}, "
                   f"AJ={video_result['aj']:.2f}, "
                   f"OA={video_result['oa']:.2f}\n")
    
    print(f"Results saved to {results_file}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
