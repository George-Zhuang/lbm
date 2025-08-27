import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import time

from lbm.models.lbm_online import LBM_online
from lbm.utils.demo_utils import load_video
from lbm.utils.eval_utils import load_config
from lbm.utils.train_utils import fix_random_seeds

def benchmark_args():
    parser = argparse.ArgumentParser(description='LBM Online PyTorch Benchmark')
    parser.add_argument('--config_path', type=str, default='lbm/configs/demo.yaml')
    parser.add_argument('--video_path', type=str, default='data/demo.mp4')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Test iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on (cuda/cpu)')
    args = parser.parse_args()
    return args

def prepare_input_blob(frame, query, collision_dist, stream_dist, vis_mask, mem_mask, last_pos):
    return {
        "frame": frame,
        "queries": query,
        "collision_dist": collision_dist,
        "stream_dist": stream_dist,
        "vis_mask": vis_mask,
        "mem_mask": mem_mask,
        "last_pos": last_pos,
    }

if __name__ == "__main__":
    args = benchmark_args()
    fix_random_seeds(42)
    config = load_config(args)
    
    try:
        # Initialize model
        model = LBM_online(config).cuda().eval()
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['model'], strict=True)
        
        # Load video and query points
        frames, video, query_points = load_video(config.video_path)
        video = video.cuda()
        query_points = query_points.cuda()
        
        # Initialize model with first frame
        first_frame = video[:, 0] # 1 c h w
        model.init(query_points, first_frame)

        # Prepare test data
        test_frame = video[:, 1] # 1 c h w

        # Warmup
        print("Starting warmup...")
        with torch.no_grad():
            for _ in range(args.warmup):
                _, _, _ = model.online_forward(test_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Speed test
        print("Starting speed test...")
        total_time = 0
        with torch.no_grad():
            for i in range(args.iterations):
                start_time = time.time()
                _, _, _ = model.online_forward(test_frame)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                total_time += (end_time - start_time)
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{args.iterations} iterations")
        
        # Calculate average FPS
        avg_time = total_time / args.iterations
        fps = 1.0 / avg_time
        
        print(f"\nTest Results:")
        print(f"Device: cuda")
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.2f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}") 