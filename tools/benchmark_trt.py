import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import pycuda.driver as cuda 
import torch.nn.functional as F
import time

from lbm.utils.trt_utils import TRTInference
from lbm.models.lbm_online import LBM_export
from lbm.utils.demo_utils import load_video

def benchmark_args():
    parser = argparse.ArgumentParser(description='LBM Online TensorRT Benchmark')
    parser.add_argument('--video_path', type=str, default='data/demo.mp4')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.engine')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iterations', type=int, default=100, help='Test iterations')
    parser.add_argument('--max_point_num', type=int, default=100, help='Maximum number of points')
    args = parser.parse_args()
    return args

def prepare_input_blob(frame, query, collision_dist, stream_dist, vis_mask, mem_mask, last_pos):
    return {
        "frame": np.ascontiguousarray(frame),
        "queries": np.ascontiguousarray(query),
        "collision_dist": np.ascontiguousarray(collision_dist),
        "stream_dist": np.ascontiguousarray(stream_dist),
        "vis_mask": np.ascontiguousarray(vis_mask),
        "mem_mask": np.ascontiguousarray(mem_mask),
        "last_pos": np.ascontiguousarray(last_pos),
    }

if __name__ == "__main__":
    args = benchmark_args()
    
    cuda.init()
    device_ctx = cuda.Device(0).make_context()

    try:
        # Initialize model
        model = TRTInference(
            args.checkpoint_path, 
            backend='cuda', 
            max_point_num=args.max_point_num
        )
        
        # Load video and query points
        frames, video, query_points = load_video(args.video_path)
        N = query_points.shape[0]
        H, W = video.shape[-2:]
        
        # Resize video
        video_resized = F.interpolate(
            video[0], 
            size=(384, 512), 
            mode='bilinear', 
            align_corners=False
        )

        # Prepare initial input
        first_frame = video_resized[0]
        init_blob = prepare_input_blob(
            first_frame.numpy(),
            np.zeros((1, 1, 256), dtype=np.float32),
            np.zeros((1, 1, 12, 256), dtype=np.float32),
            np.zeros((1, 1, 12, 256), dtype=np.float32),
            np.zeros((1, 1, 12), dtype=np.bool_),
            np.zeros((1, 1, 12), dtype=np.bool_),
            np.zeros((1, 2), dtype=np.float32)
        )
        
        # Get first frame features
        output = model(init_blob)
        first_frame_feat = output['f_t'].reshape(1, 256, 96, 128)
        query, collision_dist, stream_dist, vis_mask, mem_mask, last_pos = LBM_export.init(
            query_points.numpy(), 
            first_frame_feat, 
            (384, 512),
            memory_size=12
        )

        # Prepare test data
        test_frame = video_resized[1:2]
        test_blob = prepare_input_blob(
            test_frame,
            query.numpy(),
            collision_dist,
            stream_dist,
            vis_mask,
            mem_mask,
            last_pos
        )

        # Warmup
        print("Starting warmup...")
        for _ in range(args.warmup):
            _ = model(test_blob)
        model.synchronize()
        
        # Speed test
        print("Starting speed test...")
        total_time = 0
        for i in range(args.iterations):
            start_time = time.time()
            _ = model(test_blob)
            model.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{args.iterations} iterations")
        
        # Calculate average FPS
        avg_time = total_time / args.iterations
        fps = 1.0 / avg_time
        
        print(f"\nTest Results:")
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.2f}")

    finally:
        device_ctx.pop() 