import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
import random
import argparse
import onnxruntime as ort 
import onnx
import numpy as np

from lbm.models.lbm_online import LBM_export
from lbm.utils.demo_utils import load_video
from lbm.utils.vis_utils import draw_trajectory


random.seed(42)

def demo_args():
    parser = argparse.ArgumentParser(description='LBM Online ONNX Demo: Point Tracking')
    parser.add_argument('--video_path', type=str, default='data/demo.mp4')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.onnx')
    parser.add_argument('--save_dir', type=str, default='tmp/output_pttrack_onnx')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = demo_args()
    os.makedirs(args.save_dir, exist_ok=True)

    session = ort.InferenceSession(args.checkpoint_path)
    
    # Print model input information
    print("Model Input Information:")
    for input in session.get_inputs():
        print(f"Input Name: {input.name}")
        print(f"Input Shape: {input.shape}")
        print(f"Input Type: {input.type}")
        print("---")
    
    # Use onnx package to view more detailed model information
    model = onnx.load(args.checkpoint_path)
    print("\nONNX Model Details:")
    print(f"Model Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print("\nModel Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}")
        print(f"Type: {input.type.tensor_type.elem_type}")
        print(f"Shape: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
        print("---")

    with torch.no_grad():
        frames, video, query_points = load_video(args.video_path)
        trajectories = query_points.unsqueeze(1) # n 1 2
        first_frame = video[:, 0] # 1 c h w
        H, W = first_frame.shape[2:]
        first_frame_feat, _, _, _, _, _, _ = session.run(
            ['coord_pred', 'vis_pred', 'f_t', 'collision_dist_new', 'stream_dist_new', 'vis_mask_new', 'mem_mask_new'],
            {
                "frame": first_frame.numpy(),
                "queries": np.zeros((1, 1, 256), dtype=np.float32),
                "collision_dist": np.zeros((1, 1, 12, 256), dtype=np.float32),
                "stream_dist": np.zeros((1, 1, 12, 256), dtype=np.float32),
                "vis_mask": np.zeros((1, 1, 12), dtype=np.bool_),
                "mem_mask": np.zeros((1, 1, 12), dtype=np.bool_),
            }
        )
        query, collision_dist, stream_dist, vis_mask, mem_mask = \
            LBM_export.init(query_points, first_frame_feat, (H, W))

        for t in range(1, video.shape[1]):
            video_frame = video[:, t] # 1 c h w
            f_t, coord_pred, vis_pred, collision_dist, stream_dist, vis_mask, \
                mem_mask = session.run(
                        None,
                        {
                            "frame": video_frame.numpy(),
                            "queries": query.numpy(),
                            "collision_dist": collision_dist.numpy(),
                            "stream_dist": stream_dist.numpy(),
                            "vis_mask": vis_mask.numpy(),
                            "mem_mask": mem_mask.numpy(),
                        }
                    )
            trajectories = torch.cat((trajectories, coord_pred.unsqueeze(1)), dim=1)
            vis_frame, trajectories = draw_trajectory(frames[t], trajectories)
            cv2.imwrite(f"{args.save_dir}/{t:06d}.png", vis_frame)
            cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("vis", 960, 540)
            cv2.imshow("vis", vis_frame)
            cv2.waitKey(1)
            
