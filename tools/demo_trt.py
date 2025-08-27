import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import random
import argparse
import numpy as np
import pycuda.driver as cuda 
import torch.nn.functional as F

from lbm.utils.trt_utils import TRTInference
from lbm.models.lbm_online import LBM_export
from lbm.utils.demo_utils import load_video
from lbm.utils.vis_utils import draw_trajectory


random.seed(42)

def demo_args():
    parser = argparse.ArgumentParser(description='LBM Online TensorRT Demo: Point Tracking')
    parser.add_argument('--video_path', type=str, default='data/demo.mp4')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.engine')
    parser.add_argument('--save_dir', type=str, default='tmp/output_pttrack_trt')
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
    args = demo_args()
    os.makedirs(args.save_dir, exist_ok=True)

    cuda.init()
    device_ctx = cuda.Device(0).make_context()

    try:
        model = TRTInference(
            args.checkpoint_path, 
            backend='cuda', 
            max_point_num=args.max_point_num
        )
        
        frames, video, query_points = load_video(args.video_path)
        trajectories = query_points.unsqueeze(1) # n 1 2
        trajectories = trajectories.numpy()
        N = query_points.shape[0]
        H, W = video.shape[-2:]
        video_resized = F.interpolate(
            video[0], 
            size=(384, 512), 
            mode='bilinear', 
            align_corners=False
        ) # t c h w

        # Resize frames to match video_resized dimensions
        frames_resized = []
        for frame in frames:
            frame_resized = cv2.resize(frame, (512, 384))
            frames_resized.append(frame_resized)
        frames = np.array(frames_resized)

        query_points[:, 1] = (query_points[:, 1] / H) * 384
        query_points[:, 0] = (query_points[:, 0] / W) * 512
        first_frame = video_resized[0] # 1 c h w

        init_blob = prepare_input_blob(
            first_frame.numpy(),
            np.zeros((1, 1, 256), dtype=np.float32),
            np.zeros((1, 1, 12, 256), dtype=np.float32),
            np.zeros((1, 1, 12, 256), dtype=np.float32),
            np.zeros((1, 1, 12), dtype=np.bool_),
            np.zeros((1, 1, 12), dtype=np.bool_),
            np.zeros((1, 2), dtype=np.float32)
        )
        output = model(init_blob)
        first_frame_feat = output['f_t'].reshape(1, 256, 96, 128)
        query, collision_dist, stream_dist, vis_mask, mem_mask, last_pos = LBM_export.init(
            query_points.numpy(), 
            first_frame_feat, 
            (384, 512),
            memory_size=12
        )

        for t in range(1, video_resized.shape[0]):
            video_frame = video_resized[t:t+1] # 1 c h w
            blob = prepare_input_blob(
                video_frame,
                query.numpy(),
                collision_dist,
                stream_dist,
                vis_mask,
                mem_mask,
                last_pos
            )
            output = model(blob)
            coord_pred = output['coord_pred'][:N*2].reshape(N, 2)
            vis_pred = output['vis_pred'][:N].reshape(N)
            collision_dist = output['collision_dist_new'][:N*12*256].reshape(1, N, 12, 256)
            stream_dist = output['stream_dist_new'][:N*12*256].reshape(1, N, 12, 256)
            vis_mask = output['vis_mask_new'][:N*12].reshape(1, N, 12)
            mem_mask = output['mem_mask_new'][:N*12].reshape(1, N, 12)
            last_pos = output['last_pos_new'][:N*2].reshape(N, 2)
            trajectories = np.concatenate((trajectories, coord_pred.reshape(N, 1, 2)), axis=1)
            vis_frame = draw_trajectory(frames[t], trajectories, radius=3, line_width=2)
            cv2.imwrite(f"{args.save_dir}/{t:06d}.png", vis_frame)
            cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("vis", 512, 384)
            cv2.imshow("vis", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        device_ctx.pop()
        cv2.destroyAllWindows()
        
