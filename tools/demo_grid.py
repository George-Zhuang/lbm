import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
import random
import argparse

from lbm.models.lbm_online import LBM_online
from lbm.utils.train_utils import fix_random_seeds
from lbm.utils.eval_utils import load_config
from lbm.utils.demo_utils import load_video
from lbm.utils.vis_utils import draw_trajectory


random.seed(42)

def demo_args():
    parser = argparse.ArgumentParser(description='LBM Online Demo: Point Tracking')
    parser.add_argument('--config_path', type=str, default='lbm/configs/demo.yaml')
    parser.add_argument('--video_path', type=str, default='data/demo.mp4')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt')
    parser.add_argument('--save_dir', type=str, default='tmp/demo_grid')
    parser.add_argument('--grid_size', type=int, default=50)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = demo_args()
    fix_random_seeds(42)
    config = load_config(args)
    os.makedirs(config.save_dir, exist_ok=True)

    model = LBM_online(config).cuda().eval()
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=True)

    with torch.no_grad():
        frames, video, query_points = load_video(config.video_path, click_query=False, grid_query=True, grid_size=args.grid_size)
        video = video.cuda()
        query_points = query_points.cuda()
        trajectories = query_points.unsqueeze(1) # n 1 2
        first_frame = video[:, 0] # 1 c h w
        model.init(query_points, first_frame)

        for t in range(1, video.shape[1]):
            video_frame = video[:, t] # 1 c h w
            coord, vis, rho, _ = model.online_forward(video_frame)
            trajectories = torch.cat((trajectories.to(coord.device), coord.unsqueeze(1)), dim=1)
            vis_frame, trajectories = draw_trajectory(frames[t], trajectories, vis, rho)
            cv2.imwrite(f"{config.save_dir}/{t:06d}.png", vis_frame)
            cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("vis", 960, 540)
            cv2.imshow("vis", vis_frame)
            cv2.waitKey(1)
            
