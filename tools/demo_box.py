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
from lbm.utils.vis_utils import draw_points, draw_boxes
from ultralytics import YOLOE


def demo_args():
    parser = argparse.ArgumentParser(description='LBM Online Demo: Object Tracking')
    parser.add_argument('--config_path', type=str, default='lbm/configs/demo.yaml')
    parser.add_argument('--video_path', type=str, default='data/demo.mp4')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/lbm.pt')
    parser.add_argument('--detector_checkpoint_path', type=str, default='checkpoints/yoloe-11l-seg.pt')
    parser.add_argument('--object_points', type=int, default=20)
    parser.add_argument('--prompt', type=str, default='bird')
    parser.add_argument('--init_score_thr', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='tmp/output_objtrack')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = demo_args()
    fix_random_seeds(42)
    config = load_config(args)
    os.makedirs(config.save_dir, exist_ok=True)

    # init detector
    detector = YOLOE(config.detector_checkpoint_path)
    prompts = config.prompt.split(",")
    detector.set_classes(prompts, detector.get_text_pe(prompts))
    # init tracking model
    model = LBM_online(config).cuda().eval()
    checkpoint = torch.load(config.checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=True)

    with torch.no_grad():
        frames, video, _ = load_video(config.video_path, click_query=False)
        video = video.cuda()

        for t in range(1, video.shape[1]):
            video_frame = video[:, t] # 1 c h w
            if t % config.detect_every_n_frame == 0:
                det_frame = video_frame[0].permute(1, 2, 0).cpu().numpy() # c h w
                det_frame = cv2.cvtColor(det_frame, cv2.COLOR_RGB2BGR)
                results = detector(det_frame, conf=config.detect_threshold)
                bboxes = results[0].boxes.xyxy.cpu().numpy() # xyxy format
                scores = results[0].boxes.conf.cpu().numpy() # confidence scores
                labels = results[0].boxes.cls.cpu().numpy().astype(int)
                coord, vis, pred_track_instances = model.online_forward_obj(video_frame, bboxes, scores, labels)
                vis_frame = draw_boxes(frames[t], pred_track_instances)
            else:
                coord, vis, pred_track_instances = model.online_forward_obj(video_frame, None, None, None)
                vis_frame = frames[t]
            vis_frame = draw_points(vis_frame, coord, vis)

            cv2.imwrite(f"{config.save_dir}/{t:06d}.png", vis_frame)
            cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("vis", 960, 540)
            cv2.imshow("vis", vis_frame)
            cv2.waitKey(1)
            
