from typing import Mapping, Optional
import os
import sys
import time
import yaml
import json
import torch
import einops
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from multiprocessing import freeze_support
from lbm.utils.log_utils import log_eval_metrics


PIXEL_TO_FIXED_METRIC_THRESH = {
    1: 0.01,
    2: 0.04,
    4: 0.16,
    8: 0.64,
    16: 2.56,
}


def get_pointwise_threshold_multiplier(gt_tracks: np.ndarray, intrinsics_params: np.ndarray) -> np.ndarray | float:
    mean_focal_length = np.sqrt(intrinsics_params[..., 0] * intrinsics_params[..., 1] + 1e-12)
    return gt_tracks[..., -1] / mean_focal_length[..., np.newaxis, np.newaxis]

def set_random_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(args):
    """Loads configuration from YAML and merges with command-line arguments."""
    if not os.path.exists(args.config_path):
        print(f"Warning: Config file not found at {args.config_path}. Using default CLI args.")
        return {} # Return empty dict if file not found
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Override config with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return argparse.Namespace(**config)

def print_args(args):
    if args.validation:
        print("====== Validation ======")
        print(f"Evaluating on: {args.eval_dataset}")
        print(f"Memory size: {args.val_memory_size}")
        print(f"Visibility Delta: {args.val_vis_delta}")

    else:
        print("====== Training ======")
        print(f"Evaluating on: {args.eval_dataset}")
        print()

        print(f"Input size: {args.input_size}")
        print(f"N: {args.N}")
        print(f"T: {args.T}")
        print(f"Stride: {args.stride}")
        print(f"Transformer embedding dim: {args.embed_dim}")
        print()

        print(f"Memory size: {args.memory_size}")
        print(f"Random memory mask drop: {args.random_memory_mask_drop}")
        print()

        print(f"Classification loss: {args.lambda_cls}")
        print(f"Visibility loss: {args.lambda_vis}")
        print(f"Offset loss: {args.lambda_off}")
        print(f"Uncertainty loss: {args.lambda_unc}")
        print(f"Reference point uncertainty loss: {args.lambda_ref}")
        print()

        print(f"Epochs: {args.epoch_num}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.wd}")
        print(f"Batch size per GPU: {args.bs}")
        print(f"Using AMP: {args.amp}")
        print(f"Loss after query: {args.loss_after_query}")
        
    print("====== ======= ======\n")

def bboxes_resize(bboxes, src_size, tgt_size):
    """Resizes bounding boxes from source size to target size."""
    src_h, src_w = src_size
    tgt_h, tgt_w = tgt_size
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    bboxes[:, 0] *= scale_x
    bboxes[:, 1] *= scale_y
    bboxes[:, 2] *= scale_x
    bboxes[:, 3] *= scale_y
    return bboxes

def get_paths(config):
    """Constructs necessary paths based on the configuration."""
    # suffix = (f"{config.eval_dataset}_{config.init_score_thr}_{config.obj_score_thr}_"
    #           f"{config.match_score_thr}_{config.object_points}_"
    #           f"{config.memo_tracklet_frames}_{config.memo_point_frames}")
    suffix = f"{config.eval_dataset}"
    results_dir = Path(config.save_res_path)
    vis_dir = Path(config.save_vis_path)
    results_file = results_dir / f"{suffix}.json"
    return results_dir, vis_dir, results_file, suffix

def _reproject_2d3d(trajs_uvd, intrs):

    B, N = trajs_uvd.shape[:2]
    # intrs = sample.intrs
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]

    trajs_3d = torch.zeros((B, N, 3), device=trajs_uvd.device)
    trajs_3d[..., 0] = trajs_uvd[..., 2] * (trajs_uvd[..., 0] - cx[..., None]) / fx[..., None]
    trajs_3d[..., 1] = trajs_uvd[..., 2] * (trajs_uvd[..., 1] - cy[..., None]) / fy[..., None]
    trajs_3d[..., 2] = trajs_uvd[..., 2]

    return trajs_3d

def _project_3d2d(trajs_3d, intrs):

    B, N = trajs_3d.shape[:2]
    # intrs = sample.intrs
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]

    trajs_uvd = torch.zeros((B, N, 3), device=trajs_3d.device)
    trajs_uvd[..., 0] = trajs_3d[..., 0] * fx[..., None] / trajs_3d[..., 2] + cx[..., None]
    trajs_uvd[..., 1] = trajs_3d[..., 1] * fy[..., None] / trajs_3d[..., 2] + cy[..., None]
    trajs_uvd[..., 2] = trajs_3d[..., 2]

    return trajs_uvd


def reproject_2d3d(trajs_uvd, intrs):

    B, T, N = trajs_uvd.shape[:3]

    trajs_3d = _reproject_2d3d(trajs_uvd.reshape(-1, N, 3), intrs.reshape(-1, 3, 3))
    trajs_3d = rearrange(trajs_3d, "(B T) N C -> B T N C", T=T)

    return trajs_3d


# point tracking evaluation
def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics

def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input)

    prod = input * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (1e-6 + denom)
    return mean


def compute_dynamicreplica_metrics(
    gt_visibility,
    gt_tracks,
    pred_tracks,
    video,
):
    '''
    gt_visibility: (B, T, N)
    gt_tracks: (B, T, N, 2)
    pred_tracks: (B, T, N, 2)
    video: (B, T, 3, H, W)
    '''
    B, T, N, _ = gt_tracks.shape
    H, W = video.shape[-2:]
    device = video.device

    out_metrics = {}

    d_vis_sum = d_occ_sum = d_sum_all = 0.0
    thrs = [1, 2, 4, 8, 16]
    sx_ = (W - 1) / 255.0
    sy_ = (H - 1) / 255.0
    sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
    sc_pt = torch.from_numpy(sc_py).float().to(device)
    __, first_visible_inds = torch.max(gt_visibility, dim=1)

    frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(
        B, 1, N
    )
    start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

    for thr in thrs:
        d_ = (
            torch.norm(
                pred_tracks[..., :2] / sc_pt
                - gt_tracks[..., :2] / sc_pt,
                dim=-1,
            )
            < thr
        ).float()  # B,S-1,N
        d_occ = (
            reduce_masked_mean(
                d_, ((~gt_visibility.bool()).float()) * start_tracking_mask.float()
            ).item()
            * 100.0
        )
        d_occ_sum += d_occ
        out_metrics[f"accuracy_occ_{thr}"] = d_occ

        d_vis = (
            reduce_masked_mean(
                d_, gt_visibility.float() * start_tracking_mask.float()
            ).item()
            * 100.0
        )
        d_vis_sum += d_vis
        out_metrics[f"accuracy_vis_{thr}"] = d_vis

        d_all = reduce_masked_mean(d_, start_tracking_mask.float()).item() * 100.0
        d_sum_all += d_all
        out_metrics[f"accuracy_{thr}"] = d_all

    d_occ_avg = d_occ_sum / len(thrs)
    d_vis_avg = d_vis_sum / len(thrs)
    d_all_avg = d_sum_all / len(thrs)

    sur_thr = 50
    dists = torch.norm(
        pred_tracks[..., :2] / sc_pt - gt_tracks[..., :2] / sc_pt,
        dim=-1,
    )  # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * gt_visibility.float()  # B,S,N
    survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
    out_metrics["survival"] = torch.mean(survival).item() * 100.0

    out_metrics["accuracy_occ"] = d_occ_avg
    out_metrics["accuracy_vis"] = d_vis_avg
    out_metrics["accuracy"] = d_all_avg
    return out_metrics

@torch.no_grad()
def evaluate_pttrack(val_dataloader, model, verbose=False):
    model.eval()
    model.extend_queries = False
    model.transformer.random_mask_ratio = 0

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0

    for j, (video, trajectory, visibility, query_points_i) in enumerate(tqdm(val_dataloader, disable=verbose, file=sys.stdout)):
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

        # === === ===
        # From CoTracker
        traj = trajectory.clone()
        query_points = query_points_i.clone().cpu().numpy()
        gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
        # === === ===


        out_metrics = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, "first")
        if verbose:
            print(f"Video {j}/{len(val_dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)
        
    fps = total_frames / (total_time + 1e-6)
    print(f"Evaluation FPS: {fps:.2f}", flush=True)

    results = evaluator.get_results()
    delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    print(f"delta_avg: {results['delta_avg']:.2f}")
    print(f"delta_1: {results['delta_1']:.2f}")
    print(f"delta_2: {results['delta_2']:.2f}")
    print(f"delta_4: {results['delta_4']:.2f}")
    print(f"delta_8: {results['delta_8']:.2f}")
    print(f"delta_16: {results['delta_16']:.2f}")
    print(f"AJ: {results['aj']:.2f}")
    print(f"OA: {results['oa']:.2f}")

    return delta_avg, aj, oa

@torch.no_grad()
def evaluate_pttrack3d(val_dataloader, model, verbose=False):
    model.eval()
    model.extend_queries = False
    model.transformer.random_mask_ratio = 0

    evaluator = Evaluator()
    total_frames = 0
    total_time = 0

    for j, (video, videodepth, trajectory_2d, trajectory_3d, visibility, video_name, query_points_3d, intrinsic_mat) in enumerate(tqdm(val_dataloader, disable=verbose, file=sys.stdout)):
        # Timer start
        start_time = time.time()
        total_frames += video.shape[1]

        query_points_3d = query_points_3d.cuda(non_blocking=True)      # (1, N, 3)
        trajectory_3d = trajectory_3d.cuda(non_blocking=True)          # (1, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)                # (1, T, N)
        video = video.cuda(non_blocking=True)                          # (1, T, 3, H, W)
        videodepth = videodepth.cuda(non_blocking=True)                # (1, T, 1, H, W) in meters
        intrinsic_mat = intrinsic_mat.cuda(non_blocking=True)          # (1, 3, 3)

        device = video.device

        # tracking forward
        queries = query_points_3d.clone().float()
        pred_tracks, pred_visibility = model.inference_3d(video, videodepth, queries, intrinsic_mat) # (1, T, N, 3), (1, T, N)
        
        # Timer end
        total_time += time.time() - start_time

        # tracking backward
        inv_video = torch.flip(video.clone(), dims=[1])
        inv_video_depth = torch.flip(videodepth.clone(), dims=[1])
        inv_queries = query_points_3d.clone()
        inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1
        inv_pred_tracks, inv_pred_visibility = model.inference_3d(inv_video, inv_video_depth, inv_queries, intrinsic_mat, reverse=True)
 
        # tracking results
        time_indices = torch.arange(video.shape[1], device=video.device)[None, :, None]           # (1, T, 1)
        query_times = query_points_3d[:, :, 0:1].long().permute(0, 2, 1)                          # (1, 1, N)
        time_mask = (time_indices < query_times).unsqueeze(-1).repeat(1, 1, 1, 3)                 # (1, T, N, 3)
        pred_tracks[time_mask] = inv_pred_tracks[time_mask]                                       # (1, T, N, 3)
        pred_tracks = pred_tracks.permute(0, 2, 1, 3).cpu().numpy()                               # (1, N, T, 3)
        pred_visibility[time_mask[..., 0]] = inv_pred_visibility[time_mask[..., 0]]               # (1, T, N)
        pred_occluded = torch.logical_not(pred_visibility.clone().permute(0, 2, 1)).cpu().numpy() # (1, N, T)

        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()        # (1, N, T)
        gt_tracks = trajectory_3d.permute(0, 2, 1, 3).cpu().numpy()                               # (1, N, T, 3)

        intrinsics_params = torch.stack([
            intrinsic_mat[0, 0, 0, 0], 
            intrinsic_mat[0, 0, 1, 1], 
            intrinsic_mat[0, 0, 0, 2], 
            intrinsic_mat[0, 0, 1, 2]
        ], dim=-1).cpu().numpy() # (1, 4)

        out_metrics = compute_tapvid3d_metrics(
            gt_occluded, gt_tracks, pred_occluded, pred_tracks, intrinsics_params, 
            query_points=query_points_3d.cpu().numpy(), order="b t n")
        if verbose:
            print(f"Video {j} {video_name}/{len(val_dataloader)}: AJ: {out_metrics['average_jaccard'][0] * 100:.2f}, delta_avg: {out_metrics['average_pts_within_thresh'][0] * 100:.2f}, OA: {out_metrics['occlusion_accuracy'][0] * 100:.2f}", flush=True)
        evaluator.update(out_metrics)
        
    fps = total_frames / (total_time + 1e-6)
    print(f"Evaluation FPS: {fps:.2f}", flush=True)

    results = evaluator.get_results()
    delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    print(f"delta_avg: {results['delta_avg']:.2f}")
    print(f"delta_1: {results['delta_1']:.2f}")
    print(f"delta_2: {results['delta_2']:.2f}")
    print(f"delta_4: {results['delta_4']:.2f}")
    print(f"delta_8: {results['delta_8']:.2f}")
    print(f"delta_16: {results['delta_16']:.2f}")
    print(f"AJ: {results['aj']:.2f}")
    print(f"OA: {results['oa']:.2f}")

    return delta_avg, aj, oa
    

# point tracking evaluation
class Evaluator():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.aj = []
        self.delta_avg = []
        self.oa = []
        self.delta_1 = []
        self.delta_2 = []
        self.delta_4 = []
        self.delta_8 = []
        self.delta_16 = []
        self.cnt = 0

        self.accuracy = []
        self.accuracy_vis = []
        self.accuracy_occ = []
        self.survival = []

    def get_results(self):
        if self.delta_avg:
            return {
                "delta_avg": sum(self.delta_avg) / len(self.delta_avg),
                "delta_1": sum(self.delta_1) / len(self.delta_1),
                "delta_2": sum(self.delta_2) / len(self.delta_2),
                "delta_4": sum(self.delta_4) / len(self.delta_4),
                "delta_8": sum(self.delta_8) / len(self.delta_8),
                "delta_16": sum(self.delta_16) / len(self.delta_16),
                "aj": sum(self.aj) / len(self.aj),
                "oa": sum(self.oa) / len(self.oa),
            }
        elif self.accuracy:
            return {
                "accuracy": sum(self.accuracy) / len(self.accuracy),
                "accuracy_vis": sum(self.accuracy_vis) / len(self.accuracy_vis),
                "accuracy_occ": sum(self.accuracy_occ) / len(self.accuracy_occ),
                "survival": sum(self.survival) / len(self.survival),
            }

    def update(self, out_metrics, verbose=False):
        if "average_jaccard" in out_metrics:
            aj = out_metrics['average_jaccard'][0] * 100
            delta = out_metrics['average_pts_within_thresh'][0] * 100
            delta_1 = out_metrics['pts_within_1'][0] * 100
            delta_2 = out_metrics['pts_within_2'][0] * 100
            delta_4 = out_metrics['pts_within_4'][0] * 100
            delta_8 = out_metrics['pts_within_8'][0] * 100
            delta_16 = out_metrics['pts_within_16'][0] * 100
            oa = out_metrics['occlusion_accuracy'][0] * 100

            if verbose:
                print(f"Video {self.cnt} | AJ: {aj:.2f}, delta_avg: {delta:.2f}, OA: {oa:.2f}")
            self.cnt += 1
            
            self.aj.append(aj)
            self.delta_avg.append(delta)
            self.oa.append(oa)
            self.delta_1.append(delta_1)
            self.delta_2.append(delta_2)
            self.delta_4.append(delta_4)
            self.delta_8.append(delta_8)
            self.delta_16.append(delta_16)
        
        elif "accuracy" in out_metrics:
            accuracy = out_metrics['accuracy'] 
            accuracy_vis = out_metrics['accuracy_vis'] 
            accuracy_occ = out_metrics['accuracy_occ'] 
            survival = out_metrics['survival'] 

            if verbose:
                print(f"Video {self.cnt} | Accuracy: {accuracy:.2f}, Accuracy Vis: {accuracy_vis:.2f}, Accuracy Occ: {accuracy_occ:.2f}, Survival: {survival:.2f}")
            self.cnt += 1

            self.accuracy.append(accuracy)
            self.accuracy_vis.append(accuracy_vis)
            self.accuracy_occ.append(accuracy_occ)
            self.survival.append(survival)

    def report(self):
        if self.delta_avg:
            print(f"Mean AJ: {sum(self.aj) / len(self.aj):.1f}")
            print(f"Mean delta_avg: {sum(self.delta_avg) / len(self.delta_avg):.1f}")
            print(f"Mean delta_1: {sum(self.delta_1) / len(self.delta_1):.1f}")
            print(f"Mean delta_2: {sum(self.delta_2) / len(self.delta_2):.1f}")
            print(f"Mean delta_4: {sum(self.delta_4) / len(self.delta_4):.1f}")
            print(f"Mean delta_8: {sum(self.delta_8) / len(self.delta_8):.1f}")
            print(f"Mean delta_16: {sum(self.delta_16) / len(self.delta_16):.1f}")
            print(f"Mean OA: {sum(self.oa) / len(self.oa):.1f}")
        elif self.accuracy:
            print(f"Mean Accuracy: {sum(self.accuracy) / len(self.accuracy):.1f}")
            print(f"Mean Accuracy Vis: {sum(self.accuracy_vis) / len(self.accuracy_vis):.1f}")
            print(f"Mean Accuracy Occ: {sum(self.accuracy_occ) / len(self.accuracy_occ):.1f}")
            print(f"Mean Survival: {sum(self.survival) / len(self.survival):.1f}")


def create_local_tracks(
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather all points within a threshold distance of ground truth."""
    out_gt_tr = []
    out_gt_occ = []
    out_pr_tr = []
    out_pr_occ = []

    # for each track, find the points that are within a threshold distance of
    # the track's point on the corresponding frame and grab them
    for idx in range(gt_occluded.shape[0]):
        diffs = gt_tracks - gt_tracks[idx : idx + 1]
        is_neighbor = np.sum(np.square(diffs), axis=-1) < thresh * thresh
        is_neighbor = np.reshape(is_neighbor, [-1])

        def grab(x):
            x = np.reshape(x, [-1, x.shape[-1]])
            return x[is_neighbor]  # pylint: disable=cell-var-from-loop

        out_gt_tr.append(grab(gt_tracks))
        out_pr_tr.append(grab(pred_tracks))
        out_gt_occ.append(grab(gt_occluded[..., np.newaxis]))
        out_pr_occ.append(grab(pred_occluded[..., np.newaxis]))

    # need to pad to the longest length before stacking
    largest = np.max([x.shape[0] for x in out_gt_tr])

    def pad(x):
        res = np.zeros([largest, x.shape[-1]], dtype=x.dtype)
        res[: x.shape[0]] = x
        return res

    out_gt_tr = np.stack([pad(x) for x in out_gt_tr])
    out_pr_tr = np.stack([pad(x) for x in out_pr_tr])
    valid = np.stack([pad(np.ones_like(x)) for x in out_gt_occ])[..., 0]
    out_gt_occ = np.stack([pad(x) for x in out_gt_occ])[..., 0]
    out_pr_occ = np.stack([pad(x) for x in out_pr_occ])[..., 0]
    weighting = np.sum((1.0 - gt_occluded), axis=1, keepdims=True) / np.maximum(
        1.0, np.sum((1.0 - out_gt_occ) * valid, axis=1, keepdims=True)
    )

    return out_gt_occ, out_gt_tr, out_pr_occ, out_pr_tr, valid * weighting


# 3D point tracking evaluation
def compute_tapvid3d_metrics(
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    intrinsics_params: np.ndarray,
    get_trackwise_metrics: bool = False,
    scaling: str = "median",
    query_points: Optional[np.ndarray] = None,
    use_fixed_metric_threshold: bool = False,
    local_neighborhood_thresh: Optional[float] = 0.05,
    order: str = "n t",
    return_scaled_pred: bool = False,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.).

    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.


    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.

    Args:
        gt_occluded: A boolean array, generally of shape [b, n, t] or [n, t], where
        t is the number of frames and n is the number of tracks. True indicates
        that the point is occluded. Must be consistent with 'order' parameter, so
        if passing in [t, n] or [b, t, n] instead, order string must reflect
        this!
        gt_tracks: The target points, of shape [b, n, t, 2] or [n, t, 2], unless
        specified otherwise in the order parameter. Each point is in the format
        [x, y].
        pred_occluded: A boolean array of predicted occlusions, in the same format
        as gt_occluded.
        pred_tracks: An array of track predictions from your algorithm, in the same
        format as gt_tracks.
        intrinsics_params: camera intrinsic parameters, [fx, fy, cx, cy].  Full
        intrinsic matrix has the form [[fx, 0, cx],[0, fy, cy],[0,0,1]]
        get_trackwise_metrics: if True, the metrics will be computed for every
        track (rather than every video, which is the default).  This means every
        output tensor will have an extra axis [batch, num_tracks] rather than
        simply (batch).
        scaling: How to rescale the estimated point tracks to match the global
        scale of the ground truth points.  Possible options are "median" (to
        scale by the median norm of points visible in both prediction and ground
        truth; default), "mean" (same as "median", but using the Euclidean mean),
        "per_trajectory" which rescales predicted the predicted track so that the
        predicted depth on the query frame matches the ground truth,
        "local_neighborhood" which gathers, for every track, all points that are
        within a threshold (local_neighborhood_thresh) and treats all such points
        as a single track, afterward performing "per_trajectory" scaling, "none"
        (don't rescale points at all), and "reproduce_2d" which scales every
        point to match ground truth depth without changing the reprojection,
        which will reproduce the thresholds from 2D TAP.  Note that this won't
        exactly match the output of compute_tapvid_metrics because that function
        ignores the query points.
        query_points: query points, of shape [b, n, 3] or [n, 3] t/y/x points. Only
        needed if scaling == "per_trajectory", so we know which frame to use for
        rescaling.
        use_fixed_metric_threshold: if True, the metrics will be computed using
        fixed metric bubble thresholds, rather than the depth-adaptive thresholds
        scaled depth and by the camera intrinsics.
        local_neighborhood_thresh: distance threshold for local_neighborhood
        scaling.
        order: order of the prediction and visibility tensors.  Can be 'n t'
            (default), 't n', or 'b n t' or 'b t n'.

    Returns:
        A dict with the following keys:

        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
            predicted to be within the given back-projected pixel threshold,
            ignoring occlusion prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
            threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    # Adjust variable input shapes and orders to expected 'b n t',
    # except in case of local_neighborhood, which expects 'n t'.
    batched_input = len(order.split(" ")) == 3
    if scaling == "local_neighborhood":
        assert not batched_input, "Local neighborhood doesn't support batched inputs."
        output_order = "n t"
    else:
        if batched_input:
            output_order = "b n t"
        else:
            output_order = "() n t"  # Append batch axis.

    gt_occluded = einops.rearrange(gt_occluded, f"{order} -> {output_order}")
    pred_occluded = einops.rearrange(pred_occluded, f"{order} -> {output_order}")
    gt_tracks = einops.rearrange(gt_tracks, f"{order} d -> {output_order} d")
    pred_tracks = einops.rearrange(pred_tracks, f"{order} d -> {output_order} d")

    summing_axis = (-1,) if get_trackwise_metrics else (-2, -1)

    # eye = np.eye(gt_tracks.shape[2])
    # query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye

    # query_frame = query_points[..., 0]
    # query_frame = np.round(query_frame).astype(np.int32)
    # evaluation_weights = query_frame_to_eval_frames[query_frame]
    evaluation_weights = np.ones(gt_occluded.shape)

    metrics = {}

    pred_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(pred_tracks), axis=-1)))
    gt_norms = np.sqrt(np.maximum(1e-12, np.sum(np.square(gt_tracks), axis=-1)))
    if scaling == "reproduce_2d":
        scale_factor = gt_tracks[..., -1:] / pred_tracks[..., -1:]
    elif scaling == "per_trajectory" or scaling == "local_neighborhood":
        query_frame = np.round(query_points[..., 0]).astype(np.int32)[..., np.newaxis]

        def do_take(x):
            took = np.take_along_axis(x, query_frame, axis=-1)
            return np.maximum(took, 1e-12)[..., np.newaxis]

        scale_factor = do_take(gt_tracks[..., -1]) / do_take(pred_tracks[..., -1])
        if scaling == "local_neighborhood":
            gt_occluded, gt_tracks, pred_occluded, pred_tracks, evaluation_weights = create_local_tracks(
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                thresh=local_neighborhood_thresh,
            )
    else:
        either_occluded = np.logical_or(gt_occluded, pred_occluded)
        nan_mat = np.full(pred_norms.shape, np.nan)
        pred_norms = np.where(either_occluded, nan_mat, pred_norms)
        gt_norms = np.where(either_occluded, nan_mat, gt_norms)
        if scaling == "median":
            scale_factor = np.nanmedian(gt_norms, axis=(-2, -1), keepdims=True) / np.nanmedian(
                pred_norms, axis=(-2, -1), keepdims=True
            )
        elif scaling == "mean":
            scale_factor = np.nanmean(gt_norms, axis=(-2, -1), keepdims=True) / np.nanmean(
                pred_norms, axis=(-2, -1), keepdims=True
            )
        elif scaling == "none":
            scale_factor = 1.0
        elif scaling == "median_on_queries":
            range_n_pts = np.arange(pred_norms.shape[-2])
            query_frame = np.round(query_points[..., 0]).astype(np.int32).squeeze()
            pred_norms = pred_norms[:, range_n_pts, query_frame][..., None]
            gt_norms = gt_norms[:, range_n_pts, query_frame][..., None]
            scale_factor = np.nanmedian(gt_norms, axis=(-2, -1), keepdims=True) / np.nanmedian(
                pred_norms, axis=(-2, -1), keepdims=True
            )
        else:
            raise ValueError("Unknown scaling:" + scaling)
    # breakpoint()
    pred_tracks = pred_tracks * scale_factor

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    metrics["occlusion_accuracy"] = np.sum(
        np.equal(pred_occluded, gt_occluded) * evaluation_weights,
        axis=summing_axis,
    ) / np.sum(evaluation_weights, axis=summing_axis)

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_frac_within_vis_and_occ = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        if use_fixed_metric_threshold:
            pointwise_thresh = PIXEL_TO_FIXED_METRIC_THRESH[thresh]
        else:
            if intrinsics_params[2] < 260:  # ADT dataset
                ori_dim = 512
            elif intrinsics_params[2] < 500:  # pstudio dataset
                ori_dim = 360
            else:  # drivetrack dataset
                ori_dim = 1280

            # ori_dim = 360 if intrinsics_params[2] < 500 else 1280

            multiplier = get_pointwise_threshold_multiplier(gt_tracks, intrinsics_params * 256 / ori_dim)
            pointwise_thresh = thresh * multiplier

            # breakpoint()
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(pointwise_thresh)

        # breakpoint()
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct = np.sum(
            is_correct * evaluation_weights,
            axis=summing_axis,
        )
        count_visible_points = np.sum(visible * evaluation_weights, axis=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        count_correct_with_occ = np.sum(
            within_dist * evaluation_weights,
            axis=summing_axis,
        )
        count_visible_points_with_occ = np.sum(evaluation_weights, axis=summing_axis)
        frac_correct_with_occ = count_correct_with_occ / count_visible_points_with_occ
        metrics["pts_within_with_occ_" + str(thresh)] = frac_correct_with_occ
        all_frac_within_vis_and_occ.append(frac_correct_with_occ)

        true_positives = np.sum((is_correct & pred_visible) * evaluation_weights, axis=summing_axis)

        # breakpoint()
        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives = np.sum(visible * evaluation_weights, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives * evaluation_weights, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    # breakpoint()
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=-2),
        axis=-2,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=-2),
        axis=-2,
    )

    metrics["average_pts_within_thresh_with_occ"] = np.mean(
        np.stack(all_frac_within_vis_and_occ, axis=-2),
        axis=-2,
    )

    if return_scaled_pred:
        pred_tracks = einops.rearrange(pred_tracks, f"{output_order} d -> {order} d")
        return metrics, pred_tracks
    return metrics

def evaluate_pttrack_3d(val_dataloader, model, lift_3d=False, verbose=False):
    '''
    Evaluate 3D point tracking metrics from https://github.com/snap-research/DELTA_densetrack3d
    '''

    metrics = {}
    for ind, (video, videodepth, trajectory_2d, trajectory_3d, visibility, video_name, query_points_3d, intrinsic_mat) in enumerate(val_dataloader):

        video = video.cuda(non_blocking=True)
        trajectory_2d = trajectory_2d.cuda(non_blocking=True)
        trajectory_3d = trajectory_3d.cuda(non_blocking=True)
        visibility = visibility.cuda(non_blocking=True)
        query_points_3d = query_points_3d.cuda(non_blocking=True)
        intrinsic_mat = intrinsic_mat.cuda(non_blocking=True)
        
        queries = query_points_3d.clone().float()  # B N 4 in (t, x, y, d)


        n_queries = queries.shape[1]

        intrs = sample.intrs

        out = model.inference(
            video=video,
            depths=videodepth,
            queries=query_points_3d,
            intrs=intrinsic_mat,
            return_3d=True,
        )

        traj_e = traj_e[:, :, :n_queries]
        traj_d_e = traj_d_e[:, :, :n_queries]
        vis_e = vis_e[:, :, :n_queries]

        if "tapvid3d" in dataset_name:
            # NOTE tracking backward
            inv_video = sample.video.flip(1).clone()
            inv_videodepth = sample.videodepth.flip(1).clone()
            inv_queries = queries.clone()
            inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

            inv_traj_e, inv_traj_d_e, inv_vis_e = model(
                video=inv_video,
                videodepth=inv_videodepth,
                queries=inv_queries,
                depth_init=inv_videodepth[:, 0],
                return_3d=True,
                is_sparse=is_sparse,
                lift_3d=lift_3d,
            )

            inv_traj_e = inv_traj_e[:, :, :n_queries].flip(1)
            inv_traj_d_e = inv_traj_d_e[:, :, :n_queries].flip(1)
            inv_vis_e = inv_vis_e[:, :, :n_queries].flip(1)

            arange = torch.arange(sample.video.shape[1], device=queries.device)[None, :, None]
            mask = (arange < queries[:, None, :, 0]).unsqueeze(-1).repeat(1, 1, 1, inv_traj_e.shape[-1])

            traj_e[mask] = inv_traj_e[mask]
            traj_d_e[mask[:, :, :, 0]] = inv_traj_d_e[mask[:, :, :, 0]]
            vis_e[mask[:, :, :, 0]] = inv_vis_e[mask[:, :, :, 0]]

        traj_uvd = torch.cat([traj_e, traj_d_e], dim=-1)
        traj_3d = reproject_2d3d(traj_uvd, intrs)

        scaled_pred_traj_3d = self.compute_metrics_3d(metrics, sample, traj_3d, vis_e, dataset_name)
        if verbose:
            print("Avg:", metrics["avg"])
    return metrics

# object tracking evaluation
def evaluate_results(
    res_file,
    gt_file,
    track_name,
    metric_list,
    save_folder=None,
    subset="all",
):
    _metric_list = []
    for metric in metric_list:
        if metric == "TETA":
            from teta.datasets import TAO
            from teta.eval import Evaluator
            from teta.metrics import TETA
            freeze_support()
            eval_config, dataset_config, metrics_config = default_configs()
            metrics_config["METRICS"] = ["TETA"]
            dataset_config["TRACKERS_TO_EVAL"] = [track_name]
            dataset_config["GT_FOLDER"] = gt_file
            dataset_config["TRACKER_SUB_FOLDER"] = res_file
            dataset_config["TRACKERS_FOLDER"] = save_folder
            if save_folder:
                dataset_config["OUTPUT_FOLDER"] = save_folder
            else:
                dataset_config["OUTPUT_FOLDER"] = os.path.dirname(res_file)
            evaluator = Evaluator(eval_config)
            dataset_list = [TAO(dataset_config)]
            metrics_list = []
            metric = TETA(exhaustive=False)
            if metric.get_name() in metrics_config["METRICS"]:
                metrics_list.append(metric)
            if len(metrics_list) == 0:
                raise Exception("No metrics selected for evaluation")

            tracker_name = dataset_config["TRACKERS_TO_EVAL"][0]
            resfile_path = dataset_config["TRACKERS_FOLDER"]
            dataset_gt = json.load(open(dataset_config["GT_FOLDER"]))
            eval_results, _ = evaluator.evaluate(dataset_list, metrics_list)

            eval_results_path = os.path.join(
                resfile_path, tracker_name, "teta_summary_results.pth"
            )
            # eval_results_path = os.path.join(
            #     save_folder, tracker_name, "teta_summary_results.pth"
            # )
            eval_res = pickle.load(open(eval_results_path, "rb"))

            base_class_synset = set(
                [
                    c["name"]
                    for c in dataset_gt["categories"]
                    if c["frequency"] != "r"
                ]
            )
            novel_class_synset = set(
                [
                    c["name"]
                    for c in dataset_gt["categories"]
                    if c["frequency"] == "r"
                ]
            )
            freq_teta_mean, rare_teta_mean = compute_teta_on_ovsetup(
                eval_res, base_class_synset, novel_class_synset
            )
            with open(os.path.join(dataset_config["OUTPUT_FOLDER"], tracker_name, "freq_teta_mean.txt"), "w") as f:
                if freq_teta_mean is not None:
                    f.write(
                        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                        "TETA",
                        "LocA",
                        "AssocA",
                        "ClsA",
                        "LocRe",
                        "LocPr",
                        "AssocRe",
                        "AssocPr",
                        "ClsRe",
                        "ClsPr")
                    )
                    f.write("\n")
                    f.write("".join(["{:<10.3f}".format(num) for num in freq_teta_mean]))
                else:
                    f.write("No Base classes to evaluate!")

            with open(os.path.join(dataset_config["OUTPUT_FOLDER"], tracker_name, "rare_teta_mean.txt"), "w") as f:
                if rare_teta_mean is not None:
                    f.write(
                        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                        "TETA",
                        "LocA",
                        "AssocA",
                        "ClsA",
                        "LocRe",
                        "LocPr",
                        "AssocRe",
                        "AssocPr",
                        "ClsRe",
                        "ClsPr")
                    )
                    f.write("\n")
                    f.write("".join(["{:<10.3f}".format(num) for num in rare_teta_mean]))
                else:
                    f.write("No Novel classes to evaluate!")
        elif metric == "TrackMAP":
            import trackeval
            freeze_support()
            # Command line interface:
            default_eval_config = trackeval.Evaluator.get_default_eval_config()
            # print only combined since TrackMAP is undefined for per sequence breakdowns
            default_eval_config['PRINT_ONLY_COMBINED'] = True
            default_eval_config['DISPLAY_LESS_PROGRESS'] = True
            default_dataset_config = trackeval.datasets.TAO.get_default_dataset_config()
            default_metrics_config = {'METRICS': ['TrackMAP']}
            config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
            config["TRACKERS_TO_EVAL"] = [track_name]
            config["GT_FOLDER"] = gt_file
            config["TRACKER_SUB_FOLDER"] = res_file
            eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
            dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
            metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
            if save_folder:
                dataset_config["OUTPUT_FOLDER"] = save_folder
            else:
                dataset_config["OUTPUT_FOLDER"] = os.path.dirname(res_file)

            # Run code
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.TAO_OW(dataset_config)]
            metrics_list = []
            for metric in [trackeval.metrics.TrackMAP]:
                if metric.get_name() in metrics_config['METRICS']:
                    metrics_list.append(metric())
            if len(metrics_list) == 0:
                raise Exception('No metrics selected for evaluation')
            evaluator.evaluate(dataset_list, metrics_list)
        else:
            _metric_list.append(metric)
        if len(_metric_list) > 0:
            import trackeval
            _metrics = []
            for metric in _metric_list:
                if metric == "OWTA":
                    _metrics.append(trackeval.metrics.HOTA)
                elif metric == "CLEAR":
                    _metrics.append(trackeval.metrics.CLEAR)
                elif metric == "Identity":
                    _metrics.append(trackeval.metrics.Identity)
                elif metric == "TrackMAP":
                    _metrics.append(trackeval.metrics.TrackMAP)
                    
            freeze_support()
            # Command line interface:
            default_eval_config = trackeval.Evaluator.get_default_eval_config()
            # print only combined since TrackMAP is undefined for per sequence breakdowns
            default_eval_config['PRINT_ONLY_COMBINED'] = True
            default_eval_config['DISPLAY_LESS_PROGRESS'] = True
            default_dataset_config = trackeval.datasets.TAO_OW.get_default_dataset_config()
            default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
            config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
            config["TRACKERS_TO_EVAL"] = [track_name]
            config["GT_FOLDER"] = gt_file
            config["TRACKER_SUB_FOLDER"] = res_file
            config["SUBSET"] = subset
            eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
            dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
            metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
            if save_folder:
                dataset_config["OUTPUT_FOLDER"] = save_folder
            else:
                dataset_config["OUTPUT_FOLDER"] = os.path.dirname(res_file)

            # Run code
            evaluator = trackeval.Evaluator(eval_config)
            dataset_list = [trackeval.datasets.TAO_OW(dataset_config)]
            metrics_list = []
            for metric in _metrics:
                if metric.get_name() in metrics_config['METRICS']:
                    metrics_list.append(metric())
            if len(metrics_list) == 0:
                raise Exception('No metrics selected for evaluation')
            evaluator.evaluate(dataset_list, metrics_list)


# object tracking evaluation
def compute_teta_on_ovsetup(teta_res, base_class_names, novel_class_names):
    if "COMBINED_SEQ" in teta_res:
        teta_res = teta_res["COMBINED_SEQ"]

    frequent_teta = []
    rare_teta = []
    for key in teta_res:
        if key in base_class_names:
            frequent_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))
        elif key in novel_class_names:
            rare_teta.append(np.array(teta_res[key]["TETA"][50]).astype(float))

    print("Base and Novel classes performance")

    # print the header
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "TETA50:",
            "TETA",
            "LocA",
            "AssocA",
            "ClsA",
            "LocRe",
            "LocPr",
            "AssocRe",
            "AssocPr",
            "ClsRe",
            "ClsPr",
        )
    )

    if frequent_teta:
        freq_teta_mean = np.mean(np.stack(frequent_teta), axis=0)

        # print the frequent teta mean
        print("".join(["{:<10} ".format("Base")]+["{:<10.3f}".format(num) for num in freq_teta_mean]))

    else:
        print("No Base classes to evaluate!")
        freq_teta_mean = None
    if rare_teta:
        rare_teta_mean = np.mean(np.stack(rare_teta), axis=0)

        # print the rare teta mean
        print("".join(["{:<10} ".format("Novel")]+["{:<10.3f}".format(num) for num in rare_teta_mean]))
    else:
        print("No Novel classes to evaluate!")
        rare_teta_mean = None

    return freq_teta_mean, rare_teta_mean

def default_configs():
    from teta.config import get_default_eval_config, get_default_dataset_config
    """Parse command line."""
    default_eval_config = get_default_eval_config()
    default_eval_config["DISPLAY_LESS_PROGRESS"] = True
    default_dataset_config = get_default_dataset_config()
    default_metrics_config = {"METRICS": ["TETA"]}
    config = {
        **default_eval_config,
        **default_dataset_config,
        **default_metrics_config,
    }
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {
        k: v for k, v in config.items() if k in default_dataset_config.keys()
    }
    metrics_config = {
        k: v for k, v in config.items() if k in default_metrics_config.keys()
    }

    return eval_config, dataset_config, metrics_config

