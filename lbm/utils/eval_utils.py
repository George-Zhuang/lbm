from typing import Mapping
import os
import yaml
import json
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import freeze_support


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
        if hasattr(args, 'val_memory_size'):
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
        
        if hasattr(args, 'lambda_cls'):
            print(f"Classification loss: {args.lambda_cls}")
        if hasattr(args, 'lambda_vis'):
            print(f"Visibility loss: {args.lambda_vis}")
        if hasattr(args, 'lambda_reg'):
            print(f"Regression loss: {args.lambda_reg}")
        if hasattr(args, 'lambda_unc'):
            print(f"Uncertainty loss: {args.lambda_unc}")
        if hasattr(args, 'lambda_ref'):
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
    suffix = f"{config.eval_dataset}"
    results_dir = Path(config.save_res_path)
    vis_dir = Path(config.save_vis_path)
    results_file = results_dir / f"{suffix}.json"
    return results_dir, vis_dir, results_file, suffix

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

    def get_results(self):
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

    def update(self, out_metrics, verbose=False):
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

    def report(self):
        print(f"Mean AJ: {sum(self.aj) / len(self.aj):.1f}")
        print(f"Mean delta_avg: {sum(self.delta_avg) / len(self.delta_avg):.1f}")
        print(f"Mean delta_1: {sum(self.delta_1) / len(self.delta_1):.1f}")
        print(f"Mean delta_2: {sum(self.delta_2) / len(self.delta_2):.1f}")
        print(f"Mean delta_4: {sum(self.delta_4) / len(self.delta_4):.1f}")
        print(f"Mean delta_8: {sum(self.delta_8) / len(self.delta_8):.1f}")
        print(f"Mean delta_16: {sum(self.delta_16) / len(self.delta_16):.1f}")
        print(f"Mean OA: {sum(self.oa) / len(self.oa):.1f}")

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