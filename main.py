# ---------------------------------------------------------------------
# LBM: lattice Boltzmann model
#   - Adapted from Track-On (https://github.com/gorkaydemir/track_on)
# ---------------------------------------------------------------------
import os
import sys
import argparse
import wandb
import time
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from lbm.utils.train_utils import init_distributed_mode, fix_random_seeds
from lbm.utils.train_utils import get_dataloaders, get_scheduler
from lbm.utils.train_utils import restart_from_checkpoint, save_on_master
from lbm.utils.log_utils import init_wandb, log_eval_metrics, log_batch_loss, log_epoch_loss
from lbm.utils.eval_utils import Evaluator, compute_tapvid_metrics
from lbm.utils.coord_utils import get_queries
from lbm.utils.eval_utils import load_config, print_args

from lbm.models.lbm import LBM


def get_args():
    parser = argparse.ArgumentParser("LBM")
    parser.add_argument('--config_path', type=str, default='lbm/configs/default.yaml')
    parser.add_argument('--movi_f_root', type=str, default="data/kubric_lbm")
    parser.add_argument('--tapvid_root', type=str, default="data/tapvid_davis/tapvid_davis.pkl")
    # eval
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--eval_dataset', type=str, choices=[
        "davis", "kinetics", "robotap", "bft", "tao", "ovt-b"
    ], default="davis")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    # training
    parser.add_argument('--epoch_num', type=int, default=150)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('--model_save_path', type=str, default="checkpoints/lbm")
    args = parser.parse_args()
    args = load_config(args)
    return args

def train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, epoch):
    model.train()
    total_loss = 0

    model.module.extend_queries = False
    model.module.transformer.random_mask_ratio = args.random_memory_mask_drop

    train_dataloader = tqdm(train_dataloader, disable=args.rank != 0, file=sys.stdout)
    printed_memory = False
    update_num = 0

    torch.cuda.reset_peak_memory_stats(device=args.gpu)
    for i, (video, tracks, visibility, k_points) in enumerate(train_dataloader):
        video = video.cuda(non_blocking=True)               # (B, T, C, H, W)
        tracks = tracks.cuda(non_blocking=True)             # (B, T, N, 2)
        visibility = visibility.cuda(non_blocking=True)     # (B, T, N)
        k_points = k_points.cuda(non_blocking=True)         # (B,)

        min_k = k_points.min()
        tracks = tracks[:, :, :min_k]
        visibility = visibility[:, :, :min_k]

        queries = get_queries(tracks, visibility)           # (B, N, 3)

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            out = model(video, queries, tracks, visibility)

        if args.rank == 0 and epoch == 0 and not printed_memory:
            print(f"\nMemory Usage after forward: {torch.cuda.max_memory_allocated(device=args.gpu) / 1024 ** 3:.1f} GB")
            torch.cuda.reset_peak_memory_stats(device=args.gpu)

        # Compute loss
        loss = torch.zeros(1).cuda()
        for key, value in out.items():
            if "loss" in key:
                loss += value

        total_loss += loss.item()

        # Backward pass
        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if args.rank == 0 and epoch == 0 and not printed_memory:
            print(f"Memory Usage after backward: {torch.cuda.max_memory_allocated(device=args.gpu) / 1024 ** 3:.1f} GB")
            printed_memory = True

        if args.amp:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        update_num += 1

        total_update_num = epoch * len(train_dataloader) + update_num
        log_batch_loss(args, optimizer, train_dataloader, total_update_num, i, out)

    log_epoch_loss(args, total_loss, epoch, train_dataloader)


@torch.no_grad()
def evaluate(args, val_dataloader, model, epoch, verbose=False):
    model.eval()
    model.module.extend_queries = True
    model.module.transformer.random_mask_ratio = 0

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
        video = video.cuda(non_blocking=True)                    # (1, T, 3, H, W)
        device = video.device

        # Change (t, y, x) to (t, x, y)
        queries = query_points_i.clone().float()
        queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)


        if args.validation:
            out = model.module.inference(video, queries)
        else:
            out = model.module(video, queries, trajectory, visibility)
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
    smaller_delta_avg = results["delta_avg"]
    aj = results["aj"]
    oa = results["oa"]

    
    log_eval_metrics(args, results, epoch)

    return smaller_delta_avg, aj, oa


def main_worker(args):
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    print_args(args)
    os.makedirs(args.model_save_path, exist_ok=True)
    start_time = time.time()

    # ##### Data #####
    train_dataloader, val_dataloader = get_dataloaders(args)
    if train_dataloader is not None:
        print(f"Total number of iterations: {len(train_dataloader) * args.epoch_num / 1000:.1f}K")
    # ##### ##### #####
    
    # ##### Model & Training #####
    model = LBM(args).to(args.gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)


    if not args.validation:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = get_scheduler(args, optimizer, train_dataloader)
        scaler = torch.amp.GradScaler() if args.amp else None
        init_wandb(args)

        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 10**6:.2f}M")
        print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}M")
        print()

    # ##### Load from checkpoint #####
    to_restore = {"epoch": 0}
    if args.checkpoint_path is not None:
        if args.validation:
            restart_from_checkpoint(args, 
                                run_variables=to_restore, 
                                model=model)
        else:
            restart_from_checkpoint(args, 
                                    run_variables=to_restore, 
                                    model=model,
                                    scaler=scaler,
                                    optimizer=optimizer, 
                                    scheduler=lr_scheduler)
    
    
    start_epoch = to_restore["epoch"]

    if args.validation and args.rank == 0:
        model.module.visibility_threshold = args.val_vis_delta
        model.module.set_memory_size(args.val_memory_size)
        evaluate(args, val_dataloader, model, -1, verbose=True)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print()
        print('Validation time {}'.format(total_time_str))

    dist.barrier()
    if args.validation:
        dist.destroy_process_group()
        return
    
    print("Training starts")

    # === Sanity Check ===
    if args.rank == 0:
        print("Running sanity check on validation set...")
        try:
            for j, (video, trajectory, visibility, query_points_i) in enumerate(val_dataloader):
                if j >= 1:  
                    break
                    
                query_points_i = query_points_i.cuda(non_blocking=True)
                trajectory = trajectory.cuda(non_blocking=True)
                visibility = visibility.cuda(non_blocking=True)
                video = video.cuda(non_blocking=True)
                device = video.device

                queries = query_points_i.clone().float()
                queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)

                out = model.module(video, queries, trajectory, visibility)
                pred_trajectory = out["points"]
                pred_visibility = out["visibility"]
                
                print(f"Sanity check passed! Video shape: {video.shape}, Pred shape: {pred_trajectory.shape}")
                break
                
        except Exception as e:
            print(f"Sanity check failed: {e}")
            raise e
    
    dist.barrier()
    # === === ===

    best_models = {"aj": [-1, -1], "oa": [-1, -1], "delta_avg": [-1, -1]}       # [epoch, value]
    for epoch in range(start_epoch, args.epoch_num):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"=== === Epoch {epoch} === ===")

        # === === Training === ===

        train(args, train_dataloader, model, optimizer, lr_scheduler, scaler, epoch)
        print()
        # === === ===
        
        # === Evaluation ===
        if args.rank == 0:

            smaller_delta_avg, aj, oa = evaluate(args, val_dataloader, model, epoch)
            
            # === Save Model ===
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch + 1,
                "args": args,
            }
            save_on_master(save_dict, os.path.join(args.model_save_path, "checkpoint.pt"))
            
            if aj > best_models["aj"][1]:
                best_models["aj"] = [epoch, aj]
                save_on_master(save_dict, os.path.join(args.model_save_path, "best_aj.pt"))
            
            if oa > best_models["oa"][1]:
                best_models["oa"] = [epoch, oa]
                save_on_master(save_dict, os.path.join(args.model_save_path, "best_oa.pt"))
            
            if smaller_delta_avg > best_models["delta_avg"][1]:
                best_models["delta_avg"] = [epoch, smaller_delta_avg]
                save_on_master(save_dict, os.path.join(args.model_save_path, "best_delta.pt"))

            # === === ===

        dist.barrier()

        print(f"=== === === === === ===")
        print()

    # print best results
    if args.rank == 0:
        print("Best Results")
        print(f"Best AJ: {best_models['aj'][1]:.3f} at epoch {best_models['aj'][0]}")
        print(f"Best OA: {best_models['oa'][1]:.3f} at epoch {best_models['oa'][0]}")
        print(f"Best Smaller Delta Avg: {best_models['delta_avg'][1]:.3f} at epoch {best_models['delta_avg'][0]}")
    
    wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    main_worker(args)