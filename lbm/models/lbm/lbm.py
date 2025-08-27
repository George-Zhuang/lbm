import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lbm.models.lbm.backbone import Backbone
from lbm.models.lbm.loss import Loss_Function
from lbm.models.lbm.transformer import LBMTransformer
from lbm.utils.eval_utils import reproject_2d3d
from lbm.utils.kalman_filter import MultiPointKF3D


class LBM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.N = args.N                 # number of queries
        self.T = args.T                 # number of frames
        self.visibility_threshold = 0.8

        self.size = args.input_size
        self.stride = args.stride
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.num_levels = args.num_levels
        self.num_neighbors = args.num_neighbors
        self.memory_size = args.memory_size
        self.top_k_regions = args.top_k_regions
        self.dropout = args.dropout
        self.spa_corr_thre = args.spa_corr_thre
        self.add_spatial_penalty = args.add_spatial_penalty

        self.lambda_cls = args.lambda_cls
        self.lambda_vis = args.lambda_vis
        self.lambda_off = args.lambda_off
        self.lambda_unc = args.lambda_unc
        self.lambda_ref = args.lambda_ref
        self.loss_after_query = args.loss_after_query

        self.h = self.size[0] // self.stride
        self.w = self.size[1] // self.stride

        self.backbone = Backbone(
            size=self.size,
            stride=self.stride,
            embed_dim=self.embed_dim,
        )

        self.transformer = LBMTransformer(
            size=self.size, 
            stride=self.stride, 
            n_layer=self.num_layers, 
            hidden_dim=self.embed_dim, 
            n_head=self.num_heads, 
            n_level=self.num_levels, 
            n_neighbor=self.num_neighbors, 
            n_mem=self.memory_size,
            n_topk=self.top_k_regions, 
            dropout=self.dropout,
            spa_corr_thre=self.spa_corr_thre,
            add_spatial_penalty=self.add_spatial_penalty,
        )

        self.loss = Loss_Function(
            size=self.size,
            stride=self.stride,
            lambda_cls=self.lambda_cls, 
            lambda_vis=self.lambda_vis,
            lambda_off=self.lambda_off,
            lambda_unc=self.lambda_unc,
            lambda_ref=self.lambda_ref,
            loss_after_query=self.loss_after_query,
        )

        self.kf = MultiPointKF3D(
            process_noise=0.1,
            measurement_noise=10.0,
            visibility_threshold=0.3,
            reinit_visibility_threshold=0.7,
            lost_frames_threshold=10,
        )

    def set_memory_size(self, memory_size):
        for i in range(self.num_layers):
            collision_time_emb = self.transformer.layers[i].collision_time_emb
            collision_time_emb_past = collision_time_emb[:, :-1]
            collision_time_emb_now = collision_time_emb[:, -1].unsqueeze(dim=1)
            collision_time_emb_past = collision_time_emb_past.permute(0, 2, 1)
            collision_time_emb_interpolated = F.interpolate(collision_time_emb_past, size=memory_size, mode='linear', align_corners=True)
            collision_time_emb_interpolated = collision_time_emb_interpolated.permute(0, 2, 1)
            self.transformer.layers[i].collision_time_emb = nn.Parameter(torch.cat([collision_time_emb_interpolated, collision_time_emb_now], dim=1))
    
            stream_time_emb = self.transformer.layers[i].stream_time_emb
            stream_time_emb_past = stream_time_emb[:, :-1]
            stream_time_emb_now = stream_time_emb[:, -1].unsqueeze(dim=1)
            stream_time_emb_past = stream_time_emb_past.permute(0, 2, 1)
            stream_time_emb_interpolated = F.interpolate(stream_time_emb_past, size=memory_size, mode='linear', align_corners=True)
            stream_time_emb_interpolated = stream_time_emb_interpolated.permute(0, 2, 1)
            self.transformer.layers[i].stream_time_emb = nn.Parameter(torch.cat([stream_time_emb_interpolated, stream_time_emb_now], dim=1))
        self.memory_size = memory_size
        
    def scale_inputs(self, queries, gt_tracks, H, W):
        '''
        Args:
            queries: (B, N, 3) where 3 is (t, x, y), 
                or (B, N, 4) where 4 is (t, x, y, d)
            gt_tracks: (B, T, N, 2) in pixel space
        '''
        queries[:, :, 2] = (queries[:, :, 2] / H) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W) * self.size[1]

        gt_tracks_tmp = gt_tracks.clone()
        gt_tracks_tmp[:, :, :, 1] = (gt_tracks_tmp[:, :, :, 1] / H) * self.size[0]
        gt_tracks_tmp[:, :, :, 0] = (gt_tracks_tmp[:, :, :, 0] / W) * self.size[1]

        return queries, gt_tracks_tmp

    def forward(self, video, queries, gt_tracks, gt_visibility):
        '''
        Args:
            video: (B, T, C, H, W)
            queries: (B, N, 3) where 3 is (t, x, y)
            gt_tracks: (B, T, N, 2) in pixel space
            gt_visibility: (B, T, N), in [0, 1]
        '''

        B, T, _, H, W = video.shape
        N = queries.shape[1]
        device = video.device

        out = {}

        # backbone 
        queries_scaled, gt_tracks_scaled = self.scale_inputs(queries, gt_tracks, H, W) # b n 3, b t n 2
        query_times = queries_scaled[:, :, 0].long() # b n                                            
        tokens, q_init = self.backbone(video, queries_scaled) # b hw c, b n c
        query_num = q_init.shape[1]

        # init memory
        collision_dist = torch.zeros(B, query_num, self.memory_size, self.embed_dim, device=device) # b n m c
        stream_dist = torch.zeros(B, query_num, self.memory_size, self.embed_dim, device=device) # b n m c
        vis_mask = torch.ones(B, query_num, self.memory_size, device=device, dtype=torch.bool) # b n m
        mem_mask = torch.ones(B, query_num, self.memory_size, device=device, dtype=torch.bool) # b n m

        # online forward
        corr = [[] for _ in range(self.num_layers)]
        ref_rho = [[] for _ in range(self.num_layers)]
        ref_pts = [[] for _ in range(self.num_layers)]
        offsets = []
        visibility = []
        rhos = []
        for t in range(T):
            # forward layers
            queried_now_or_before = (query_times <= t) # b n
            f_t = tokens[:, t] # b hw c
            f_t = rearrange(f_t, 'b (h w) c -> b c h w', h=self.h, w=self.w) # b c h w
            out_t, memory, offset, vis, rho = self.transformer(
                query=q_init.clone(), 
                feat=f_t, 
                stream_dist=stream_dist.clone(), 
                collision_dist=collision_dist.clone(), 
                vis_mask=vis_mask, 
                mem_mask=mem_mask, 
                queried_now_or_before=queried_now_or_before,
            )

            # update
            collision_dist = torch.cat([collision_dist[:, :, 1:], memory['collision'].unsqueeze(2)], dim=2) # b n m c
            stream_dist = torch.cat([stream_dist[:, :, 1:], memory['stream'].unsqueeze(2)], dim=2) # b n m c
            mem_mask = torch.cat([mem_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2) # b n m                          
            vis_mask = torch.cat([vis_mask[:, :, 1:], (F.sigmoid(vis) < self.visibility_threshold).unsqueeze(-1)], dim=2) # b n m
        
            offsets.append(offset)
            visibility.append(vis)
            rhos.append(rho)
            for i in range(self.num_layers):
                corr[i].append(out_t[i]['corr'])
                ref_rho[i].append(out_t[i]['reference_rho'])
                ref_pts[i].append(out_t[i]['reference_points'])

        offsets = torch.stack(offsets, dim=1) # b t n 2
        visibility = torch.stack(visibility, dim=1) # b t n
        rhos = torch.stack(rhos, dim=1) # b t n
        for i in range(self.num_layers):
            corr[i] = torch.stack(corr[i], dim=1) # b t n p
            ref_rho[i] = torch.stack(ref_rho[i], dim=1) # b t n k
            ref_pts[i] = torch.stack(ref_pts[i], dim=1) # b t n k 2

        # loss
        loss_point = 0
        for i in range(self.num_layers):
            loss_point += self.loss.point_loss(corr[i], gt_tracks_scaled, gt_visibility, query_times)
        out["point_loss"] = loss_point / self.num_layers

        out["visibility_loss"] = self.loss.visibility_loss(visibility, gt_visibility, query_times)

        ref_tracks = ref_pts[-1].squeeze(-2) # b t n 2
        out["offset_loss"] = self.loss.offset_loss(offsets.unsqueeze(2), ref_tracks, gt_tracks_scaled, gt_visibility, query_times)
        
        pred_tracks = ref_tracks + offsets # b t n 2
        out["uncertainty_loss"] = self.loss.uncertainty_loss(rhos, pred_tracks, gt_tracks_scaled, gt_visibility, query_times) * self.loss.lambda_unc

        loss_rho = 0
        num_k = 0
        for i in range(self.num_layers - 1):
            n_topk = ref_rho[i].shape[-1]
            for j in range(n_topk):
                num_k += 1
                loss_rho += self.loss.uncertainty_loss(ref_rho[i][:, :, :, j], ref_pts[i][:, :, :, j], gt_tracks_scaled, gt_visibility, query_times) * self.loss.lambda_ref
        out["uncertainty_loss_topk"] = loss_rho / num_k

        # outputs
        pred_tracks[..., 0] = (pred_tracks[..., 0] / self.size[1]) * W
        pred_tracks[..., 1] = (pred_tracks[..., 1] / self.size[0]) * H
        out['points'] = pred_tracks
        out["visibility"] = F.sigmoid(visibility) > self.visibility_threshold

        return out

    def inference(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W)
            queries: (B, N, 3) where 3 is (t, y, x)
        '''

        B, T, C, H, W = video.shape
        N = queries.shape[1]
        device = video.device

        out = {}

        # query
        queries[:, :, 2] = (queries[:, :, 2] / H) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W) * self.size[1]
        last_pos = queries[:, :, 1:].clone()

        query_times = queries[:, :, 0].long()  
        
        q_init = self.backbone.sample_queries_online(video, queries) # b n c
        _N = q_init.shape[1]

        # init memory
        collision_dist = torch.zeros(B, _N, self.memory_size, self.embed_dim, device=device) # b n m c
        stream_dist = torch.zeros(B, _N, self.memory_size, self.embed_dim, device=device) # b n m c
        vis_mask = torch.ones(B, _N, self.memory_size, device=device, dtype=torch.bool) # b n m
        mem_mask = torch.ones(B, _N, self.memory_size, device=device, dtype=torch.bool) # b n m

        # online forward
        coord_pred = [] 
        vis_pred = [] 
        for t in range(T):
            queried_now_or_before = (query_times <= t)
            f_t = self.backbone.encode_frames_online(video[:, t]) # b hw c
            f_t = rearrange(f_t, 'b (h w) c -> b c h w', h=self.h, w=self.w) # b c h w
            out_t, memory, offset, vis, rho = self.transformer(
                query=q_init.clone(), 
                feat=f_t, 
                stream_dist=stream_dist.clone(), 
                collision_dist=collision_dist.clone(), 
                vis_mask=vis_mask, 
                mem_mask=mem_mask, 
                queried_now_or_before=queried_now_or_before,
                last_pos=last_pos,
            )

            # update
            collision_dist = torch.cat([collision_dist[:, :, 1:], memory['collision'].unsqueeze(2)], dim=2) # b n m c
            stream_dist = torch.cat([stream_dist[:, :, 1:], memory['stream'].unsqueeze(2)], dim=2) # b n m c
            mem_mask = torch.cat([mem_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2) # b n m                          
            vis_mask = torch.cat([vis_mask[:, :, 1:], (F.sigmoid(vis) < self.visibility_threshold).unsqueeze(-1)], dim=2) # b n m

            ref_pts = out_t[-1]['reference_points'].squeeze(-2) # b n 2
            pred_tracks = ref_pts + offset # b n 2
            coord_pred.append(pred_tracks.unsqueeze(1)) # b 1 n 2
            vis_pred.append(vis.unsqueeze(1)) # b 1 n
            last_pos = torch.where(queried_now_or_before.unsqueeze(-1), pred_tracks, last_pos)

        # output
        coord_pred = torch.cat(coord_pred, dim=1) # b t n 2
        vis_pred = torch.cat(vis_pred, dim=1) # b t n

        coord_pred[:, :, :, 1] = (coord_pred[:, :, :, 1] / self.size[0]) * H
        coord_pred[:, :, :, 0] = (coord_pred[:, :, :, 0] / self.size[1]) * W

        out["points"] = coord_pred
        out["visibility"] = F.sigmoid(vis_pred) > self.visibility_threshold

        return out

    def inference_3d(self, video, videodepth, queries, intrinsic_mat, reverse=False, use_kf=False):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255], C=3
            videodepth: (B, T, 1, H, W) in meters
            queries: (B, N, 4) where 4 is (t, x, y, d)
            intrinsic_mat: (B, 3, 3) camera intrinsic matrix
        '''
        b, t, c, h, w = video.shape
        queries_2d = queries[:, :, :3]
        out = self.inference(video, queries_2d)

        # Get predicted 2D points
        B, T, N = out['points'].shape[:3]
        pred_points = out['points']  # (B, T, N, 2)
        
        # Normalize to [0,1] for sampling
        normalized_points = pred_points.clone()
        normalized_points[..., 0] = normalized_points[..., 0] / w
        normalized_points[..., 1] = normalized_points[..., 1] / h
        
        # Convert to grid_sample format
        grid = normalized_points.reshape(B, T*N, 1, 2) * 2 - 1  # convert to [-1,1]
        
        # Sample depth
        depths = []
        for t in range(T):
            depth_t = videodepth[:, t:t+1]  # (B, 1, 1, H, W)
            depth_t = depth_t.squeeze(1)  # (B, 1, H, W)
            
            sampled_depth = F.grid_sample(
                depth_t,
                grid[:, t*N:(t+1)*N],
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # (B, 1, N, 1)
            
            depths.append(sampled_depth.squeeze(-1).squeeze(1))  # (B, N)
        
        depths = torch.stack(depths, dim=1)  # (B, T, N)
        out['depth'] = depths

        if reverse:
            pred_uv = torch.flip(out["points"], dims=[1])              # (B, T, N, 2)
            pred_d = torch.flip(out["depth"], dims=[1])                # (B, T, N)
            pred_visibility = torch.flip(out["visibility"], dims=[1])  # (B, T, N)
        else:
            pred_uv = out["points"]                # (B, T, N, 2)
            pred_d = out["depth"]                  # (B, T, N)
            pred_visibility = out["visibility"]    # (B, T, N)

        # Convert predictions to 3D coordinates
        pred_uvd = torch.cat([pred_uv, pred_d.unsqueeze(-1)], dim=3)   # (B, T, N, 3)
        pred_xyz = reproject_2d3d(pred_uvd, intrinsic_mat)             # (B, T, N, 3)

        if use_kf:
            # Get query times
            query_times = queries[:, :, 0].long()  # (B, N)
            
            # Apply Kalman filter for each point
            for n in range(N):
                query_time = query_times[0, n]  # Get query time for current point
                for t in range(T):
                    # Only update after query time
                    if t >= query_time:
                        # Get current point's 3D coordinates and visibility
                        measurement = pred_xyz[0, t, n].cpu().numpy()  # (3,)
                        visibility = pred_visibility[0, t, n].cpu().numpy()
                        
                        # Update Kalman filter
                        self.kf.update(
                            point_id=n,
                            measurement=measurement,
                            visibility=visibility
                        )
                        
                        # Get filtered position
                        filtered_pos = self.kf.get_point_position(n)
                        if filtered_pos is not None:
                            pred_xyz[0, t, n] = torch.from_numpy(filtered_pos).to(pred_xyz.device)

        return pred_xyz, pred_visibility
    