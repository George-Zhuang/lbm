import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lbm.models.lbm.backbone import Backbone, LBMConvNeXt
from lbm.models.lbm.loss import Loss_Function
from lbm.models.lbm.transformer import LBMTransformer


class LBM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.N = args.N # number of queries
        self.T = args.T # number of frames
        self.visibility_treshold = 0.8

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

        self.lambda_cls = args.lambda_cls
        self.lambda_vis = args.lambda_vis
        self.lambda_reg = args.lambda_reg
        self.lambda_unc = args.lambda_unc
        self.lambda_ref = args.lambda_ref
        self.loss_after_query = args.loss_after_query

        self.h = self.size[0] // self.stride
        self.w = self.size[1] // self.stride

        # self.backbone = Backbone(
        #     size=self.size,
        #     stride=self.stride,
        #     embed_dim=self.embed_dim,
        # )
        
        self.backbone = LBMConvNeXt(
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
        )

        self.loss = Loss_Function(
            size=self.size,
            stride=self.stride,
            lambda_cls=self.lambda_cls, 
            lambda_vis=self.lambda_vis,
            lambda_reg=self.lambda_reg,
            lambda_unc=self.lambda_unc,
            lambda_ref=self.lambda_ref,
            loss_after_query=self.loss_after_query,
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
            queries: (B, N, 3), in format (t, x, y)
            gt_tracks: (B, T, N, 2) in pixel space
        Returns:
            queries: (B, N, 3), in format (t, x, y), scaled to self.size
            gt_tracks: (B, T, N, 2) in pixel space, scaled to self.size
        '''
        device = queries.device

        # scale queries
        queries[:, :, 2] = (queries[:, :, 2] / H) * self.size[0]
        queries[:, :, 1] = (queries[:, :, 1] / W) * self.size[1]

        # scale gt_tracks
        gt_tracks_tmp = gt_tracks.clone()
        gt_tracks_tmp[:, :, :, 1] = (gt_tracks_tmp[:, :, :, 1] / H) * self.size[0]
        gt_tracks_tmp[:, :, :, 0] = (gt_tracks_tmp[:, :, :, 0] / W) * self.size[1]

        return queries, gt_tracks_tmp

    def forward(self, video, queries, gt_tracks, gt_visibility):
        '''
        Args:
            video: (B, T, C, H, W)
            queries: (B, N, 3), in format (t, x, y)
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

        # init memory
        collision_dist = torch.zeros(B, N, self.memory_size, self.embed_dim, device=device) # b n m c
        stream_dist = torch.zeros(B, N, self.memory_size, self.embed_dim, device=device) # b n m c
        vis_mask = torch.ones(B, N, self.memory_size, device=device, dtype=torch.bool) # b n m
        mem_mask = torch.ones(B, N, self.memory_size, device=device, dtype=torch.bool) # b n m

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
            vis_mask = torch.cat([vis_mask[:, :, 1:], (F.sigmoid(vis) < self.visibility_treshold).unsqueeze(-1)], dim=2) # b n m
        
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
        loss_cls = 0
        for i in range(self.num_layers):
            loss_cls += self.loss.classification_loss(corr[i], gt_tracks_scaled, gt_visibility, query_times)
        out["classification_loss"] = loss_cls / self.num_layers

        out["visibility_loss"] = self.loss.visibility_loss(visibility, gt_visibility, query_times)

        ref_tracks = ref_pts[-1].squeeze(-2) # b t n 2
        out["regression_loss"] = self.loss.regression_loss(offsets, ref_tracks, gt_tracks_scaled, gt_visibility, query_times)
        
        pred_tracks = ref_tracks + offsets # b t n 2
        out["uncertainty_loss"] = self.loss.uncertainty_loss(rhos, pred_tracks, gt_tracks_scaled, gt_visibility, query_times) * self.loss.lambda_unc

        loss_ref = 0
        num_k = 0
        for i in range(self.num_layers - 1):
            n_topk = ref_rho[i].shape[-1]
            for j in range(n_topk):
                num_k += 1
                loss_ref += self.loss.uncertainty_loss(ref_rho[i][:, :, :, j], ref_pts[i][:, :, :, j], gt_tracks_scaled, gt_visibility, query_times) * self.loss.lambda_ref
        out["uncertainty_loss_topk"] = loss_ref / num_k

        # outputs
        pred_tracks[..., 0] = (pred_tracks[..., 0] / self.size[1]) * W
        pred_tracks[..., 1] = (pred_tracks[..., 1] / self.size[0]) * H
        out['points'] = pred_tracks
        out["visibility"] = F.sigmoid(visibility) > self.visibility_treshold

        return out

    def inference(self, video, queries, K=20):
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
            )

            # update
            collision_dist = torch.cat([collision_dist[:, :, 1:], memory['collision'].unsqueeze(2)], dim=2) # b n m c
            stream_dist = torch.cat([stream_dist[:, :, 1:], memory['stream'].unsqueeze(2)], dim=2) # b n m c
            mem_mask = torch.cat([mem_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2) # b n m                          
            vis_mask = torch.cat([vis_mask[:, :, 1:], (F.sigmoid(vis) < self.visibility_treshold).unsqueeze(-1)], dim=2) # b n m

            ref_pts = out_t[-1]['reference_points'].squeeze(-2) # b n 2
            pred_tracks = ref_pts + offset # b n 2
            coord_pred.append(pred_tracks.unsqueeze(1)) # b 1 n 2
            vis_pred.append(vis.unsqueeze(1)) # b 1 n

        # output
        coord_pred = torch.cat(coord_pred, dim=1) # b t n 2
        vis_pred = torch.cat(vis_pred, dim=1) # b t n

        if self.extend_queries:
            coord_pred = coord_pred[:, :, :N] # b t n 2
            vis_pred = vis_pred[:, :, :N] # b t n

        coord_pred[:, :, :, 1] = (coord_pred[:, :, :, 1] / self.size[0]) * H
        coord_pred[:, :, :, 0] = (coord_pred[:, :, :, 0] / self.size[1]) * W

        out["points"] = coord_pred
        out["visibility"] = F.sigmoid(vis_pred) > self.visibility_treshold

        return out