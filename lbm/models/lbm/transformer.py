import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
USE_MMCV = True
if USE_MMCV:
    print('Using MMCV MSDeformAttn')
    from mmcv.ops import MultiScaleDeformableAttention
else:
    print('Using custom MSDeformAttn')
    from lbm.models.lbm.msdeformattn import MultiScaleDeformableAttention


class LBMTransformer(nn.Module):
    def __init__(self, size, stride, n_layer, hidden_dim, n_head, n_level, n_neighbor, n_mem, n_topk, offset_dim=2, dropout=0.1, random_mask_ratio=0, spa_corr_thre=1000, add_spatial_penalty=False):
        super().__init__()
        self.size = size
        self.stride = stride
        self.n_neighbor = n_neighbor
        self.random_mask_ratio = random_mask_ratio
        self.layers_n_topk = [n_topk,] + [4,] * (n_layer-2) + [1]

        H, W = size
        h, w = H // stride, W // stride
        self.register_buffer('spatial_shapes', torch.tensor([[h, w], [h//2, w//2], [h//4, w//4], [h//8, w//8]], dtype=torch.long))
        self.register_buffer('start_levels', torch.tensor([0, h*w, h*w + h*w//4, h*w + h*w//4 + h*w//16], dtype=torch.long))
        
        self.layers = nn.ModuleList([
            LBMTransformerLayer(size, stride, hidden_dim, n_head, n_level, n_neighbor, n_mem, dropout, spa_corr_thre, add_spatial_penalty)
            for i in range(n_layer)
        ])

        self.offset_layer = MultiScaleDeformableAttention(hidden_dim, n_head, n_level, n_neighbor, dropout=dropout, batch_first=True)
        self.offset_dropout = nn.Dropout(dropout)
        self.offset_norm = nn.LayerNorm(hidden_dim)
        self.offset_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.offset_norm2 = nn.LayerNorm(hidden_dim)
        self.offset_head = MLP(hidden_dim, out_dim=offset_dim, expansion_factor=1, dropout=0)
        self.offset_act = nn.Tanh()
            
        self.vis_layer = MultiScaleDeformableAttention(hidden_dim, n_head, n_level, n_neighbor, dropout=dropout, batch_first=True)
        self.vis_dropout = nn.Dropout(dropout)
        self.vis_norm = nn.LayerNorm(hidden_dim)
        self.vis_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.vis_norm2 = nn.LayerNorm(hidden_dim)
        self.vis_head = MLP(hidden_dim, out_dim=2, expansion_factor=1, dropout=0)

        self.collision_mem_layer = MultiScaleDeformableAttention(hidden_dim, n_head, n_level, num_points=4, dropout=dropout, batch_first=True)
        self.collision_mem_dropout = nn.Dropout(dropout)
        self.collision_mem_norm = nn.LayerNorm(hidden_dim)
        self.collision_mem_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.collision_mem_norm2 = nn.LayerNorm(hidden_dim)


    def get_multiscale_feats(self, feat):
        '''
        Args:
            feat: b c h w
        Returns:
            f_scales: [b h*w c, b h*w/4 c, b h*w/16 c, b h*w/64 c]
        '''
        b, c, h, w = feat.shape

        f1 = feat # b c h w
        f2 = F.avg_pool2d(f1, kernel_size=2, stride=2) # b c h/2 w/2
        f3 = F.avg_pool2d(f1, kernel_size=4, stride=4) # b c h/4 w/4
        f4 = F.avg_pool2d(f1, kernel_size=8, stride=8) # b c h/8 w/8

        f1 = f1.view(b, c, h*w).permute(0, 2, 1) # b h*w c
        f2 = f2.view(b, c, (h//2) * (w//2)).permute(0, 2, 1) # b h*w/4 c
        f3 = f3.view(b, c, (h//4) * (w//4)).permute(0, 2, 1) # b h*w/16 c
        f4 = f4.view(b, c, (h//8) * (w//8)).permute(0, 2, 1) # b h*w/64 c
        f_scales = [f1, f2, f3, f4]

        return f_scales
    

    def forward(self, query, feat, stream_dist, collision_dist, vis_mask, mem_mask, queried_now_or_before, last_pos=None):
        '''
        Args:
            query: b n c
            feat: b c h w
            stream_dist: b n m c, stream memory query
            collision_dist: b n m c, collision memory query
            vis_mask: b n m, visibility of query points
            mem_mask: b n m, memory mask
            queried_now_or_before: b n, queried now or before
            last_pos: b n 2 or None, last position of each query point
        '''
        b, c, h, w = feat.shape
        device = feat.device
        query_init = query.clone()

        out = []
        memory = {}
        feats = self.get_multiscale_feats(feat) # [b h*w c, b h*w/4 c, b h*w/16 c, b h*w/64 c]
        for i in range(len(self.layers)):
            corr, reference_points, reference_rho, query = self.layers[i](
                query, feats, stream_dist, collision_dist, vis_mask, mem_mask, self.random_mask_ratio,
                self.spatial_shapes, self.start_levels, self.layers_n_topk[i], last_pos
            )
            out.append({
                'corr': corr, # b n hw
                'reference_points': reference_points, # b n k 2
                'reference_rho': reference_rho, # b n k
            })

        # streaming memory
        memory['stream'] = query.clone() # b n c

        # offset
        feats_cat = torch.cat(feats, dim=1) # b h*w + h*w/4 + h*w/16 + h*w/64 c
        reference_points_norm = reference_points / torch.tensor(self.size[::-1], device=device) # b n k 2
        reference_points_norm = torch.clamp(reference_points_norm, 0, 1) # b n k 2
        
        reference_points_ms = reference_points_norm.unsqueeze(3).expand(-1, -1, -1, 4, -1) # b n k 4 2
        reference_points_ms = reference_points_ms.view(b, -1, 4, 2) # b n*k 4 2
        
        q_offset = query.clone()
        attn_output = self.offset_layer(q_offset, feats_cat, feats_cat, reference_points=reference_points_ms, spatial_shapes=self.spatial_shapes.to(device), level_start_index=self.start_levels.to(device))
        q_offset = q_offset + self.offset_dropout(attn_output)
        q_offset = self.offset_norm(q_offset)
        q_offset = q_offset + self.offset_ffn(q_offset)
        q_offset = self.offset_norm2(q_offset)
        offset = self.offset_head(q_offset)
        offset = self.offset_act(offset) * self.stride # b n 2

        # collision memory
        pred_points = reference_points + offset.unsqueeze(2) # b n 1 2
        pred_points_norm = pred_points / torch.tensor(self.size[::-1], device=device) # b n 1 2
        pred_points_norm = torch.clamp(pred_points_norm, 0, 1) # b n 1 2
        pred_points_ms = pred_points_norm.unsqueeze(3).expand(-1, -1, -1, 4, -1) # b n 1 4 2
        pred_points_ms = pred_points_ms.view(b, -1, 4, 2) # b n*1 4 2

        collision_dist = query.clone()
        attn_output = self.collision_mem_layer(collision_dist, feats_cat, feats_cat, reference_points=pred_points_ms, spatial_shapes=self.spatial_shapes.to(device), level_start_index=self.start_levels.to(device))
        collision_dist = collision_dist + self.collision_mem_dropout(attn_output)
        collision_dist = self.collision_mem_norm(collision_dist)
        collision_dist = collision_dist + self.collision_mem_ffn(collision_dist)
        collision_dist = self.collision_mem_norm2(collision_dist)

        update_mask = queried_now_or_before.unsqueeze(-1).expand(-1, -1, c) # b n c
        collision_dist = torch.where(update_mask, collision_dist, query_init) # b n c
        memory['collision'] = collision_dist # b n c


        # visibility
        attn_output = self.vis_layer(query, feats_cat, feats_cat, reference_points=reference_points_ms, spatial_shapes=self.spatial_shapes.to(device), level_start_index=self.start_levels.to(device))
        query = query + self.vis_dropout(attn_output)
        query = self.vis_norm(query)
        query = query + self.vis_ffn(query)
        query = self.vis_norm2(query)
        vis_rho = self.vis_head(query) # b n 2
        vis = vis_rho[..., 0] # b n
        rho = vis_rho[..., 1] # b n

        return out, memory, offset, vis, rho


class LBMTransformerLayer(nn.Module): 
    def __init__(self, size, stride, hidden_dim, n_head, n_level, n_neighbor, n_mem, dropout=0.1, spa_corr_thre=1000, add_spatial_penalty=False):
        super().__init__()
        self.size = size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.spa_corr_thre = spa_corr_thre
        self.add_spatial_penalty = add_spatial_penalty
        
        self.collision_attention = nn.MultiheadAttention(hidden_dim, n_head, dropout, batch_first=True)
        self.collision_dropout = nn.Dropout(dropout)
        self.collision_norm = nn.LayerNorm(hidden_dim)
        self.collision_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.collision_norm2 = nn.LayerNorm(hidden_dim)
        self.collision_time_emb = nn.Parameter(torch.zeros(1, n_mem+1, hidden_dim))
        nn.init.trunc_normal_(self.collision_time_emb, std=0.02)

        self.stream_attention = nn.MultiheadAttention(hidden_dim, n_head, dropout, batch_first=True)
        self.stream_dropout = nn.Dropout(dropout)
        self.stream_norm = nn.LayerNorm(hidden_dim)
        self.stream_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.stream_norm2 = nn.LayerNorm(hidden_dim)
        self.stream_time_emb = nn.Parameter(torch.zeros(1, n_mem+1, hidden_dim))
        nn.init.trunc_normal_(self.stream_time_emb, std=0.02)

        self.corr_conv = nn.Conv2d(n_level, 1, kernel_size=1, stride=1, padding=0)

        self.lattice_attention = MultiScaleDeformableAttention(hidden_dim, n_head, n_level, n_neighbor, dropout=dropout, batch_first=True)
        self.lattice_dropout = nn.Dropout(dropout)
        self.lattice_norm = nn.LayerNorm(hidden_dim)
        self.lattice_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.lattice_norm2 = nn.LayerNorm(hidden_dim)

        self.lattice_update = nn.MultiheadAttention(hidden_dim, n_head, dropout, batch_first=True)
        self.lattice_update_dropout = nn.Dropout(dropout)
        self.lattice_update_norm = nn.LayerNorm(hidden_dim)
        self.lattice_update_ffn = MLP(hidden_dim, hidden_dim, dropout=dropout)
        self.lattice_update_norm2 = nn.LayerNorm(hidden_dim)
        self.reference_rho = nn.Linear(hidden_dim, 1)

    def correlation(self, query, feats, size, last_pos):
        '''
        Args:
            query: b n c
            feats: [b h*w c, b h*w/4 c, b h*w/16 c, b h*w/64 c]
            size: (h, w)
            last_pos: b n 2, last position of each query point
        Returns:
            c: b n hw
        '''

        b, n, c = query.shape
        f1, f2, f3, f4 = feats
        h, w = size

        q_normalized = F.normalize(query, p=2, dim=-1)    # b n c

        c1 = torch.einsum("bnc,bpc->bnp", q_normalized, F.normalize(f1, p=2, dim=-1)) # b n hw
        c2 = torch.einsum("bnc,bpc->bnp", q_normalized, F.normalize(f2, p=2, dim=-1)) # b n hw/4
        c3 = torch.einsum("bnc,bpc->bnp", q_normalized, F.normalize(f3, p=2, dim=-1)) # b n hw/16
        c4 = torch.einsum("bnc,bpc->bnp", q_normalized, F.normalize(f4, p=2, dim=-1)) # b n hw/64

        c1 = c1.view(b*n, 1, h, w)  # b n hw
        c2 = c2.view(b*n, 1, (h//2), (w//2))
        c3 = c3.view(b*n, 1, (h//4), (w//4))
        c4 = c4.view(b*n, 1, (h//8), (w//8))

        c1 = F.interpolate(c1, size=(h, w), mode="bilinear", align_corners=False)  # bn 1 h w
        c2 = F.interpolate(c2, size=(h, w), mode="bilinear", align_corners=False)  # bn 1 h w
        c3 = F.interpolate(c3, size=(h, w), mode="bilinear", align_corners=False)  # bn 1 h w
        c4 = F.interpolate(c4, size=(h, w), mode="bilinear", align_corners=False)  # bn 1 h w

        c = [c1, c2, c3, c4]
        c = torch.cat(c, dim=1) # bn level h w
        c = self.corr_conv(c) + c[:, 0:1] # bn 1 h w
        c = c.view(b, n, h*w)

        if self.add_spatial_penalty:
            y_coords = torch.arange(h, device=c.device).float()
            x_coords = torch.arange(w, device=c.device).float()
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            grid_coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # hw 2
            
            last_pos = last_pos / self.stride  # n 2
            last_pos = last_pos.unsqueeze(-2)  # n 1 2
            grid_coords = grid_coords.unsqueeze(0).expand(n, -1, -1)  # n hw 2
            dist = torch.norm(last_pos - grid_coords, dim=-1)  # b n hw
            
            dist_weight = torch.exp(-dist / self.spa_corr_thre)  # b n hw
            c = c * dist_weight

        return c


    def forward(self, query, feats, stream_dist, collision_dist, vis_mask, mem_mask, random_mask_ratio, spatial_shapes, start_levels, n_topk, last_pos):
        '''
        Args:
            query: b n c
            feats: [b h*w c, b h*w/4 c, b h*w/16 c, b h*w/64 c]
            stream_dist: b n m c, stream memory query
            collision_dist: b n m c, collision memory query
            vis_mask: b n m, visibility of query points
            mem_mask: b n m, memory mask
            spatial_shapes: [h w, h/2 w/2, h/4 w/4, h/8 w/8]
            start_levels: [0, h*w, h*w+h*w/4, h*w+h*w/4+h*w/16]
            n_topk: int, number of top-k reference points
            last_pos: b n 2, last position of each query point
        '''
        b, n, c = query.shape
        H, W = self.size
        h, w = H // self.stride, W // self.stride
        device = query.device

        ignore_mask = mem_mask | vis_mask # b n m
        random_mask = torch.rand(mem_mask.shape, device=device) < random_mask_ratio
        ignore_mask = ignore_mask | random_mask
        fully_ignored = ignore_mask.all(dim=-1) # b n
        if fully_ignored.all():
            q_streaming = query
        else:
            # collision
            q_collision = query.clone()
            collision_dist = rearrange(collision_dist, 'b n m c -> (b n) m c')
            q = rearrange(query, 'b n c -> (b n) 1 c')
            mask = rearrange(ignore_mask, 'b n m -> (b n) m')

            valid_indices = ~fully_ignored.view(-1)
            q_valid = q[valid_indices]
            kv_valid = collision_dist[valid_indices]
            mask_valid = mask[valid_indices]

            q_valid = q_valid + self.collision_time_emb[:, -1:]
            k_valid = kv_valid + self.collision_time_emb[:, :-1]
            v_valid = kv_valid

            attn_output, _ = self.collision_attention(q_valid, k_valid, v_valid, key_padding_mask=mask_valid, need_weights=False)
            q_valid = q_valid + self.collision_dropout(attn_output)
            q_valid = self.collision_norm(q_valid)
            q_valid = q_valid + self.collision_ffn(q_valid)
            q_valid = self.collision_norm2(q_valid)

            q_collision = rearrange(q_collision, 'b n c -> (b n) 1 c')
            q_collision[valid_indices] = q_valid
            q_collision = rearrange(q_collision, '(b n) 1 c -> b n c', b=b, n=n)
        
            # streaming 
            q_streaming = q_collision.clone()
            stream_dist = rearrange(stream_dist, 'b n m c -> (b n) m c')
            q = rearrange(q_collision, 'b n c -> (b n) 1 c')

            q_valid = q[valid_indices]
            kv_valid = stream_dist[valid_indices]

            q_valid = q_valid + self.stream_time_emb[:, -1:]
            k_valid = kv_valid + self.stream_time_emb[:, :-1]
            v_valid = kv_valid

            attn_output, _ = self.stream_attention(q_valid, k_valid, v_valid, key_padding_mask=mask_valid, need_weights=False)
            q_valid = q_valid + self.stream_dropout(attn_output)
            q_valid = self.stream_norm(q_valid)
            q_valid = q_valid + self.stream_ffn(q_valid)
            q_valid = self.stream_norm2(q_valid)

            q_streaming = rearrange(q_streaming, 'b n c -> (b n) 1 c')
            q_streaming[valid_indices] = q_valid
            q_streaming = rearrange(q_streaming, '(b n) 1 c -> b n c', b=b, n=n)

        # correlation
        corr = self.correlation(q_streaming, feats, size=(h, w), last_pos=last_pos) # b n hw
        top_k_indices = torch.topk(corr, n_topk, dim=-1)[1] # b n k
        reference_points = indices_to_coords(top_k_indices, self.size, self.stride) # b n k
        reference_points_norm = reference_points / torch.tensor(self.size[::-1], device=device) # b n k 2
        reference_points_norm = torch.clamp(reference_points_norm, 0, 1) # b n k 2
        
        reference_points_ms = reference_points_norm.unsqueeze(3).expand(-1, -1, -1, 4, -1) # b n k 4 2
        reference_points_ms = reference_points_ms.view(b, -1, 4, 2) # b n*k 4 2

        # lattice 
        feats_cat = torch.cat(feats, dim=1) # b h*w + h*w/4 + h*w/16 + h*w/64 c
        q_lattice = q_streaming.clone().unsqueeze(2).expand(-1, -1, n_topk, -1) # b n k c
        q_lattice = rearrange(q_lattice, 'b n k c -> b (n k) c')

        attn_output = self.lattice_attention(q_lattice, feats_cat, feats_cat, reference_points=reference_points_ms, spatial_shapes=spatial_shapes, level_start_index=start_levels)
        q_lattice = q_lattice + self.lattice_dropout(attn_output)
        q_lattice = self.lattice_norm(q_lattice)
        q_lattice = q_lattice + self.lattice_ffn(q_lattice)
        q_lattice = self.lattice_norm2(q_lattice)

        q_lattice = rearrange(q_lattice, 'b (n k) c -> b n k c', n=n, k=n_topk)
        reference_rho = self.reference_rho(q_lattice).squeeze(-1) # b n k

        # lattice update
        random_mask = torch.rand(b*n, n_topk, device=device) < random_mask_ratio
        q_streaming = rearrange(q_streaming, 'b n c -> (b n) 1 c')
        q_lattice = rearrange(q_lattice, 'b n k c -> (b n) k c')
        attn_output, _ = self.lattice_update(q_streaming, q_lattice, q_lattice, need_weights=False)
        q_streaming = q_streaming + self.lattice_update_dropout(attn_output)
        q_streaming = self.lattice_update_norm(q_streaming)
        q_streaming = q_streaming + self.lattice_update_ffn(q_streaming)
        q_streaming = self.lattice_update_norm2(q_streaming)
        q_streaming = rearrange(q_streaming, '(b n) 1 c -> b n c', b=b, n=n)

        return corr, reference_points, reference_rho, q_streaming


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * expansion_factor, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def indices_to_coords(indices, size, ps):
    '''
    Args:
        indices: (B, T, N)
        size: (H, W)
        ps: int, patch size
    Returns:
        coordinates: (B, T, N, 2) in [0, W] and [0, H] range
    '''

    B, T, N = indices.shape
    H, W = size

    num_columns = W // ps

    
    rows = indices // num_columns
    cols = indices % num_columns
    
    y_coords = rows * ps + 0.5 * ps
    x_coords = cols * ps + 0.5 * ps
    
    coordinates = torch.stack((x_coords, y_coords), dim=-1)

    assert coordinates.shape == (B, T, N, 2)
    assert torch.all(coordinates[:, :, :, 0] <= W)
    assert torch.all(coordinates[:, :, :, 1] <= H)
    
    return coordinates