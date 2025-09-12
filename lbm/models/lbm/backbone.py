import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Optional

from torchvision.models import resnet18, ResNet18_Weights
from lbm.utils.convnext import ConvNeXt

class Backbone(nn.Module):
    def __init__(self, size, stride, embed_dim):
        super().__init__()
        proj_dim = 64
        out_dim = 256
        self.size = size
        self.stride = stride
        self.embedding_dim = embed_dim
        self.normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        # fusion
        self.proj1 = nn.Conv2d(64, proj_dim, kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(128, proj_dim, kernel_size=1, stride=1, padding=0)
        self.proj3 = nn.Conv2d(256, proj_dim, kernel_size=1, stride=1, padding=0)

        self.fusion = nn.Sequential(
            nn.Conv2d(proj_dim*3, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

        # positional embedding
        self.P = (self.size[0] // self.stride) * (self.size[1] // self.stride)
        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride

        self.frame_pos_embedding = nn.Parameter(torch.zeros(1, self.embedding_dim, self.H_prime, self.W_prime))  # (1, D, H, W)
        nn.init.trunc_normal_(self.frame_pos_embedding, std=0.02)


    def forward_backbone(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        
        c1 = self.proj1(c1) # stride = 4
        c2 = self.proj2(c2) # stride = 8
        c3 = self.proj3(c3) # stride = 16
    
        h1, w1 = c1.shape[2], c1.shape[3] # stride = 4

        c2 = F.interpolate(c2, size=(h1, w1), mode='bilinear', align_corners=False)
        c3 = F.interpolate(c3, size=(h1, w1), mode='bilinear', align_corners=False)

        f1 = torch.cat([c1, c2, c3], dim=1)
        f1 = self.fusion(f1) # stride = 4

        return f1
        
    def get_query_tokens(self, features, queries):
        '''
        Args:
            features: (B, C, H4, W4)
            queries: (B, N, 3) where 3 is (t, y, x)
        '''

        B, N, _ = queries.shape
        C = features.shape[1]
        device = features.device

        assert C == self.embedding_dim, f"Token embedding dim: {C} expected: {self.embedding_dim}"

        H4, W4 = features.shape[-2], features.shape[-1]
        features = features.view(B, -1, C, H4, W4) # b t c h w

        query_features = torch.zeros(B, N, C, device=device) # b n c 
        query_points_reshaped = queries.view(-1, 3) # bn 3
        t, x, y = query_points_reshaped[:, 0].long(), query_points_reshaped[:, 1], query_points_reshaped[:, 2] # bn

        source_frame_f = features[torch.arange(B).repeat_interleave(N), t].view(-1, C, H4, W4) # bn c h w
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1

        grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N, 1, 1, 2).to(device)
        sampled = F.grid_sample(source_frame_f, grid, mode='bilinear', padding_mode='border', align_corners=False)
        query_features.view(-1, C)[:] = sampled.reshape(-1, C) # bn c
        query_features = query_features.view(B, N, C)

        return query_features
    
    def forward(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255]
            queries: (B, N, 3) where 3 is (t, x, y)
        Returns:
            tokens: (B, T, P, C)
            query_features: (B, N, C)
        '''

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape

        # Normalize & Resize the video 
        video_flat = video.view(B * T, C, H, W) / 255.0 # bt c h w 
        video_flat = F.interpolate(video_flat, size=self.size, mode="bilinear", align_corners=False)
        video_flat = self.normalization(video_flat) # to [-1, 1]

        # Forward backbone 
        f = self.forward_backbone(video_flat) # bt c h w
        f = f + self.frame_pos_embedding # bt c h w

        q = self.get_query_tokens(f, queries) # b n c
        f = f.permute(0, 2, 3, 1)  # bt h w c
        tokens = f.view(B, T, self.P, self.embedding_dim) # b t hw c

        assert tokens.shape == (B, T, self.P, self.embedding_dim), f"Tokens shape: {tokens.shape}, expected: {(B, T, self.P, self.embedding_dim)}"
        assert q.shape == (B, N, self.embedding_dim), f"Queries shape: {queries.shape}, expected: {(B, N, self.embedding_dim)}"

        return tokens, q
    
    # Online query sampling
    def sample_queries_online(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255]
            queries: (B, N, 3) where 3 is (t, x, y)
        '''

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape
        device = video.device


        query_features = torch.zeros(B, N, self.embedding_dim, device=queries.device) # b n c
        for t in range(T):
            queries_of_this_time = queries[:, :, 0] == t # b n
            query_positions = queries[queries_of_this_time].view(B, -1, 3)[:, :, 1:] # b n' 2
            N_prime = query_positions.shape[1]

            # No queries sampled at this time
            if N_prime == 0:
                continue

            x, y = query_positions[:, :, 0], query_positions[:, :, 1] # b n'
            x_grid = (x / self.size[1]) * 2 - 1
            y_grid = (y / self.size[0]) * 2 - 1
            grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N_prime, 1, 1, 2).to(device) # bn' 1 1 2

            video_frame = video[:, t] / 255.0 # b c h w
            video_frame = F.interpolate(video_frame, size=self.size, mode="bilinear", align_corners=False)
            video_frame = self.normalization(video_frame) # [-1, 1]

            f = self.forward_backbone(video_frame) # b c h w

            H_prime, W_prime = f.shape[-2], f.shape[-1]
            assert H_prime == self.H_prime and W_prime == self.W_prime, f"Frame shape: {(H_prime, W_prime)}, expected: {(self.H_prime, self.W_prime)}"

            f = f + self.frame_pos_embedding # b c h w
            C = f.shape[1]
            f = f.unsqueeze(1).expand(-1, N_prime, -1, -1, -1).reshape(B * N_prime, C, H_prime, W_prime)  # bn' c h w
            sampled = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=False) # bn' c 1 1
            query_features[queries_of_this_time] = sampled.view(B, N_prime, C) # b n' c

        return query_features
    
    
    def encode_frames_online(self, frames):
        '''
        Args:
            frames: (B, C, H, W) in range [0, 255]
        '''

        frames = frames.clone() / 255.0 # b c h w
        frames = F.interpolate(frames, size=self.size, mode="bilinear", align_corners=False)
        frames = self.normalization(frames) 

        f = self.forward_backbone(frames) # b c h w

        f = f + self.frame_pos_embedding # b c h w
        C = f.shape[1]

        f = f.permute(0, 2, 3, 1) # b h w c
        f = f.reshape(f.shape[0], -1, C) # b hw c

        return f
    

class LBMConvNeXt(nn.Module):
    def __init__(self, size, stride, embed_dim, weight_path='checkpoints/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth'):
        super().__init__()
        proj_dim = 96
        out_dim = embed_dim
        self.size = size
        self.stride = stride
        self.embedding_dim = embed_dim
        self.normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # backbone: ConvNeXt tiny, we will only use first 3 stages
        convnext = ConvNeXt(in_chans=3)

        # optionally load weights
        if weight_path is not None:
            try:
                ckpt = torch.load(weight_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
                convnext.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[LBMConvNeXt] Failed to load weights from {weight_path}: {e}")

        # keep only first three stages (stride 4, 8, 16)
        # ConvNeXt has: downsample_layers[0] (stride 4 stem) + stages[0]
        # then each of the next two: downsample_layers[i] + stages[i]
        self.stage1 = nn.Sequential(
            convnext.downsample_layers[0],  # stride 4
            convnext.stages[0],
        )
        self.stage2 = nn.Sequential(
            convnext.downsample_layers[1],  # stride 8
            convnext.stages[1],
        )
        self.stage3 = nn.Sequential(
            convnext.downsample_layers[2],  # stride 16
            convnext.stages[2],
        )

        # dims for tiny: [96, 192, 384]
        dims = [96, 192, 384]

        # fusion (project each to same dim and fuse)
        self.proj1 = nn.Conv2d(dims[0], proj_dim, kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(dims[1], proj_dim, kernel_size=1, stride=1, padding=0)
        self.proj3 = nn.Conv2d(dims[2], proj_dim, kernel_size=1, stride=1, padding=0)

        self.fusion = nn.Sequential(
            nn.Conv2d(proj_dim * 3, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

        # positional embedding
        self.P = (self.size[0] // self.stride) * (self.size[1] // self.stride)
        self.H_prime = self.size[0] // self.stride
        self.W_prime = self.size[1] // self.stride

        self.frame_pos_embedding = nn.Parameter(
            torch.zeros(1, self.embedding_dim, self.H_prime, self.W_prime)
        )  # (1, D, H, W)
        nn.init.trunc_normal_(self.frame_pos_embedding, std=0.02)

    def forward_backbone(self, x):
        # x: (B, C, H, W)
        c1 = self.stage1(x)   # (B, 96, H/4, W/4)
        c2 = self.stage2(c1)  # (B, 192, H/8, W/8)
        c3 = self.stage3(c2)  # (B, 384, H/16, W/16)

        c1 = self.proj1(c1)  # stride = 4
        c2 = self.proj2(c2)  # stride = 8
        c3 = self.proj3(c3)  # stride = 16

        h1, w1 = c1.shape[2], c1.shape[3]  # stride = 4
        c2 = F.interpolate(c2, size=(h1, w1), mode='bilinear', align_corners=True)
        c3 = F.interpolate(c3, size=(h1, w1), mode='bilinear', align_corners=True)

        f1 = torch.cat([c1, c2, c3], dim=1)
        f1 = self.fusion(f1)  # stride = 4, channels = embedding_dim
        return f1

    def get_query_tokens(self, features, queries):
        '''
        Args:
            features: (B, C, H4, W4)
            queries: (B, N, 3) where 3 is (t, y, x)
        '''

        B, N, _ = queries.shape
        C = features.shape[1]
        device = features.device

        assert C == self.embedding_dim, f"Token embedding dim: {C} expected: {self.embedding_dim}"

        H4, W4 = features.shape[-2], features.shape[-1]
        features = features.view(B, -1, C, H4, W4)  # b t c h w

        query_features = torch.zeros(B, N, C, device=device)  # b n c
        query_points_reshaped = queries.view(-1, 3)  # bn 3
        t, x, y = (
            query_points_reshaped[:, 0].long(),
            query_points_reshaped[:, 1],
            query_points_reshaped[:, 2],
        )  # bn

        source_frame_f = features[torch.arange(B).repeat_interleave(N), t].view(-1, C, H4, W4)  # bn c h w
        x_grid = (x / self.size[1]) * 2 - 1
        y_grid = (y / self.size[0]) * 2 - 1

        grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N, 1, 1, 2).to(device)
        sampled = F.grid_sample(
            source_frame_f, grid, mode='bilinear', padding_mode='border', align_corners=False
        )
        query_features.view(-1, C)[:] = sampled.reshape(-1, C)  # bn c
        query_features = query_features.view(B, N, C)

        return query_features

    def forward(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255]
            queries: (B, N, 3) where 3 is (t, x, y)
        Returns:
            tokens: (B, T, P, C)
            query_features: (B, N, C)
        '''

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape

        # Normalize & Resize the video
        video_flat = video.view(B * T, C, H, W) / 255.0  # bt c h w
        video_flat = F.interpolate(video_flat, size=self.size, mode="bilinear", align_corners=True)
        video_flat = self.normalization(video_flat)  # to [-1, 1]

        # Forward backbone
        f = self.forward_backbone(video_flat)  # bt c h w
        f = f + self.frame_pos_embedding  # bt c h w

        q = self.get_query_tokens(f, queries)  # b n c
        f = f.permute(0, 2, 3, 1)  # bt h w c
        tokens = f.view(B, T, self.P, self.embedding_dim)  # b t hw c

        assert tokens.shape == (B, T, self.P, self.embedding_dim), (
            f"Tokens shape: {tokens.shape}, expected: {(B, T, self.P, self.embedding_dim)}"
        )
        assert q.shape == (B, N, self.embedding_dim), (
            f"Queries shape: {queries.shape}, expected: {(B, N, self.embedding_dim)}"
        )

        return tokens, q

    # Online query sampling
    def sample_queries_online(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255]
            queries: (B, N, 3) where 3 is (t, x, y)
        '''

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape
        device = video.device

        query_features = torch.zeros(B, N, self.embedding_dim, device=queries.device)  # b n c
        for t_idx in range(T):
            queries_of_this_time = queries[:, :, 0] == t_idx  # b n
            query_positions = queries[queries_of_this_time].view(B, -1, 3)[:, :, 1:]  # b n' 2
            N_prime = query_positions.shape[1]

            # No queries sampled at this time
            if N_prime == 0:
                continue

            x, y = query_positions[:, :, 0], query_positions[:, :, 1]  # b n'
            x_grid = (x / self.size[1]) * 2 - 1
            y_grid = (y / self.size[0]) * 2 - 1
            grid = torch.stack([x_grid, y_grid], dim=-1).view(B * N_prime, 1, 1, 2).to(device)  # bn' 1 1 2

            video_frame = video[:, t_idx] / 255.0  # b c h w
            video_frame = F.interpolate(video_frame, size=self.size, mode="bilinear", align_corners=True)
            video_frame = self.normalization(video_frame)  # [-1, 1]

            f = self.forward_backbone(video_frame)  # b c h w

            H_prime, W_prime = f.shape[-2], f.shape[-1]
            assert H_prime == self.H_prime and W_prime == self.W_prime, (
                f"Frame shape: {(H_prime, W_prime)}, expected: {(self.H_prime, self.W_prime)}"
            )

            f = f + self.frame_pos_embedding  # b c h w
            Cc = f.shape[1]
            f = (
                f.unsqueeze(1)
                .expand(-1, N_prime, -1, -1, -1)
                .reshape(B * N_prime, Cc, H_prime, W_prime)
            )  # bn' c h w
            sampled = F.grid_sample(
                f, grid, mode='bilinear', padding_mode='border', align_corners=False
            )  # bn' c 1 1
            query_features[queries_of_this_time] = sampled.view(B, N_prime, Cc)  # b n' c

        return query_features

    def encode_frames_online(self, frames):
        '''
        Args:
            frames: (B, C, H, W) in range [0, 255]
        '''

        frames = frames.clone() / 255.0  # b c h w
        frames = F.interpolate(frames, size=self.size, mode="bilinear", align_corners=True)
        frames = self.normalization(frames)

        f = self.forward_backbone(frames)  # b c h w

        f = f + self.frame_pos_embedding  # b c h w
        Cc = f.shape[1]

        f = f.permute(0, 2, 3, 1)  # b h w c
        f = f.reshape(f.shape[0], -1, Cc)  # b hw c

        return f