import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

from torchvision.models import resnet18, ResNet18_Weights


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

        self.frame_pos_embedding = nn.Parameter(torch.zeros(1, self.embedding_dim, self.H_prime, self.W_prime), requires_grad=True)  # (1, D, H, W)
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
            queries: (B, N, 3) where 3 is (t, x, y)
                  or (B, N, 4) where 4 is (t, x, y, d)
        '''

        B, N, D = queries.shape
        C = features.shape[1]
        device = features.device

        assert C == self.embedding_dim, f"Token embedding dim: {C} expected: {self.embedding_dim}"

        H4, W4 = features.shape[-2], features.shape[-1]
        features = features.view(B, -1, C, H4, W4) # b t c h w

        query_features = torch.zeros(B, N, C, device=device) # b n c 
        query_points_reshaped = queries.view(-1, D) # bn d
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
            queries: (B, N, 3) where 3 is (t, y, x)
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
    

class Backbone3D(Backbone):
    def __init__(self, size, stride, embed_dim):
        super().__init__(size, stride, embed_dim)
        self.beta = 0.1

        # backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        pretrained_conv1 = backbone.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        new_conv1.weight.data[:, :3, :, :] = pretrained_conv1 # copy rgb weights
        new_conv1.weight.data[:, 3:4, :, :] = pretrained_conv1.mean(dim=1, keepdim=True)
        
        self.layer1 = nn.Sequential(
            new_conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255], C = 4: RGB + Depth
            queries: (B, N, 3) where 3 is (t, y, x)
        '''

        B, T, C, H, W = video.shape
        B, N, _ = queries.shape

        # Normalize & Resize the video 
        video_flat = video.view(B * T, C, H, W) # bt c h w 
        video_flat = F.interpolate(video_flat, size=self.size, mode="bilinear", align_corners=False)
        rgb_norm = video_flat[:, :3] / 255.0 # bt 3 h w
        rgb_norm = self.normalization(rgb_norm) # to [-1, 1]
        
        # Depth normalization: e^(-beta*d) 
        depth = video_flat[:, 3:4] # bt 1 h w

        # depth_exp = torch.exp(-self.beta * depth) # bt 1 h w, range [0,1]
        # depth_norm = depth_exp * 2 - 1 # to [-1, 1]
        depth_median = torch.median(depth)
        depth_scale = torch.abs(depth - depth_median).sum() / depth.numel()
        depth_norm = (depth - depth_median) / depth_scale

        video_flat = torch.cat([rgb_norm, depth_norm], dim=1) # bt 4 h w

        # Forward backbone 
        f = self.forward_backbone(video_flat) # bt c h w
        f = f + self.frame_pos_embedding # bt c h w

        q = self.get_query_tokens(f, queries) # b n c
        f = f.permute(0, 2, 3, 1)  # bt h w c
        tokens = f.view(B, T, self.P, self.embedding_dim) # b t hw c

        assert tokens.shape == (B, T, self.P, self.embedding_dim), f"Tokens shape: {tokens.shape}, expected: {(B, T, self.P, self.embedding_dim)}"
        assert q.shape == (B, N, self.embedding_dim), f"Queries shape: {queries.shape}, expected: {(B, N, self.embedding_dim)}"

        return tokens, q

    def encode_frames_online(self, frames, framedepth):
        '''
        Args:
            frames: (B, C, H, W) in range [0, 255]
            framedepth: (B, 1, H, W) in range [0, 1]
        '''

        frames = frames.clone() / 255.0 # b c h w
        framedepth = framedepth.clone() # b 1 h w
        
        # Resize both frames and depth to target size
        frames = F.interpolate(frames, size=self.size, mode="bilinear", align_corners=False)
        framedepth = F.interpolate(framedepth, size=self.size, mode="bilinear", align_corners=False)
        
        # Normalize RGB frames
        frames = self.normalization(frames) # to [-1, 1]
        
        # Depth normalization: consistent with forward method
        depth_median = torch.median(framedepth)
        depth_scale = torch.abs(framedepth - depth_median).sum() / framedepth.numel()
        depth_norm = (framedepth - depth_median) / depth_scale
        
        # Concatenate RGB and depth
        frames_4d = torch.cat([frames, depth_norm], dim=1) # b 4 h w

        # Forward backbone 
        f = self.forward_backbone(frames_4d) # b c h w

        f = f + self.frame_pos_embedding # b c h w
        C = f.shape[1]

        f = f.permute(0, 2, 3, 1) # b h w c
        f = f.reshape(f.shape[0], -1, C) # b hw c

        return f

    def sample_queries_online(self, video, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255], C = 4: RGB + Depth
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

            video_frame = video[:, t] # b c h w
            video_frame = F.interpolate(video_frame, size=self.size, mode="bilinear", align_corners=False)
            
            # Normalize RGB and depth separately
            rgb_frame = video_frame[:, :3] / 255.0 # b 3 h w
            rgb_frame = self.normalization(rgb_frame) # to [-1, 1]
            
            depth_frame = video_frame[:, 3:4] # b 1 h w
            depth_median = torch.median(depth_frame)
            depth_scale = torch.abs(depth_frame - depth_median).sum() / depth_frame.numel()
            depth_norm = (depth_frame - depth_median) / depth_scale

            # assert inf or nan
            assert not torch.isinf(depth_norm).any(), "Depth normalization contains inf"
            assert not torch.isnan(depth_norm).any(), "Depth normalization contains nan"
            
            video_frame_4d = torch.cat([rgb_frame, depth_norm], dim=1) # b 4 h w

            f = self.forward_backbone(video_frame_4d) # b c h w

            H_prime, W_prime = f.shape[-2], f.shape[-1]
            assert H_prime == self.H_prime and W_prime == self.W_prime, f"Frame shape: {(H_prime, W_prime)}, expected: {(self.H_prime, self.W_prime)}"

            f = f + self.frame_pos_embedding # b c h w
            C = f.shape[1]
            f = f.unsqueeze(1).expand(-1, N_prime, -1, -1, -1).reshape(B * N_prime, C, H_prime, W_prime)  # bn' c h w
            sampled = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=False) # bn' c 1 1
            query_features[queries_of_this_time] = sampled.view(B, N_prime, C) # b n' c

        return query_features


class BackboneV2(nn.Module):
    def __init__(self, size, stride, n_dim):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=4,
            out_channels=n_dim,
            kernel_size=stride,
            stride=stride,
        )
        h = self.size[0] // self.stride
        w = self.size[1] // self.stride
        self.pos_embed = nn.Parameter(torch.zeros(1, n_dim, h, w), requires_grad=True)

    def forward(self, video, videodepth, queries):
        '''
        Args:
            video: (B, T, C, H, W) in range [0, 255]
            videodepth: (B, T, 1, H, W) 
            queries: (B, N, 3) where 3 is (t, x, y)
        '''

        B, T, C, H, W = video.shape
        video = rearrange(video, "b t c h w -> (b t) c h w")
        embed = self.patch_embed(video)
        
        
        
        