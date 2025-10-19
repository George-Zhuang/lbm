import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from lbm.models.lbm.lbm import LBM
from lbm.models.lbm.object_tracker import LBMObjTracker


class LBM_online(LBM):
    def __init__(self, args, with_obj_tracker=False):
        super().__init__(args=args)
        self.num_query = None
        self.t = 0
        self.N = 0
        self.query = None
        self.query_points = None
        self.collision_dist = None
        self.stream_dist = None
        self.vis_mask = None
        self.mem_mask = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if with_obj_tracker:
            self.objtracker = LBMObjTracker(args)

    def init(self, query_points, first_frame):
        '''
        Initializes the model with query points and the first frame.

        Args:
            query_points: (N, 2) in xy format, where N is the number of points
            first_frame: (1, C, H, W) video frame 
        '''
        self.query_points = query_points
        self.N = query_points.shape[0]
        self.img_size = first_frame.shape[-2:]
        self.device = first_frame.device

        # Encode the frame
        f_t = self.backbone.encode_frames_online(first_frame) # 1 c h w
        f_t = rearrange(f_t, 'b (h w) c -> b c h w', h=self.h, w=self.w) # 1 c h w

        update_mask = torch.ones(self.N, device=self.device, dtype=torch.bool)
        remove_mask = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        self.update(self.query_points, f_t, update_mask, remove_mask)

    def update(self, query_points, frame_feat, update_mask, remove_mask):
        '''
        Updates query features and memory with new points.
        
        Args:
            query_points: (N, 2) in xy format, where N is the number of points
            frame_feat: (1, C, H, W) video frame features
            update_mask: (N, ) boolean mask indicating which queries to update
        '''
        N = query_points.shape[0]

        if N > 0:
            remove_mask = remove_mask[:self.N]
            N_prev = (~remove_mask).sum().item()
            N_update = update_mask.sum().item()
            
            # Scale and normalize points for grid sampling
            update_points = query_points[update_mask]  # (N_update, 2)
            H, W = self.img_size
            update_points = update_points.clone()
            update_points[:, 1] = (update_points[:, 1] / H) * self.size[0]
            update_points[:, 0] = (update_points[:, 0] / W) * self.size[1]
            
            # Create sampling grid
            x, y = update_points[:, 0], update_points[:, 1]
            x_grid = (x / self.size[1]) * 2 - 1
            y_grid = (y / self.size[0]) * 2 - 1
            grid = torch.stack([x_grid, y_grid], dim=-1).view(N_update, 1, 1, 2)

            # Sample features at query points
            frame_feat_expanded = frame_feat.expand(N_update, -1, -1, -1)
            update_query = F.grid_sample(frame_feat_expanded, grid, mode='bilinear', 
                                        padding_mode='border', align_corners=False)
            update_query = update_query.reshape(1, N_update, self.embed_dim)

            # Initialize new tensors with increased capacity
            query = torch.zeros(1, N, self.embed_dim, device=self.device)
            collision_dist = torch.zeros(1, N, self.memory_size, self.embed_dim, device=self.device)
            stream_dist = torch.zeros(1, N, self.memory_size, self.embed_dim, device=self.device)
            vis_mask = torch.ones(1, N, self.memory_size, device=self.device, dtype=torch.bool)
            mem_mask = torch.ones(1, N, self.memory_size, device=self.device, dtype=torch.bool)

            # Copy existing data
            if self.query is not None and N_prev > 0:
                try:
                    query[:, :N_prev] = self.query[:, ~remove_mask]
                    query[:, update_mask] = update_query
                    collision_dist[:, :N_prev] = self.collision_dist[:, ~remove_mask]
                    stream_dist[:, :N_prev] = self.stream_dist[:, ~remove_mask]
                    vis_mask[:, :N_prev] = self.vis_mask[:, ~remove_mask]
                    mem_mask[:, :N_prev] = self.mem_mask[:, ~remove_mask]
                except:
                    print("Error in copying existing data. Check tensor shapes.")
            else:
                query = update_query

            # Update instance variables
            self.query = query
            self.query_points = query_points
            self.collision_dist = collision_dist
            self.stream_dist = stream_dist
            self.vis_mask = vis_mask
            self.mem_mask = mem_mask

        self.N = N

    def online_forward(self, frame):
        '''
        Online forward with given points on the first frame.

        Args:
            frame: (1, C, H, W) video frame to extract features from
        '''

        H, W = frame.shape[-2], frame.shape[-1]

        # Encode the frame
        f_t = self.backbone.encode_frames_online(frame) # 1 c h w
        f_t = rearrange(f_t, 'b (h w) c -> b c h w', h=self.h, w=self.w) # 1 c h w

        query_times = torch.zeros([1, self.N], device=self.device, dtype=torch.long)
        queried_now_or_before = (query_times <= self.t)

        # forward
        out_t, memory, offset, vis, rho = self.transformer(
            query=self.query.clone(), 
            feat=f_t, 
            stream_dist=self.stream_dist.clone(), 
            collision_dist=self.collision_dist.clone(), 
            vis_mask=self.vis_mask.clone(), 
            mem_mask=self.mem_mask.clone(), 
            queried_now_or_before=queried_now_or_before,
        )

        # update memory
        self.collision_dist = torch.cat([self.collision_dist[:, :, 1:], memory['collision'].unsqueeze(2)], dim=2) # b n m c
        self.stream_dist = torch.cat([self.stream_dist[:, :, 1:], memory['stream'].unsqueeze(2)], dim=2) # b n m c
        self.mem_mask = torch.cat([self.mem_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2) # b n m                          
        self.vis_mask = torch.cat([self.vis_mask[:, :, 1:], (F.sigmoid(vis) < self.visibility_treshold).unsqueeze(-1)], dim=2) # b n m

        self.t += 1

        ref_tracks = out_t[-1]['reference_points'].squeeze(-2, -4) # n 2
        coord_pred = ref_tracks + offset[0] # n 2
        vis_pred = F.sigmoid(vis)[0] > self.visibility_treshold
        rho_pred = F.sigmoid(rho)[0]

        coord_pred[:, 1] = (coord_pred[:, 1] / self.size[0]) * H
        coord_pred[:, 0] = (coord_pred[:, 0] / self.size[1]) * W

        return coord_pred, vis_pred, rho_pred, f_t

    def online_forward_obj(self, frame, bboxes, scores, labels):
        '''
        Online forward for object tracking.

        Args:
            frame: (1, C, H, W) video frame to extract features from
            bboxes: (M, 4) in xyxy format, where M is the number of objects
            scores: (M, ) confidence scores
            labels: (M, ) class labels
        '''
        coords, visibility, frame_feat, pred_track_instances = None, None, None, None

        if self.N > 0:
            coords, visibility, rho, frame_feat = self.online_forward(frame)

        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, device=self.device, dtype=torch.float32)
            scores = torch.as_tensor(scores, device=self.device, dtype=torch.float32)
            labels = torch.as_tensor(labels, device=self.device, dtype=torch.long)

            # Update object tracker state
            pred_track_instances = self.objtracker.track(self.t, bboxes, scores, labels, coords, visibility)

            # Get points to add/remove based on tracker results
            update_mask = self.objtracker.memo[5].flatten() # M*npt
            remove_mask = self.objtracker.points_remove_mask.flatten() # M'*npt
            query_points = rearrange(self.objtracker.memo[4], 'm n c -> (m n) c') # M*npt 2
            # Calculate frame features if not already done (i.e., first frame)
            if self.N == 0:
                f_t = self.backbone.encode_frames_online(frame) # 1 c h w
                frame_feat = rearrange(f_t, 'b (h w) c -> b c h w', h=self.h, w=self.w) # 1 c h w
                self.img_size = frame.shape[-2:]
                self.device = frame.device

            # Update LBM state (queries, memory)
            self.update(query_points, frame_feat, update_mask, remove_mask)

            self.query_points = query_points

        # Return predictions from *before* the update and the tracker instances
        return coords, visibility, pred_track_instances

        
class LBM_export(LBM):
    @staticmethod
    def init(query_points, first_frame_feat, img_size, memory_size=12):
        '''
        Initializes the model with query points and the first frame.

        Args:
            query_points: (N, 2) in xy format, where N is the number of points
            first_frame_feat: (1, C, H, W) video frame features
            img_size: (H, W) image size
            memory_size: memory size
        '''
        N = query_points.shape[0]
        query = None
        collision_dist = None
        stream_dist = None
        vis_mask = None
        mem_mask = None
        update_mask = np.ones(N, dtype=np.bool_)
        remove_mask = np.zeros(N, dtype=np.bool_)
        last_pos = query_points

        query, collision_dist, stream_dist, vis_mask, mem_mask = \
            LBM_export.update(query, query_points, first_frame_feat, update_mask, 
                remove_mask, collision_dist, stream_dist, vis_mask, mem_mask,
                img_size, N, memory_size)
        
        return query, collision_dist, stream_dist, vis_mask, mem_mask, last_pos

    @staticmethod
    def update(query, query_points, frame_feat, update_mask, remove_mask,
            collision_dist, stream_dist, vis_mask, mem_mask, img_size, N, 
            memory_size, stride=4):
        '''
        Updates query features and memory with new points.

        Args:
            query: (1, N, C) query features
            query_points: (N, 2) in xy format, where N is the number of points
            frame_feat: (1, C, H, W) video frame features
            update_mask: (N, ) boolean mask indicating which queries to update
            remove_mask: (N, ) boolean mask indicating which queries to remove
            img_size: (H, W) image size
            N: number of points
            memory_size: memory size
            stride: stride of features
        '''
        N_new = query_points.shape[0]
        _, c, h, w = frame_feat.shape
        input_size_h = h * stride
        input_size_w = w * stride

        if N_new > 0:
            remove_mask = remove_mask[:N]
            N_prev = (~remove_mask).sum().item()
            N_update = update_mask.sum().item()
            
            # Scale and normalize points for grid sampling
            update_points = query_points[update_mask]  # (N_update, 2)
            H, W = img_size
            update_points[:, 1] = (update_points[:, 1] / H) * input_size_h
            update_points[:, 0] = (update_points[:, 0] / W) * input_size_w
            
            # Create sampling grid
            x, y = update_points[:, 0], update_points[:, 1]
            x_grid = (x / input_size_w) * 2 - 1
            y_grid = (y / input_size_h) * 2 - 1
            grid = np.stack([x_grid, y_grid], axis=-1)
            grid = torch.tensor(grid).view(N_update, 1, 1, 2)

            # Sample features at query points
            frame_feat_tensor = torch.tensor(frame_feat)
            frame_feat_expanded = frame_feat_tensor.expand(N_update, -1, -1, -1)
            update_query = F.grid_sample(frame_feat_expanded, grid, mode='bilinear', 
                                        padding_mode='border', align_corners=False)
            update_query = update_query.reshape(1, N_update, c)

            # Initialize new tensors with increased capacity
            query_new = np.zeros((1, N_new, c))
            collision_dist_new = np.zeros((1, N_new, memory_size, c))
            stream_dist_new = np.zeros((1, N_new, memory_size, c))
            vis_mask_new = np.ones((1, N_new, memory_size), dtype=np.bool_)
            mem_mask_new = np.ones((1, N_new, memory_size), dtype=np.bool_)

            # Copy existing data
            if query is not None and N_prev > 0:
                try:
                    query_new[:, :N_prev] = query[:, ~remove_mask]
                    query_new[:, update_mask] = update_query
                    collision_dist_new[:, :N_prev] = collision_dist[:, ~remove_mask]
                    stream_dist_new[:, :N_prev] = stream_dist[:, ~remove_mask]
                    vis_mask_new[:, :N_prev] = vis_mask[:, ~remove_mask]
                    mem_mask_new[:, :N_prev] = mem_mask[:, ~remove_mask]
                except:
                    print("Error in copying existing data. Check tensor shapes.")
            else:
                query_new = update_query

        return query_new, collision_dist_new, stream_dist_new, vis_mask_new, mem_mask_new

    
    def forward(self, frame, queries, collision_dist, stream_dist, 
            vis_mask, mem_mask, last_pos):
        '''
        Forward pass for LBM_export.

        Args:
            frame: (1, C, H, W) video frame to extract features from
            queries: (1, N, C) query features
            collision_dist: (1, N, M, C) collision distance
            stream_dist: (1, N, M, C) stream distance
            vis_mask: (1, N, M) visibility mask
            mem_mask: (1, N, M) memory mask
            last_pos: (N, 2) last position
        '''
        H, W = frame.shape[-2], frame.shape[-1]
        device = frame.device
        N = queries.shape[1]

        # Encode the frame
        f_t = self.backbone.encode_frames_online(frame) # 1 c h w
        f_t = rearrange(f_t, 'b (h w) c -> b c h w', h=self.h, w=self.w) # 1 c h w
        queried_now_or_before = torch.ones([1, N], device=device, dtype=torch.bool)

        # forward
        out_t, memory, offset, vis, rho = self.transformer(
            query=queries.clone(), 
            feat=f_t, 
            stream_dist=stream_dist.clone(), 
            collision_dist=collision_dist.clone(), 
            vis_mask=vis_mask.clone(), 
            mem_mask=mem_mask.clone(), 
            queried_now_or_before=queried_now_or_before,
            last_pos=last_pos,
        )

        # update memory
        collision_dist_new = torch.cat([collision_dist[:, :, 1:], memory['collision'].unsqueeze(2)], dim=2) # b n m c
        stream_dist_new = torch.cat([stream_dist[:, :, 1:], memory['stream'].unsqueeze(2)], dim=2) # b n m c
        vis_mask_new = torch.cat([vis_mask[:, :, 1:], (F.sigmoid(vis) < self.visibility_threshold).unsqueeze(-1)], dim=2) # b n m
        mem_mask_new = torch.cat([mem_mask[:, :, 1:], ~queried_now_or_before.unsqueeze(-1)], dim=2) # b n m                          

        ref_tracks = out_t[-1]['reference_points'][0, :, 0] # n 2
        coord_pred = ref_tracks + offset[0] # n 2
        last_pos_new = coord_pred.clone()
        vis_pred = F.sigmoid(vis)[0] > self.visibility_threshold

        coord_pred[:, 1] = (coord_pred[:, 1] / self.size[0]) * H
        coord_pred[:, 0] = (coord_pred[:, 0] / self.size[1]) * W

        return f_t, coord_pred, vis_pred, collision_dist_new, stream_dist_new, vis_mask_new, mem_mask_new, last_pos_new


