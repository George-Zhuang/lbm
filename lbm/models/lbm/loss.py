import torch
import torch.nn as nn
import torch.nn.functional as F

from lbm.utils.coord_utils import coords_to_indices, indices_to_coords


class Loss_Function(nn.Module):
    def __init__(self, size, stride, lambda_cls, lambda_vis, lambda_off, lambda_unc,
        lambda_ref, lambda_depth=1.0, loss_after_query=True, tmp=0.05):
        super().__init__()

        self.gamma = 1.0

        self.size = size
        self.stride = stride
        self.lambda_cls = lambda_cls
        self.lambda_vis = lambda_vis
        self.lambda_off = lambda_off
        self.lambda_unc = lambda_unc
        self.lambda_ref = lambda_ref
        self.lambda_depth = lambda_depth
        self.loss_after_query = loss_after_query

        self.temperature = tmp

    def get_gt_offset(self, gt_tracks, stride, p_t):
        '''
        Args:
            gt_tracks: (B, T, N, 2)
            stride: int
            p_t: (B, T, N, P_prime)
        Returns:
            gt_offsets: (B, T, N, 2)
        '''
        point_pred = torch.argmax(p_t, dim=-1)                            # (B, T, N)
        point_pred = indices_to_coords(point_pred, self.size, stride)     # (B, T, N, 2)
        gt_offsets = gt_tracks - point_pred                               # (B, T, N, 2)

        return gt_offsets
    
    def get_masks(self, gt_visibility, query_times):
        '''
        Args:
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            mask_point: (B * T * N)
            mask_visible: (B * T * N)
        '''

        B, T, N = gt_visibility.shape
        device = gt_visibility.device

        # === Masks ===
        # visibility mask
        vis_mask = gt_visibility.float().view(-1)                                         # (B * T * N)

        # time mask
        if self.loss_after_query:
            time_indices = torch.arange(T).reshape(1, T, 1).to(device)                      # (1, T, 1)
            query_times_expanded = query_times.unsqueeze(1)                                 # (B, 1, N)
            time_mask = (time_indices > query_times_expanded).float()                      # (B, T, N)
            time_mask = time_mask.view(B * T * N)                                           # (B * T * N)
        else:
            time_mask = torch.ones(B * T * N, device=device).float()
        
        mask_point = vis_mask * time_mask                                              # (B * T * N)
        mask_visible = time_mask

        return mask_point, mask_visible


    def point_loss(self, p_t, gt_tracks, gt_visibility, query_times):
        '''
        Args:
            p_t: (B, T, N, P_prime)
            gt_tracks: (B, T, N, 2)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''

        B, T, N, _ = gt_tracks.shape

        mask_point, _ = self.get_masks(gt_visibility, query_times)

        gt_indices = coords_to_indices(gt_tracks, self.size, self.stride).view(-1)         # (B * T * N)

        p_t = p_t.reshape(B * T * N, -1)                                                           # (B * T * N, P)
        L_p = F.cross_entropy(p_t / self.temperature, gt_indices.long(), reduction="none")  # (B * T * N)
        L_p = L_p * mask_point                                                       # (B * T * N)
        L_p = L_p.sum() / mask_point.sum()

        L_p *= self.lambda_cls

        return L_p
    

    def visibility_loss(self, V_t, gt_visibility, query_times):
        '''
        Args:
            V_t: (B, T, N)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''

        B, T, N = gt_visibility.shape

        _, mask_visible = self.get_masks(gt_visibility, query_times)

        L_vis = F.binary_cross_entropy_with_logits(V_t, gt_visibility.float(), reduction="none")    # (B, T, N)
        L_vis = L_vis.view(B * T * N)                                                   # (B * T * N)
        L_vis = L_vis * mask_visible                                                   # (B * T * N)
        L_vis = L_vis.sum() / mask_visible.sum()

        L_vis *= self.lambda_vis

        return L_vis
    

    def offset_loss(self, O_t, ref_point, gt_tracks, gt_visibility, query_times):
        '''
        Args:
            O_t: (B, T, #offset_layers, N, 2)
            ref_point: (B, T, N, 2)
            gt_tracks: (B, T, N, 2)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''

        B, T, N, _ = gt_tracks.shape

        mask_point, _ = self.get_masks(gt_visibility, query_times)

        offset_layer_num = O_t.size(2)

        gt_offset = gt_tracks - ref_point

        cnt_offset = 0
        L_offset = 0
        for l in range(offset_layer_num):

            o_l = O_t[:, :, l]                                          # (B, T, N, 2)
            offset_loss = F.l1_loss(o_l, gt_offset, reduction="none")   # (B, T, N, 2)
            offset_loss = offset_loss.sum(dim=-1).view(-1)                          # (B * T * N)
            offset_loss = offset_loss * mask_point                                  # (B * T * N)

            offset_loss = torch.clamp(offset_loss, min=0, max=2 * self.stride)

            offset_loss = offset_loss.sum() / mask_point.sum()
            offset_loss *= (self.gamma ** (offset_layer_num - l - 1))
            cnt_offset += 1

            L_offset += offset_loss

        L_offset *= self.lambda_off
        L_offset /= cnt_offset

        return L_offset

    def depth_loss(self, pred_depths, gt_tracks_depth, gt_visibility, query_times):
        '''
        Args:
            pred_depths: (B, T, N)
            gt_tracks_depth: (B, T, N)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''
        # mask
        mask_point, _ = self.get_masks(gt_visibility, query_times)
        
        # L1 loss
        depth_loss = F.l1_loss(pred_depths, gt_tracks_depth, reduction="none")  # (B, T, N)
        depth_loss = depth_loss.view(-1)  # (B * T * N)
        depth_loss = depth_loss * mask_point  # (B * T * N)
        depth_loss = depth_loss.sum() / mask_point.sum()
        
        depth_loss *= self.lambda_depth
        
        return depth_loss

    def uncertainty_loss(self, U_t, point_pred, gt_tracks, gt_visibility, query_times):
        '''
        Args:
            U_t: (B, T, N)
            point_pred: (B, T, N, 2)
            gt_tracks: (B, T, N, 2)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''

        B, T, N, _ = gt_tracks.shape

        _, mask_visible = self.get_masks(gt_visibility, query_times)                 # (B * T * N)

        # L2 difference between point_pred and gt_tracks
        uncertainty_loss = F.mse_loss(point_pred, gt_tracks, reduction="none")        # (B, T, N, 2)
        uncertainty_loss = uncertainty_loss.sum(dim=-1).sqrt()                        # (B, T, N)

        # (uncertainty_loss > 8.0) or (~gt_visibility)
        uncertains = (uncertainty_loss > 8.0) | (~gt_visibility)      # (B, T, N)

        L_unc = F.binary_cross_entropy_with_logits(U_t, uncertains.float(), reduction="none")        # (B, T, N)
        L_unc = L_unc.view(B * T * N)                                                   # (B * T * N)
        L_unc = L_unc * mask_visible                                                   # (B * T * N)
        L_unc = L_unc.sum() / mask_visible.sum()

        return L_unc