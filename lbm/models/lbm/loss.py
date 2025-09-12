import torch
import torch.nn as nn
import torch.nn.functional as F
from lbm.utils.coord_utils import coords_to_indices


class Loss_Function(nn.Module):
    def __init__(self, size, stride, lambda_cls, lambda_vis, lambda_reg, lambda_unc, lambda_ref, loss_after_query, tmp=0.05):
        super().__init__()
        self.size = size
        self.stride = stride
        self.lambda_cls = lambda_cls
        self.lambda_vis = lambda_vis
        self.lambda_reg = lambda_reg
        self.lambda_unc = lambda_unc
        self.lambda_ref = lambda_ref
        self.loss_after_query = loss_after_query

        self.temperature = tmp
    
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
        # visibility mask
        vis_mask = gt_visibility.float().view(-1) # b t n
        # time mask
        if self.loss_after_query:
            time_indices = torch.arange(T).reshape(1, T, 1).to(device) # 1 t 1
            query_times_expanded = query_times.unsqueeze(1) # b 1 n
            time_mask = (time_indices > query_times_expanded).float() # b t n
            time_mask = time_mask.view(B * T * N) # b t n
        else:
            time_mask = torch.ones(B * T * N, device=device).float() # b t n
        mask_point = vis_mask * time_mask # b t n
        mask_visible = time_mask
        return mask_point, mask_visible

    def classification_loss(self, corr_map, gt_tracks, gt_visibility, query_times):
        '''
        Args:
            corr_map: (B, T, N, h*w)
            gt_tracks: (B, T, N, 2)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''
        B, T, N, _ = gt_tracks.shape
        mask_point, _ = self.get_masks(gt_visibility, query_times)
        gt_indices = coords_to_indices(gt_tracks, self.size, self.stride).view(-1) # b t n
        corr_map = corr_map.reshape(B * T * N, -1) # b t n h*w
        L_cls = F.cross_entropy(corr_map / self.temperature, gt_indices.long(), reduction="none") # b t n
        L_cls = L_cls * mask_point # b t n
        L_cls = L_cls.sum() / mask_point.sum()
        L_cls *= self.lambda_cls
        return L_cls
    

    def visibility_loss(self, visibility, gt_visibility, query_times):
        '''
        Args:
            visibility: (B, T, N)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''
        B, T, N = gt_visibility.shape
        _, mask_visible = self.get_masks(gt_visibility, query_times)
        L_vis = F.binary_cross_entropy_with_logits(visibility, gt_visibility.float(), reduction="none") # b t n
        L_vis = L_vis.view(B * T * N) # b t n
        L_vis = L_vis * mask_visible # b t n
        L_vis = L_vis.sum() / mask_visible.sum()
        L_vis *= self.lambda_vis
        return L_vis
    

    def regression_loss(self, offset, ref_point, gt_tracks, gt_visibility, query_times):
        '''
        Args:
            offset: (B, T, N, 2)
            ref_point: (B, T, N, 2)
            gt_tracks: (B, T, N, 2)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''
        mask_point, _ = self.get_masks(gt_visibility, query_times)
        gt_offset = gt_tracks - ref_point
        L_reg = F.l1_loss(offset, gt_offset, reduction="none") # b t n 2
        L_reg = L_reg.sum(dim=-1).view(-1) # b t n
        L_reg = L_reg * mask_point # b t n
        L_reg = torch.clamp(L_reg, min=0, max=2 * self.stride)
        L_reg = L_reg.sum() / mask_point.sum()
        L_reg *= self.lambda_reg
        return L_reg

    
    def uncertainty_loss(self, uncertainty, tracks, gt_tracks, gt_visibility, query_times):
        '''
        Args:
            uncertainty: (B, T, N)
            tracks: (B, T, N, 2)
            gt_tracks: (B, T, N, 2)
            gt_visibility: (B, T, N)
            query_times: (B, N)
        Returns:
            loss: float
        '''
        B, T, N, _ = gt_tracks.shape
        _, mask_visible = self.get_masks(gt_visibility, query_times) # b t n
        # L2 difference between tracks and gt_tracks
        uncertainty_loss = F.mse_loss(tracks, gt_tracks, reduction="none") # b t n 2
        uncertainty_loss = uncertainty_loss.sum(dim=-1).sqrt() # b t n
        # (uncertainty_loss > 8.0) or (~gt_visibility)
        uncertains = (uncertainty_loss > 8.0) | (~gt_visibility) # b t n
        L_unc = F.binary_cross_entropy_with_logits(uncertainty, uncertains.float(), reduction="none") # b t n
        L_unc = L_unc.view(B * T * N) # b t n
        L_unc = L_unc * mask_visible # b t n
        L_unc = L_unc.sum() / mask_visible.sum()
        L_unc *= self.lambda_unc
        return L_unc