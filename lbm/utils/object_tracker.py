import torch
from torch import Tensor

from lbm.utils.tracker_utils import (
    BaseTracker, 
    bbox_overlaps, 
    check_points_in_box, 
    sample_points_in_box
)


class LBMObjTracker(BaseTracker):
    """LBM Object Tracker.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.1.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.1.
        match_score_thr (list): The match threshold. Defaults to [0.5, 0.5].
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_point_frames (int): The most frames to keep points. Defaults to 5.
        distractor_score_thr (float): The score threshold to consider an object as a distractor.
            Defaults to 0.5.
        distractor_nms_thr (float): The NMS threshold for filtering out distractors.
            Defaults to 0.3.
        max_distance (float): Maximum distance for considering matches. Defaults to -1.
        fps (int): Frames per second of the input video. Used for calculating growth factor. Defaults to 1.
        object_points (int): Number of points to sample in each box. Defaults to 9.
        rm_distractor (bool): Whether to remove distractors. Defaults to True.
        weight_match_with_det_scores (bool): Whether to weight matches with detection scores. Defaults to True.
        weight_match_with_det_labels (bool): Whether to weight matches with detection labels. Defaults to True.
    """

    def __init__(self, args):
        super().__init__()
        
        # Extract parameters from args
        self.object_points = args.object_points
        self.init_score_thr = args.init_score_thr
        self.obj_score_thr = args.obj_score_thr
        self.match_score_thr = args.match_score_thr
        self.memo_tracklet_frames = args.memo_tracklet_frames
        self.memo_point_frames = args.memo_point_frames
        self.distractor_score_thr = args.distractor_score_thr
        self.distractor_nms_thr = args.distractor_nms_thr
        self.rm_distractor = args.rm_distractor
        self.max_distance = args.max_distance

        # Frame rate related parameters
        self.fps = args.fps
        self.growth_factor = self.fps / 6
        self.distance_smoothing_factor = 100 / self.fps

        # Match weighting parameters
        self.weight_match_with_det_scores = args.weight_match_with_det_scores
        self.weight_match_with_det_labels = args.weight_match_with_det_labels
        
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.points_remove_mask = None

    @property
    def memo(self):
        """Memory of tracker."""
        memo_bboxes = []
        memo_labels = []
        memo_ids = []
        memo_frame_ids = []
        memo_points = []
        memo_points_update_mask = []
        memo_points_last_frame = []

        if not self.tracks:
            memo_bboxes = torch.empty((0, 4), dtype=torch.float32)
            memo_labels = torch.empty((0, 1), dtype=torch.long)
            memo_ids = torch.empty((0,), dtype=torch.long)
            memo_frame_ids = torch.empty((0,), dtype=torch.long)
            object_points = self.object_points if hasattr(self, 'object_points') else 0 # Default if not initialized
            memo_points = torch.empty((0, object_points, 2), dtype=torch.float32)
            memo_points_update_mask = torch.empty((0, object_points), dtype=torch.bool)
            memo_points_last_frame = torch.empty((0, object_points), dtype=torch.long)
        else:
            # If tracks exist, collect data and concatenate.
            memo_bboxes_list = []
            memo_labels_list = []
            memo_ids_list = []
            memo_frame_ids_list = []
            memo_points_list = []
            memo_points_update_mask_list = []
            memo_points_last_frame_list = []

            for k, v in self.tracks.items():
                memo_bboxes_list.append(v['bbox'][None, :])
                memo_labels_list.append(v['label'].view(1, 1)) # Keep shape (1, 1) for cat
                memo_ids_list.append(k)
                memo_frame_ids_list.append(v['last_frame'])
                memo_points_list.append(v['points'][None, :])
                memo_points_update_mask_list.append(v['points_update_mask'][None, :])
                memo_points_last_frame_list.append(v['points_last_frame'][None, :])

            # Concatenate collected data into tensors on the inferred device.
            memo_bboxes = torch.cat(memo_bboxes_list, dim=0)  # M, 4
            memo_labels = torch.cat(memo_labels_list, dim=0)  # M, 1
            memo_ids = torch.tensor(memo_ids_list, dtype=torch.long) # M,
            memo_frame_ids = torch.tensor(memo_frame_ids_list, dtype=torch.long)# M,
            memo_points = torch.cat(memo_points_list, dim=0)  # M, npt, 2
            memo_points_update_mask = torch.cat(memo_points_update_mask_list, dim=0)  # M, npt
            memo_points_last_frame = torch.cat(memo_points_last_frame_list, dim=0)  # M, npt

        return (
            memo_bboxes, 
            memo_labels, 
            memo_ids, 
            memo_frame_ids,
            memo_points, 
            memo_points_update_mask,
            memo_points_last_frame,
        )

    def compute_distance_mask(self, bboxes1, bboxes2, frame_ids1, frame_ids2):
        """Compute a mask based on the pairwise center distances and frame IDs with piecewise soft-weighting."""
        centers1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0
        centers2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0
        distances = torch.cdist(centers1, centers2)

        frame_id_diff = torch.abs(frame_ids1[:, None] - frame_ids2[None, :]).to(
            distances.device
        )

        # Define a scaling factor for the distance based on frame difference (exponential growth)
        scaling_factor = torch.exp(frame_id_diff.float() / self.growth_factor)

        # Apply the scaling factor to max_distance
        adaptive_max_distance = self.max_distance * scaling_factor

        # Create a piecewise function for soft gating
        soft_distance_mask = torch.where(
            distances <= adaptive_max_distance,
            torch.ones_like(distances),
            torch.exp(-(distances - adaptive_max_distance) / self.distance_smoothing_factor),
        )

        return soft_distance_mask

    def update(
        self,
        ids: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        scores: Tensor,
        frame_id: int,
    ) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1
        device = bboxes.device

        for id, bbox, label, score in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
        ):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                self.tracks[id]["bbox"] = bbox
                self.tracks[id]["last_frame"] = frame_id
                self.tracks[id]["label"] = label
                self.tracks[id]["score"] = score
                # points still in box? 
                inside_mask = check_points_in_box(bbox.unsqueeze(0), self.tracks[id]["points"]).squeeze(0) # npt,
                self.tracks[id]["points_last_frame"][inside_mask] = frame_id
                points_update_mask = torch.zeros(self.object_points, dtype=torch.bool, device=bbox.device)
                self.tracks[id]["points_update_mask"] = points_update_mask
                self.tracks[id]["tentative"] = False
            else:
                # initialize a new track
                points = sample_points_in_box(bbox, self.object_points)
                points_update_mask = torch.ones(self.object_points, dtype=torch.bool, device=bbox.device)
                points_last_frame = torch.full((self.object_points,), frame_id, device=bbox.device)
                self.tracks[id] = dict(
                    bbox=bbox,
                    points=points,
                    points_update_mask=points_update_mask,
                    points_last_frame=points_last_frame,
                    label=label,
                    score=score,
                    last_frame=frame_id,
                    tentative=True,
                )
        

        # pop memo
        invalid_ids = []
        points_remove_mask = []
        if not self.tracks:
            self.points_remove_mask = torch.ones(0, dtype=torch.bool, device=device)
        else:
            for k, v in self.tracks.items():
                lost = frame_id - v["last_frame"] >= self.memo_tracklet_frames
                tentative = v["tentative"] and v['last_frame'] < frame_id
                if lost or tentative:
                    invalid_ids.append(k)
                    points_remove_mask.append(torch.ones(self.object_points, dtype=torch.bool, device=device))
                else:
                    points_remove_mask.append(torch.zeros(self.object_points, dtype=torch.bool, device=device))
            for invalid_id in invalid_ids:
                self.tracks.pop(invalid_id)
            self.points_remove_mask = torch.stack(points_remove_mask, dim=0) # M, npt

        keep_mask = ~self.points_remove_mask
        query_points = self.memo[4]
        if keep_mask.sum() != query_points.shape[0]*query_points.shape[1]:
            print('Error: points remove mask not match')

        # pop memo points and add new points
        for k, v in self.tracks.items():
            # Identify points that are outside the bbox and should be removed
            out_points_mask = (frame_id - v["points_last_frame"]) >= self.memo_point_frames
            num_out_points = out_points_mask.sum().item()
            if num_out_points > 0:
                # Sample new points for positions that are too old
                new_points = sample_points_in_box(v["bbox"], num_out_points)
                # Replace old points with new ones, maintaining the original order
                v["points"][out_points_mask] = new_points
                # Update the last frame for the new points to the current frame
                v["points_last_frame"][out_points_mask] = frame_id
                v["points_update_mask"][out_points_mask] = True
        

    def track(
        self, 
        frame_id: int, 
        bboxes: Tensor, 
        scores: Tensor,
        labels: Tensor,
        point_coords: Tensor,
        point_vis: Tensor,
    ):
        """Tracking forward function.

        Args:
            frame_id (int): The id of current frame.
            bboxes (Tensor): Bounding boxes in xyxy format.
            scores (Tensor): Detection scores.
            labels (Tensor): Class labels.
            point_coords (Tensor): Point coordinates.
            point_vis (Tensor): Point visibility.

        Returns:
            dict: Tracking results with bboxes, labels, scores, inside_mask, and instances_id.
        """
        device = bboxes.device
        pred_track_instances = dict()

        if frame_id == 0:
            self.reset()

        if self.empty or bboxes.shape[0] == 0:
            valid_inds = scores > self.init_score_thr
            scores = scores[valid_inds]
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.shape[0]
            ids = torch.arange(self.num_tracks, self.num_tracks + num_new_tracks).to(device=device)
            self.num_tracks += num_new_tracks

        else:
            # init
            ids = torch.full((bboxes.shape[0], ), -1, dtype=labels.dtype, device=labels.device)

            # get the detection bboxes
            inds = scores > self.obj_score_thr
            bboxes = bboxes[inds]
            labels = labels[inds]
            scores = scores[inds]
            ids = ids[inds]
            
            memo_bboxes, memo_labels, memo_ids, memo_frame_ids, _, _, _ = self.memo
            
            # remove the distractor
            if self.rm_distractor:
                bboxes, labels, scores, ids, embeds, mask_inds = self.remove_distractor(
                    bboxes,
                    labels,
                    scores,
                    ids,
                    nms="inter",
                    distractor_score_thr=self.distractor_score_thr,
                    distractor_nms_thr=self.distractor_nms_thr,
                )
                
            # point in box
            inside_mask = check_points_in_box(bboxes, point_coords)  # N, M*npt
            point_counts = inside_mask.view(bboxes.shape[0], memo_bboxes.shape[0], self.object_points).sum(dim=-1)  # N, M
            
            # penalize large boxes
            bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])  # N,
            memo_areas = (memo_bboxes[:, 2] - memo_bboxes[:, 0]) * (memo_bboxes[:, 3] - memo_bboxes[:, 1])  # M,
            weight = memo_areas.unsqueeze(0) / bbox_areas.unsqueeze(1)  # N, M
            weight = torch.clamp(weight, min=0.1, max=1.0)
            
            # match score
            match_scores = point_counts * weight
            
            # reweight by detection scores
            if self.weight_match_with_det_scores:
                match_scores *= scores.unsqueeze(1)
                
            if self.weight_match_with_det_labels:
                label_match = labels.unsqueeze(1) == memo_labels.T
                label_weight = torch.where(
                    label_match,
                    torch.tensor(1.0, device=labels.device),
                    torch.tensor(0.5, device=labels.device)
                )
                match_scores *= label_weight

            # softmax
            d2t_scores = match_scores.softmax(dim=0)
            t2d_scores = match_scores.softmax(dim=1)
            match_scores = (d2t_scores + t2d_scores) / 2.0

            # reweight by distance
            if self.max_distance != -1:
                current_frame_ids = torch.full((bboxes.shape[0],), frame_id, dtype=torch.long)
                distance_mask = self.compute_distance_mask(bboxes, memo_bboxes, current_frame_ids, memo_frame_ids)
                match_scores = match_scores * distance_mask
            
            # match
            for i in range(bboxes.shape[0]):
                conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        # keep bboxes with high object score and remove background bboxes
                        if scores[i] > self.obj_score_thr:
                            ids[i] = id
                            match_scores[:i, memo_ind] = 0
                            match_scores[i + 1:, memo_ind] = 0
                            
            # initialize new tracks
            new_inds = (ids == -1) & (scores > self.init_score_thr)
            num_news = new_inds.sum()
            ids[new_inds] = torch.arange(
                self.num_tracks, self.num_tracks + num_news, dtype=torch.long, device=device
            )
            self.num_tracks += num_news
            
        # update the matched tracks
        self.update(ids, bboxes, labels, scores, frame_id)
            
        # update pred_track_instances
        tracklet_inds = ids > -1
        pred_track_instances = dict(
            bboxes=bboxes[tracklet_inds],
            labels=labels[tracklet_inds],
            scores=scores[tracklet_inds],
            instances_id=ids[tracklet_inds],
        )

        return pred_track_instances

    def remove_distractor(
        self,
        bboxes,
        labels,
        scores,
        ids,
        track_feats=None,
        mask_inds=[],
        distractor_score_thr=0.5,
        distractor_nms_thr=0.3,
        nms="inter",
    ):
        """Remove distractor objects based on scores and overlaps."""
        # all objects is valid here
        valid_inds = labels > -1
        
        # nms
        low_inds = torch.nonzero(scores < distractor_score_thr, as_tuple=False).squeeze(1)
        
        if nms == "inter":
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes[:, :])
        elif nms == "intra":
            cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
            ious = bbox_overlaps(bboxes[low_inds, :], bboxes)
            ious *= cat_same.to(ious.device)
        else:
            raise NotImplementedError

        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        ids = ids[valid_inds]
        
        if track_feats is not None:
            track_feats = track_feats[valid_inds]

        if len(mask_inds) > 0:
            mask_inds = mask_inds[valid_inds]

        return bboxes, labels, scores, ids, track_feats, mask_inds
