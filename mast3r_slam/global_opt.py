import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size = self.cfg["window_size"]

        self.K = K

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        self._prune_old_edges()

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def _prune_old_edges(self):
        # Sliding-window memory bound: the per-edge tensors (idx_ii2jj,
        # valid_match_*, Q_*) each scale with num_edges * num_points
        # (147456 for MASt3R's default patch grid), so an ever-growing
        # factor graph on a long video eventually hits CUDA OOM (observed
        # in practice around ~900 edges / 200+ keyframes on a 32GB GPU).
        # Edges where BOTH endpoints are older than window_size frames
        # behind the newest keyframe have already served their purpose
        # (both sides were already mutually localized when the edge was
        # formed) -- drop them permanently. Edges with only one old
        # endpoint (a loop closure from a recent keyframe back to an old
        # one) are kept, so loop closing still works within the window.
        if self.ii.numel() == 0:
            return
        newest = torch.max(self.ii.max(), self.jj.max())
        cutoff = newest - self.window_size
        if cutoff <= 0:
            return
        keep = (self.ii >= cutoff) | (self.jj >= cutoff)
        if keep.all():
            return
        self.ii = self.ii[keep]
        self.jj = self.jj[keep]
        self.idx_ii2jj = self.idx_ii2jj[keep]
        self.idx_jj2ii = self.idx_jj2ii[keep]
        self.valid_match_j = self.valid_match_j[keep]
        self.valid_match_i = self.valid_match_i[keep]
        self.Q_ii2jj = self.Q_ii2jj[keep]
        self.Q_jj2ii = self.Q_jj2ii[keep]

    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        # T_WCs.data lives on CPU (forced there as the fix for a separate,
        # unrelated lietorch .act()/.retr() CUDA crash -- see frame.py's
        # SharedStates note). gauss_newton_rays is a different custom CUDA
        # kernel with no such lietorch dependency; it needs its pose input
        # on the same device as Xs. Passing a CPU tensor into this kernel's
        # packed_accessor32 (which treats it as a device pointer) caused a
        # genuine illegal-memory-access crash -- root-caused by dumping the
        # real inputs and replaying them in isolation (repro_gauss_newton_
        # crash_real.py) after the CPU-tensor mismatch was noticed directly
        # in the dump.
        pose_data = T_WCs.data[:, 0, :].to(Xs.device)
        if not (torch.isfinite(pose_data).all() and torch.isfinite(Xs).all()):
            raise RuntimeError(
                f"solve_GN_rays: non-finite input before gauss_newton_rays -- "
                f"pose_data finite={torch.isfinite(pose_data).all().item()} "
                f"Xs finite={torch.isfinite(Xs).all().item()} "
                f"pose_data.device={pose_data.device} Xs.device={Xs.device}"
            )
        num_points = Xs.shape[1]
        idx_max = idx_ii2jj.max().item() if idx_ii2jj.numel() else -1
        idx_min = idx_ii2jj.min().item() if idx_ii2jj.numel() else -1
        print(
            f"[solve_GN_rays debug] num_poses={pose_data.shape[0]} "
            f"num_points(Xs.shape[1])={num_points} num_edges={ii.shape[0]} "
            f"idx_ii2jj.shape={tuple(idx_ii2jj.shape)} "
            f"idx_ii2jj range=[{idx_min},{idx_max}] "
            f"valid_match.shape={tuple(valid_match.shape)} "
            f"ii.max={ii.max().item() if ii.numel() else -1} "
            f"jj.max={jj.max().item() if jj.numel() else -1} "
            f"unique_kf_idx.numel={n_unique_kf}",
            flush=True,
        )
        if idx_max >= num_points or idx_min < 0:
            raise RuntimeError(
                f"solve_GN_rays: idx_ii2jj out of range for Xs point dim -- "
                f"idx range=[{idx_min},{idx_max}] but num_points={num_points}"
            )
        import os
        import time as _time
        _gn_start = _time.time()
        if os.environ.get("DUMP_GN_INPUTS"):
            torch.save(
                {
                    "pose_data": pose_data,
                    "Xs": Xs,
                    "Cs": Cs,
                    "ii": ii,
                    "jj": jj,
                    "idx_ii2jj": idx_ii2jj,
                    "valid_match": valid_match,
                    "Q_ii2jj": Q_ii2jj,
                    "sigma_ray": sigma_ray,
                    "sigma_dist": sigma_dist,
                    "C_thresh": C_thresh,
                    "Q_thresh": Q_thresh,
                    "max_iter": max_iter,
                    "delta_thresh": delta_thresh,
                },
                "gn_inputs_dump.pt",
            )
            print("Dumped real gauss_newton_rays inputs to gn_inputs_dump.pt", flush=True)
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )
        print(f"[solve_GN_rays debug] solve took {_time.time() - _gn_start:.2f}s", flush=True)

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        # See solve_GN_rays's identical fix: T_WCs.data lives on CPU (forced
        # there for the separate lietorch crash fix), but this CUDA kernel
        # needs it on the same device as Xs.
        pose_data = T_WCs.data[:, 0, :].to(Xs.device)

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
