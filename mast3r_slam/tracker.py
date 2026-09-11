import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device

        self.reset_idx_f2k()

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf

        # Use pose to transform points to update keyframe
        # lietorch's CUDA kernels crash on this GPU (see frame.py's
        # SharedStates note) -- do act() on CPU, restore Xkf's device after.
        _orig_device = Xkf.device
        Xkk = T_CkCf.cpu().act(Xkf.cpu()).to(_orig_device)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3.cpu())

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3.cpu())

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
