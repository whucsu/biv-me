import numpy as np
from typing import Dict, Tuple

from .surface_enum import Surface
from bivme.fitting import BiventricularModel

EPSILON = 1e-12


# noinspection PyPep8Naming
class ExplicitFitterNumpy:
    """
    Explicit diffeomorphic fitter using vectorized numpy.
    Mimics OSQP-style fitting with explicit linearized constraints
    @author: groberts
    @date: 2025-08-15
    """
    def __init__(self,
                 logger,
                 biv_model,
                 weight_gp,
                 data_set,
                 collision_detection: bool = False,
                 model_prior: BiventricularModel = None):
        
        self.biv_model = biv_model
        self.weight_gp = weight_gp
        self.data_set = data_set
        self.logger = logger
        self.dtype = np.float64  # for openblas compatibility
        self.collision_detection = collision_detection
        self.model_prior = model_prior

        # Jacobian (needed for checking if diffeomorphic) and mbd's for getting G and h constraints
        self.jac_11 = np.asarray(self.biv_model.jac_11, dtype=self.dtype)
        self.jac_12 = np.asarray(self.biv_model.jac_12, dtype=self.dtype)
        self.jac_13 = np.asarray(self.biv_model.jac_13, dtype=self.dtype)
        self.mbdx = np.asarray(self.biv_model.mbder_dx, dtype=self.dtype)
        self.mbdy = np.asarray(self.biv_model.mbder_dy, dtype=self.dtype)
        self.mbdz = np.asarray(self.biv_model.mbder_dz, dtype=self.dtype)

    def explicit_refine_osqp_style(self,
                                   max_outer: int = 5,
                                   min_jacobian: float = 0.1,
                                   transmural_weight: float = 0.01,
                                   smoothing_weight: float = 1e6,
                                   tau: float = 1 / 3,  # bound in |C d| <= tau (OSQP used 3.0 scaling → tau=1/3)
                                   resid_conv_tol: float = 5e-4,
                                   disp_conv_tol: float = 1e-4,
                                   relinearize: bool = False,  # False to mimic original OSQP
                                   ) -> Tuple[np.array, Dict[str, float]]:

        # Pull initial correspondences (fixed Φ, like your OSQP) and data
        if self.collision_detection:
            data_points = self.model_prior.et_pos
            phi = self.biv_model.basis_matrix
            weights = self.weight_gp * np.ones((self.biv_model.et_pos.shape[0], 1))

        else:
            idx, weights, _, phi = self.biv_model.compute_data_xi_fast(self.weight_gp, self.data_set)
            weights = weights[:, np.newaxis]
            data_points = self.data_set.points_coordinates[idx]

        # Spatial regularizer S = gtstsg_x + gtstsg_y + transmural_weight * gtstsg_z
        smooth = (self.biv_model.gtstsg_x + self.biv_model.gtstsg_y + (transmural_weight * self.biv_model.gtstsg_z))
        mesh = self.biv_model.control_mesh

        # Move to tensors
        P = np.asarray(phi, dtype=self.dtype)
        w = np.asarray(weights, dtype=self.dtype)
        y = np.asarray(data_points, dtype=self.dtype)
        S = np.asarray(smooth, dtype=self.dtype) * smoothing_weight
        M = np.asarray(mesh, dtype=self.dtype)  # (Nc,3)
        d_prev = np.zeros_like(M, dtype=self.dtype)  # accumulated displacement (Nc,3)

        k = 0
        last_res = self.rmse(P, M, y)

        collision_iteration = 1

        # Outer loop: rebuild G/h constraints with updated mesh, do QP step, check early-stops and diffeo-check
        for k in range(1, max_outer + 1):

            if self.collision_detection and collision_iteration > 3:
                self.logger.info(f" STOP -> Max collision iterations reached")
                break

            # 1) Rebuild constraint matrix G,h around CURRENT geometry (uses updated mesh)
            G, h = self.build_G_h_from_mbder(M, tau=tau)

            # 2) One QP step with constraint backtracking (fixed phi, like your OSQP)
            d_new, step_log = self.qp_step_with_constraints(P, w, S, y, M, d_prev, G=G, h=h)

            # 3) Propose new mesh and check diffeomorphism
            M_try = M + d_new
            if not self.biv_model.is_diffeomorphic(M_try, min_jacobian):
                # Backtrack globally (if needed until diffeo passes))
                M_bt, a = self._diffeo_backtrack(M, d_new, min_jacobian)
                if M_bt is None:
                    self.logger.info(f" Iteration {k}: STOP -> Diffeo failed even after backtracking")
                    break
                else:
                    self.logger.info(f" Iteration {k}: Diffeo failed. Backtrack displacement by scale {a}...")

                # Scale the step we actually took
                d_new = d_new * a
                M_try = M_bt

            # Check if displacement converged
            if np.linalg.norm(d_new) < disp_conv_tol:
                self.logger.info(f" Iteration {k}: STOP -> Displacement converged")
                break
            
            update_mesh = True

            if self.collision_detection:
                # Collision detection check
                current_collision = M_try.detect_collision()
                inter = current_collision.difference(M_try.reference_collision) 
                if bool(inter):
                    self.logger.info(f" Iteration {k}: Collision detected, updating prior model and refitting...")
                    for surface in [Surface.RV_SEPTUM, Surface.RV_FREEWALL, Surface.RV_INSERT]:
                        surface_index = self.biv_model.get_surface_vertex_start_end_index(surface)
                        self.model_prior.et_pos[surface_index[0] : surface_index[1] + 1, :] = self.biv_model.et_pos[surface_index[0] : surface_index[1] + 1, :]

                    y = self.model_prior.et_pos

                    collision_iteration += 1
                    update_mesh = False

            if update_mesh:
                # 4) RMSE before accepting
                res = self.rmse(P, M_try, y)
                a = step_log["alpha"]

                if res > last_res:
                    self.logger.info(f" Iteration {k}: STOP -> RMSE diverged (last {last_res:.6f} vs new {res:.6f})")
                    break

                # 5) Accept step
                M = M_try
                d_prev = d_new
                self.biv_model.update_control_mesh(M)

                # 6) Optional: re-linearize correspondences (recommended for extra drop)
                if relinearize and not self.collision_detection:
                    idx, weights, _, P = self.biv_model.compute_data_xi_fast(self.weight_gp, self.data_set)
                    w = weights[:, np.newaxis]
                    y = self.data_set.points_coordinates[idx]

                # 7) Early stop if little improvement
                rel = abs(last_res - res) / max(last_res, EPSILON)
                if rel < resid_conv_tol:  # tune as you like (your OSQP tol was 5e-4 on residual ratio)
                    self.logger.info(f" Iteration {k}: STOP -> Residuals converged")
                    break

                last_res = res
                self.logger.info(f"     Iteration {k}       ECF error {last_res:.6f}")

        # Return results
        final_stats = {"weighted_rmse": float(last_res), "iters": k}
        return M, final_stats

    def _diffeo_backtrack(self, M, d_new, min_jacobian, max_backtracks=3, backtrack_scale=0.5):
        """Try reduced alpha until diffeo passes. Returns (M_accepted, alpha) or (None, 0.0)."""
        a = 1.0
        for _ in range(max_backtracks):
            M_try = M + a * d_new
            if self.biv_model.is_diffeomorphic(M_try, min_jacobian):
                return M_try, a
            a *= backtrack_scale

        return None, 0.0

    def qp_step_with_constraints(self,
                                 P: np.ndarray,  # (N, Nc)
                                 w: np.ndarray,  # (N,)  OSQP uses W (not sqrtW) → weights get squared
                                 S: np.ndarray,  # (Nc, Nc) SPD, already multiplied by smoothing weight
                                 y: np.ndarray,  # (N, 3)
                                 M: np.ndarray,  # (Nc, 3) current mesh (control points)
                                 d_prev: np.ndarray,  # (Nc, 3) warm-start / accumulated displacement
                                 G: np.ndarray | None = None,  # (m, Nc)
                                 h: np.ndarray | None = None,  # (m,)
                                 safety=0.99,  # shrink alpha_max a bit to avoid boundary
                                 ) -> Tuple[np.array, Dict[str, np.array]]:
        """
        Tried to match OSQP as closely as possible with some variations.

        Solve:  min_d  1/2 d^T (2A) d + (2A d_prev - 2b)^T d   s.t.  G d <= h
        where A = P^T W^2 P + reg_weight * S    and  b = P^T W^2 (y - P M)
        (done for each coordinate independently, sharing A)

        Returns
        -------
        d_new : (Nc, 3)   next displacement iterate that satisfies G d <= h (if G,h provided)
        info  : dict      diagnostics: alpha per coord, was_projected flags
        """

        # Build A and b (W^2 weighting, same as OSQP)
        w2 = w * w  # (N,)
        W2P = P * w2  # (N, Nc)
        A = (P.T @ W2P) + S  # (Nc, Nc) SPD

        r = y - (P @ M)  # (N, 3)
        b = W2P.T @ r  # (Nc, 3)

        # argmin_d  1/2 d^T (2A) d + (2A d_prev - 2b)^T d  →  A(d + d_prev) = b, where d_star = A^{-1} b - d_prev
        L = np.linalg.cholesky(A)
        rhs = b - (A @ d_prev)  # (Nc, 3)
        ytmp = np.linalg.solve(L, rhs)  # forward solve
        d_star = np.linalg.solve(L.T, ytmp)  # back solve → d_star (Nc,3)

        # No constraints → done
        if G is None or h is None:
            return d_star, {"alpha": np.ones(3, dtype=self.dtype),
                            "projected": np.zeros(3, dtype=bool)}

        # Backtracking along the line from d_prev to d_star, per coord
        d_new = d_prev.copy()
        alphas = np.ones(3, dtype=self.dtype)
        projected = np.zeros(3, dtype=bool)

        # Loop through the dimensions
        for c in range(3):
            dir_c = d_star[:, c] - d_prev[:, c]  # (Nc,)
            if np.max(np.abs(dir_c)) <= EPSILON:
                alphas[c] = 0.0
                continue

            g_prev = G @ d_prev[:, c]  # (m,)
            g_dir = G @ dir_c  # (m,)

            # Rows where constraint tightens as alpha increases
            feasible0 = np.all(g_prev <= (h + EPSILON))
            mask = g_dir > EPSILON
            if np.any(mask):
                alpha_max = np.min((h[mask] - g_prev[mask]) / g_dir[mask])
                alpha = np.clip(alpha_max * safety, 0.0, 1.0)
                if (not feasible0) or (not np.isfinite(alpha)) or (alpha < 0):
                    alpha = 0.0
                projected[c] = (alpha < 1.0 - EPSILON)
            else:
                alpha = 1.0  # Direction does not increase constraint, take full step

            d_new[:, c] = d_prev[:, c] + alpha * dir_c  # new displacement
            alphas[c] = alpha

        return d_new, {"alpha": alphas, "projected": projected}

    def build_G_h_from_mbder(self, ctrl: np.array, tau: float = 1 / 3) -> Tuple[np.array, np.array]:
        """
        Build G, h constraint matrices like OSQP (but vectorized with numpy)
        Returns:
            G : (6*Ng, Nc) ; h : (6*Ng,)
        Such that  |C d| <= tau  with C = [Gx; Gy; Gz],
        where C maps a displacement DOF vector to physical-space gradient components at Gauss points.
        Tau: matches the "* 3.0" + h=1.0 → |C d| <= 1/3)
        """
        Ng, Nc = self.mbdx.shape
        assert ctrl.shape == (Nc, 3)

        # J = dX/d(xi,eta,zeta) at each Gauss point
        J = np.empty((Ng, 3, 3), dtype=self.dtype)
        J[:, 0, :] = self.mbdx @ ctrl
        J[:, 1, :] = self.mbdy @ ctrl
        J[:, 2, :] = self.mbdz @ ctrl

        # Solve J * X = I
        I = np.broadcast_to(np.eye(3, dtype=self.dtype), (Ng, 3, 3))
        J = J + EPSILON * I
        g_inv = np.linalg.solve(J, I)  # (Ng, 3, 3)

        # Stack parametric derivative operators:
        Mparam = np.stack([self.mbdx, self.mbdy, self.mbdz], axis=1)  # (Ng, 3, Nc)

        # C = g_inv @ Mparam along the middle dimension
        Cblocks = np.einsum('nij,njk->nik', g_inv, Mparam, optimize=True)  # (Ng, 3, Nc)

        # Order rows as [Gx0, Gy0, Gz0, Gx1, Gy1, Gz1, ...]
        C = Cblocks.reshape(3 * Ng, Nc)  # (3*Ng, Nc)

        # Two-sided bound for C
        G = np.vstack([C, -C], dtype=self.dtype)  # (6*Ng, Nc)
        h = np.full((G.shape[0],), tau, dtype=self.dtype)  # (6*Ng, 1)

        return G, h

    @staticmethod
    def rmse(phi: np.array, mesh: np.array, gp_points: np.array) -> float:
        prior_position = phi @ mesh
        return np.sqrt(np.mean(np.sum((gp_points - prior_position) ** 2, axis=1)))
