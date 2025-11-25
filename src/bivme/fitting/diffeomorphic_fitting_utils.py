import time

from bivme.fitting import BiventricularModel
from bivme.fitting import GPDataSet
import numpy as np
from loguru import logger

from .ExplicitFitterNumpy import ExplicitFitterNumpy

EPSILON = 1e-12


# noinspection PyArgumentList
def solve_convex_fast(
        biv_model: BiventricularModel,
        data_set: GPDataSet,
        weight_gp: float,
        low_smoothing_weight: float,
        transmural_weight: float,
        my_logger: logger) -> float:
    """
    This function performs an accelerated diffeomorphic fit with numpy.
    """
    start_time = time.time()
    my_logger.info(f"START -> Explicit OSQP-style fitting...")

    # Set up fitter
    fitter = ExplicitFitterNumpy(logger, biv_model, weight_gp, data_set)

    # Solve and update mesh
    new_mesh, final_stats = fitter.explicit_refine_osqp_style(max_outer=5,
                                                              min_jacobian=0.1,
                                                              transmural_weight=transmural_weight,
                                                              smoothing_weight=low_smoothing_weight,
                                                              tau=1 / 3,
                                                              resid_conv_tol=5e-4,
                                                              disp_conv_tol=1e-4,
                                                              relinearize=False)
    biv_model.update_control_mesh(new_mesh)

    final_time = time.time() - start_time
    iters = final_stats["iters"]
    residuals = final_stats["weighted_rmse"]

    my_logger.success(f"End of the explicitly constrained fit. Time taken: {final_time}")
    my_logger.success(f"PARAMS FOR TESTING: {iters} {residuals} {final_time}")

    return residuals


def fit_least_squares_model(biv_model,
                            weight_gp,
                            data_set,
                            trans_weight,
                            smoothing_factor):
    index, weights, _, projected_points_basis_coeff = biv_model.compute_data_xi_fast(weight_gp, data_set)

    prior_position = projected_points_basis_coeff @ biv_model.control_mesh
    w = weights[:, np.newaxis]  # Turn into column vector for broadcasting
    w_pg = projected_points_basis_coeff * w  # Element-wise multiply each row by the corresponding weight
    GTPTWTWPG = w_pg.T @ w_pg

    # Gram Matrix + Regularization (LHS)
    smooth_constraint = (biv_model.gtstsg_x + biv_model.gtstsg_y + (trans_weight * biv_model.gtstsg_z))
    regularizer = smoothing_factor * smooth_constraint
    A = GTPTWTWPG + regularizer

    # Weighted residuals (RHS)
    data_points_position = data_set.points_coordinates[index]
    wd = (data_points_position - prior_position) * w
    rhs = w_pg.T @ wd

    # Solve least squares problem
    solf = np.linalg.solve(A.T.dot(A), A.T.dot(rhs))
    err = np.sqrt(np.mean(np.sum((data_points_position - prior_position) ** 2, axis=1)))

    return solf, err


def solve_least_squares_problem(biv_model: BiventricularModel,
                                weight_gp: float,
                                data_set: GPDataSet,
                                trans_weight,
                                my_logger):
    start_time = time.time()

    # Establish implicit fitting parameters
    lambda_high = weight_gp * 1e10  # First regularization weight (dynamic)
    lambda_low = weight_gp * 1e2  # Lowest regularization weight
    num_iters = 20  # Maximum number of iterations
    iteration = 0
    factor = 10  # Initial lambda reduction factor (dynamic)
    min_jacobian = 0.1  # Minimum allowed Jacobian
    err = float('nan')
    last_err = float('nan')
    diffeo_tries = 0

    residual_conv_tol = 5e-4
    disp_conv_tol = 1e-4
    my_logger.info(f"START -> Implicit least-squares fitting...")

    for iteration in range(num_iters):
        if (lambda_high <= lambda_low):
            my_logger.info(f"STOP -> Reached lower regularization limit = {lambda_low}")
            break

        if (diffeo_tries == 3):
            my_logger.info(f"STOP -> Could not find diffeomorphic solution.")
            break

        displacement, err = fit_least_squares_model(biv_model, weight_gp, data_set, trans_weight, lambda_high)
        my_logger.info(f"     Iteration {iteration} Weight {lambda_high}    ICF error {err}")

        mesh_try = biv_model.control_mesh + displacement
        if biv_model.is_diffeomorphic(mesh_try, min_jacobian):
            diffeo_tries = 0
            biv_model.update_control_mesh(mesh_try)

            # Check early-stop residual convergence
            resid_converged = (abs(last_err - err) / max(last_err, EPSILON)) < residual_conv_tol
            if resid_converged:
                my_logger.info(f"STOP -> Residuals converged.")
                break

            # Check early-stop displacement convergence
            disp_converged = (np.linalg.norm(displacement) < disp_conv_tol)
            if disp_converged:
                my_logger.info(f"STOP -> Displacement vector converged.")
                break

            last_err = err
            factor = max(factor - 1, 2)  # Dynamic factor reduction (decay-like reduction)
            lambda_high = (lambda_high / factor)  # we divide weight by 'factor' and start again...
        else:
            diffeo_tries += 1
            # If not diffeomorphic, the model is not updated.
            # Try again with a smaller step (inversely proportional to number of tries)
            orig_lambda = lambda_high * factor  # Revert to last successful lambda
            new_lambda = orig_lambda - ((orig_lambda - lambda_high) / (diffeo_tries + 1))
            factor = orig_lambda / new_lambda
            lambda_high = orig_lambda / factor

    final_time = time.time() - start_time
    my_logger.success(f"End of convex optimization. Time taken: {final_time}")

def generate_contraint_matrix(mesh):
    """
    Constraint matrix generator.
    Assumes:
        mesh.mbder_dx, mbder_dy, mbder_dz: (N, number_of_control_points)
        mesh.control_mesh: (number_of_control_points, 3)
    """

    mbdx, mbdy, mbdz = mesh.mbder_dx, mesh.mbder_dy, mesh.mbder_dz
    ctrl = mesh.control_mesh  # shape: (388, 3)
    N, K = mbdx.shape

    dXdxi = np.empty((N, 3, 3), dtype=np.float64)
    dXdxi[:, 0, :] = mbdx @ ctrl  # each row: (388,) @ (388,3) = (3,)
    dXdxi[:, 1, :] = mbdy @ ctrl
    dXdxi[:, 2, :] = mbdz @ ctrl

    g_inv = np.linalg.inv(dXdxi)  # shape (N, 3, 3)

    Gx = (
            g_inv[:, 0, 0][:, None] * mbdx +
            g_inv[:, 0, 1][:, None] * mbdy +
            g_inv[:, 0, 2][:, None] * mbdz
    )
    Gy = (
            g_inv[:, 1, 0][:, None] * mbdx +
            g_inv[:, 1, 1][:, None] * mbdy +
            g_inv[:, 1, 2][:, None] * mbdz
    )
    Gz = (
            g_inv[:, 2, 0][:, None] * mbdx +
            g_inv[:, 2, 1][:, None] * mbdy +
            g_inv[:, 2, 2][:, None] * mbdz
    )

    # Stack without vstack overhead
    out = np.empty((3 * N, K), dtype=np.float64)
    out[0::3, :] = Gx
    out[1::3, :] = Gy
    out[2::3, :] = Gz

    return out
