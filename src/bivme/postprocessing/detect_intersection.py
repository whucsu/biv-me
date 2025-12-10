import numpy as np
import argparse
import os, sys
from pathlib import Path
import re
import fnmatch
from copy import deepcopy

from bivme import MODEL_RESOURCE_DIR
from bivme.fitting.BiventricularModel import BiventricularModel
from loguru import logger
from rich.progress import Progress
import tomli
import shutil
from bivme.fitting.surface_enum import Surface
from bivme.fitting.surface_enum import ContourType
from bivme.fitting.GPDataSet import GPDataSet

biv_model_folder = MODEL_RESOURCE_DIR
from bivme.fitting.diffeomorphic_fitting_utils import (
    solve_least_squares_problem,
    solve_convex_fast,
)
from bivme.fitting.perform_fit import write_all_biv_models

# mapping 
contour_map = {
   Surface.RV_FREEWALL : ContourType.SAX_RV_FREEWALL,
   Surface.RV_SEPTUM : ContourType.SAX_RV_SEPTUM,
   Surface.LV_ENDOCARDIAL : ContourType.SAX_LV_ENDOCARDIAL,
   Surface.RV_INSERT : ContourType.RV_INSERT,
   Surface.APEX : ContourType.APEX_POINT,
   Surface.MITRAL_VALVE : ContourType.MITRAL_VALVE,
   Surface.TRICUSPID_VALVE : ContourType.TRICUSPID_VALVE,
   Surface.AORTA_VALVE : ContourType.AORTA_VALVE,
   Surface.PULMONARY_VALVE : ContourType.PULMONARY_VALVE}
   # : ContourType.SAX_RV_EPICARDIAL
   # : ContourType.SAX_LV_EPICARDIAL

def fix_intersection(case_name: str, config: dict, model_file: os.PathLike, output_folder: os.PathLike, biv_model_folder: os.PathLike = MODEL_RESOURCE_DIR) -> None:
    """
        # Authors: cm
        # Date: 09/24
    """

    reference_biventricular_model = BiventricularModel(biv_model_folder, collision_detection = True)

    fitted_model = deepcopy(reference_biventricular_model)
    fitted_model.control_mesh = np.loadtxt(model_file, delimiter=',', skiprows=1, usecols=[0, 1, 2]).astype(float)
    fitted_model.update_control_mesh(fitted_model.control_mesh)
    current_collision = fitted_model.detect_collision()
    inter = current_collision.difference(fitted_model.reference_collision) 

    frame_num = int(os.path.basename(os.path.normpath(model_file)).split('_')[-1].replace('.txt',''))

    if bool(inter):
        
        logger.warning(f"Intersections detected for case {os.path.basename(os.path.normpath(model_file))}")        
        logger.info(f"Refitting of frame {frame_num:03d}")

        # create a separate output folder for each patient
        output_folder = Path(output_folder) / os.path.basename(case_name)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # initialise GP dataset from fitted model
        gp_dataset = GPDataSet()

        points = []
        slices = []
        contour_types = []
        weights = []
        for surface, contours in contour_map.items():
            start, end = fitted_model.get_surface_vertex_start_end_index(surface)
            for idx in range(start, end):
                points.append(fitted_model.et_pos[idx,:])
                slices.append(0)
                contour_types.append(contours)
                weights.append(1.0)

        gp_dataset.points_coordinates = np.array(points)
        gp_dataset.slice_number = np.array(slices)
        gp_dataset.contour_type = np.array(contour_types)
        gp_dataset.weights = np.array(weights)

        residuals = 0

        gp_dataset.apex = fitted_model.et_pos[fitted_model.APEX_INDEX,]
        gp_dataset.mitral_centroid = fitted_model.et_pos[fitted_model.get_surface_vertex_start_end_index(Surface.MITRAL_VALVE)[1],:]
        gp_dataset.tricuspid_centroid = fitted_model.et_pos[fitted_model.get_surface_vertex_start_end_index(Surface.TRICUSPID_VALVE)[1],:]

        reference_biventricular_model.update_pose_and_scale(gp_dataset)

        # Get config parameters upfront
        gp_weight = config["fitting_weights"]["guide_points"]
        conv_weight = config["fitting_weights"]["convex_problem"]
        trans_weight = config["fitting_weights"]["transmural"]
        lsq_trans_weight = config["fitting_weights"]["lsq_trans_weight"]

        # Perform least squares fit
        biv_model = reference_biventricular_model.copy()
        solve_least_squares_problem(biv_model, gp_weight, gp_dataset, lsq_trans_weight, collision_detection=True, model_prior = fitted_model, my_logger=logger)
  
        ## Perform diffeomorphic fit
        residuals = solve_convex_fast(biv_model, gp_dataset, gp_weight, conv_weight, trans_weight, collision_detection=True, model_prior = fitted_model, my_logger=logger)

        return biv_model, residuals

    else:
        logger.success(f"No intersection detected for frame {frame_num:03d} - moving on")
        return None


if __name__ == "__main__":
    biv_resource_folder = MODEL_RESOURCE_DIR

    parser = argparse.ArgumentParser(description="Removes intersection between free wall and septum if presents")
    parser.add_argument('-config', '--config_file', type=str,
                        help='Config file containing fitting parameters (the one used for fitting).')
    args = parser.parse_args()

    # Load config  - the config needs to be the same as the one used for fitting!
    assert Path(args.config_file).exists(), \
        f'Cannot not find {args.config_file}!'
    with open(args.config_file, mode="rb") as fp:
        config = tomli.load(fp)

    # TOML Schema Validation
    match config:
        case {
            "modules": {"preprocessing": bool(), "fitting": bool()},

            "logging": {"show_detailed_logging": bool(), "generate_log_file": bool()},

            "plotting": {"generate_plots_preprocessing": bool(), "generate_plots_fitting": bool(), "include_images": bool(), "export_images": bool()},

            "input_pp": {"source": str(),
                        "batch_ID": str(),
                        "analyst_id": str(),
                        "processing": str(),
                        "states": str()
                        },
            "view-selection": {"option": str(), "correct_mode": str()},
            "contouring": {"smooth_landmarks": bool()},
            "output_pp": {"overwrite": bool(), "output_directory": str()},

            "input_fitting": {"gp_directory": str(),
                        "gp_suffix": str(),
                        "si_suffix": str(),
                        },
            "breathhold_correction": {"shifting": str(), "ed_frame": int()},
            "gp_processing": {"sampling": int(), "num_of_phantom_points_av": int(), "num_of_phantom_points_mv": int(), "num_of_phantom_points_tv": int(), "num_of_phantom_points_pv": int()},
            "multiprocessing": {"workers": int()},
            "fitting_weights": {"guide_points": float(), "convex_problem": float(), "transmural": float(), "lsq_trans_weight": float()},
            "output_fitting": {"output_directory": str(), "output_meshes": list(), "closed_mesh": bool(),   "export_control_mesh": bool(), "mesh_format": str(),  "overwrite": bool()},
        }:
            pass
        case _:
            raise ValueError(f"Invalid configuration: {config}")

    if not config["logging"]["show_detailed_logging"]:
        logger.remove()

    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    if not (config["output_fitting"]["mesh_format"].endswith('.obj') or config["output_fitting"]["mesh_format"].endswith('.vtk') or config["output_fitting"]["mesh_format"] == "none"):
        logger.error(f'argument mesh_format must be .obj or .vtk. {config["output_fitting"]["mesh_format"]} given.')
        sys.exit(0)

    for mesh in config["output_fitting"]["output_meshes"]:
        if mesh not in ["LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"]:
            logger.error(f'argument output_meshes invalid. {mesh} given. Allowed values are "LV_ENDOCARDIAL", "RV_ENDOCARDIAL", "EPICARDIAL"')
            sys.exit(0)

    # save config file to the output folder
    output_folder = Path(config["output_fitting"]["output_directory"]) # Save in place
    if not os.path.exists(output_folder):
        logger.warning(f"Output folder {output_folder} does not exist. Exiting.")
        sys.exit(0)

    case_list = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder,f))]
    folders = [Path(config["output_fitting"]["output_directory"], case).as_posix() for case in case_list]

    logger.info(f"Found {len(folders)} model folders.")

    if len(folders) == 0:
        logger.warning(f"No model folders found in {config['output_fitting']['output_directory']}. Exiting")
        sys.exit(0)

    try:
        for i, folder in enumerate(folders):
            if os.path.isdir(folder):
                rule = re.compile(fnmatch.translate("*model_frame*.txt"), re.IGNORECASE)
                models = [folder / Path(name) for name in os.listdir(Path(folder)) if rule.match(name)]
                models = sorted(models)
                model_dict = {}
                case_name = os.path.basename(os.path.normpath(folder))

                logger.info(f"Processing {str(folder)} ({i+1}/{len(folders)})")
                with Progress(transient=True) as progress:
                    task = progress.add_task("Checking for intersection...", total=len(models))
                    console = progress

                    for idx, biv_model_file in enumerate(models):
                        pack = fix_intersection(folder, config, biv_model_file, output_folder, biv_resource_folder)
                        if pack is not None:
                            corrected_model, res = pack
                            model_dict[idx] = corrected_model
                            
                        progress.advance(task)

                # Export corrected models
                if len(model_dict) > 0:
                    write_all_biv_models(config, model_dict, config["output_fitting"]["mesh_format"], output_folder / case_name, case_name, logger)
                else:
                    logger.info(f"No intersections detected for case {case_name}. No refitting performed.")

        logger.success(f"Done. Results are saved in {output_folder}")
    except KeyboardInterrupt:
        logger.info(f"Program interrupted by the user")
        sys.exit(0)