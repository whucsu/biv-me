import numpy as np
from copy import deepcopy
from pathlib import Path

from bivme.fitting.GPDataSet import GPDataSet
from bivme.fitting.surface_enum import ContourType

def adjust_boundary_weights(boundary, sWeights, tWeights):
    if int(boundary) & 1:
        tWeights[2] = tWeights[2] - tWeights[0]
        tWeights[1] = tWeights[1] + 2 * tWeights[0]
        tWeights[0] = 0

    if int(boundary) & 2:
        sWeights[1] = sWeights[1] - sWeights[3]
        sWeights[2] = sWeights[2] + 2 * sWeights[3]
        sWeights[3] = 0

    if int(boundary) & 4:
        tWeights[1] = tWeights[1] - tWeights[3]
        tWeights[2] = tWeights[2] + 2 * tWeights[3]
        tWeights[3] = 0

    if int(boundary) & 8:
        sWeights[2] = sWeights[2] - sWeights[0]
        sWeights[1] = sWeights[1] + 2 * sWeights[0]
        sWeights[0] = 0

    return sWeights, tWeights


def create_ref_dataset(config, folder, output_folder_models, filename_info, case_name, frame_names, logger):
    # Pull out config params
    bh_corr_method = config["breathhold_correction"]["shifting"]
    gp_ds_factor = config["gp_processing"]["sampling"]
    ed_frame = config["breathhold_correction"]["ed_frame"]
    logger.info(f'ED set to frame #{ed_frame}')

    # create log files where to store fitting errors and shift
    shift_file = output_folder_models / f"shift_file.txt"
    pos_file = output_folder_models / f"pos_file.txt"

    ed_dataset = None
    shift_to_apply = 0  # 2D translation
    updated_slice_position = 0

    if bh_corr_method == "derived_from_ed":
        logger.info("Shift measured only at ED frame")
        filename = Path(folder) / f"GPFile_{frame_names[ed_frame]:03}.txt"

        if not filename.exists():
            logger.error(f"Cannot find {filename} file! Skipping this model")
            return None
        
        ed_dataset = GPDataSet(str(filename), str(filename_info), case_name, sampling=gp_ds_factor, frame_num=ed_frame)
        if not ed_dataset.success:
            return None

        result_at_ed = ed_dataset.sinclair_slice_shifting(logger)
        _, _ = ed_dataset.get_unintersected_slices_fast()

        shift_to_apply = result_at_ed[0]
        updated_slice_position = result_at_ed[1]

        with shift_file.open("w", encoding="utf-8") as file:
            file.write("shift measured only at ED: frame " + str(ed_frame) + "\n")
            file.write(str(shift_to_apply))
            file.close()

        with pos_file.open("w", encoding="utf-8") as file:
            file.write("pos measured only at ED: frame " + str(ed_frame) + "\n")
            file.write(str(updated_slice_position))
            file.close()

    elif bh_corr_method == "average_all_frames":
        logger.info("Shift measured on all the frames and averaged")
        counter = 0
        frames_to_fit = sorted(np.unique(frame_names))  # if you want to fit all _frames#
        for frame in frames_to_fit:
            num = int(frame)
            filename = Path(folder) / f"GPFile_{num:03}.txt"
            if not filename.exists():
                logger.error(f"Cannot find {filename} file! Skipping this model")
                return None

            dataset = GPDataSet(str(filename), str(filename_info), case_name, sampling=gp_ds_factor, frame_num=num)

            if frame == frames_to_fit[ed_frame]:
                ed_dataset = deepcopy(dataset)
            if not dataset.success:
                continue
            result_at_t = dataset.sinclair_slice_shifting(logger)

            shift_to_apply += result_at_t[0]
            updated_slice_position += result_at_t[1]
            counter += 1

        shift_to_apply /= counter
        updated_slice_position /= counter

        with shift_file.open("w", encoding="utf-8") as file:
            file.write("Average shift \n")
            file.write(str(shift_to_apply))
            file.close()

        with pos_file.open("w", encoding="utf-8") as file:
            file.write("Average shift \n")
            file.write(str(updated_slice_position))
            file.close()

    elif bh_corr_method == "none":
        logger.info("No correction applied")
        filename = Path(folder) / f"GPFile_{frame_names[ed_frame]:03}.txt"
        if not filename.exists():
            logger.error(f"Cannot find {filename} file! Skipping this model")
            return None

        ed_dataset = GPDataSet(str(filename), str(filename_info), case_name, sampling=gp_ds_factor, frame_num=ed_frame)
        if not ed_dataset.success:
            return None

    else:
        logger.error(f'Method {bh_corr_method} unavailable.  '
                     f'Allowed values: none, derived_from_ed or average_all_frame. No correction applied')
        return None

    return ed_dataset, shift_to_apply, updated_slice_position


def gp_rv_valve_generator(gp_dataset, config, logger, rv_thickness=3):
    any_sax_rv_epi = np.any(gp_dataset.contour_type == ContourType.SAX_RV_EPICARDIAL)
    any_lax_rv_epi = np.any(gp_dataset.contour_type == ContourType.LAX_RV_EPICARDIAL)
    if not any_sax_rv_epi and not any_lax_rv_epi:
        logger.info('Generating RV epicardial points')
        _, _, _ = gp_dataset.create_rv_epicardium(rv_thickness)

    try:
        num_mv_phantom_pts = config["gp_processing"]["num_of_phantom_points_mv"]
        _ = gp_dataset.create_valve_phantom_points(num_mv_phantom_pts, ContourType.MITRAL_VALVE)
    except:
        logger.warning('Error in creating mitral phantom points')

    try:
        num_tv_phantom_pts = config["gp_processing"]["num_of_phantom_points_tv"]
        _ = gp_dataset.create_valve_phantom_points(num_tv_phantom_pts, ContourType.TRICUSPID_VALVE)
    except:
        logger.warning('Error in creating tricuspid phantom points')

    try:
        num_pv_phantom_pts = config["gp_processing"]["num_of_phantom_points_pv"]
        _ = gp_dataset.create_valve_phantom_points(num_pv_phantom_pts, ContourType.PULMONARY_VALVE)
    except:
        logger.warning('Error in creating pulmonary phantom points')

    try:
        num_av_phantom_pts = config["gp_processing"]["num_of_phantom_points_av"]
        _ = gp_dataset.create_valve_phantom_points(num_av_phantom_pts, ContourType.AORTA_VALVE)
    except:
        logger.warning('Error in creating aorta phantom points')


def set_default_weights(data_set):
    data_set.weights[data_set.contour_type == ContourType.MITRAL_PHANTOM] = 1
    data_set.weights[data_set.contour_type == ContourType.AORTA_PHANTOM] = 1
    data_set.weights[data_set.contour_type == ContourType.PULMONARY_PHANTOM] = 1
    data_set.weights[data_set.contour_type == ContourType.TRICUSPID_PHANTOM] = 1

    data_set.weights[data_set.contour_type == ContourType.APEX_POINT] = 1
    data_set.weights[data_set.contour_type == ContourType.RV_INSERT] = 2

    data_set.weights[data_set.contour_type == ContourType.MITRAL_VALVE] = 1
    data_set.weights[data_set.contour_type == ContourType.AORTA_VALVE] = 1
    data_set.weights[data_set.contour_type == ContourType.PULMONARY_VALVE] = 1
    data_set.weights[data_set.contour_type == ContourType.TRICUSPID_VALVE] = 1

