import os
import numpy as np

from bivme.fitting.surface_enum import *
from bivme.fitting.GPDataSet import GPDataSet


def write_to_gp_file(path, coords, label, slice_id, weight=1.0, phase=1.0):

    """ Writes a coordinate/line to the guide point file """
    
    # check if output exists
    if os.path.exists(path):
        flag = 'a'
    else:
        flag = 'w'
    
    count = 0
    while count<100000000:
        try:
            with open(os.path.join(path), flag) as f:
                if flag == 'w':
                    f.write('x\ty\tz\tcontour type\tframeID\tweight\ttime frame\n')
                for coord in coords:
                    f.write('{:.5f}\t{:.5f}\t{:.5f}\t{}\t{}\t{}\t{}\n'.format(coord[0], coord[1], coord[2],
                                                                label, slice_id, weight, phase))
                break
        except:
            count += 1
            pass


def inverse_coordinate_transformation(coordinate, imagePositionPatient, imageOrientationPatient, ps):

    """ Performs a coordinate transformation from image coordinates to patient coordinates """

    # image position and orientation
    S = imagePositionPatient
    X = imageOrientationPatient[:3]
    Y = imageOrientationPatient[3:]

    # construct affine transform
    M = np.asarray([[X[0]*ps[0], Y[0]*ps[1], 0, S[0]],
                [X[1]*ps[0], Y[1]*ps[1], 0, S[1]],
                [X[2]*ps[0], Y[2]*ps[1], 0, S[2]],
                [0, 0, 0, 1]])

    coord = np.array([coordinate[0], coordinate[1], 0, 1.0])
    
    # perform transformation and return as list
    return [np.round(x,5) for x in M @ coord.T]


def remove_misaligned_rv_sax_contours(dataset, my_logger, frame):
    point_coords = dataset.points_coordinates
    point_slices = dataset.slice_number
    point_labels = dataset.contour_type

    rv_fw_idx = np.where(point_labels == ContourType.SAX_RV_FREEWALL)[0]
    rv_s_idx = np.where(point_labels == ContourType.SAX_RV_SEPTUM)[0]
    rv_endo_idx = np.concat((rv_fw_idx, rv_s_idx))
    rv_epi_idx = np.where(point_labels == ContourType.SAX_RV_EPICARDIAL)[0]
    rv_inserts_idx = np.where(point_labels == ContourType.RV_INSERT)[0]

    rv_endo_points = {}
    rv_epi_points = {}
    rv_insert_points = {}

    for endo_idx in rv_endo_idx:
        # Endocardial points
        slice_id = point_slices[endo_idx]
        if slice_id not in rv_endo_points:
            rv_endo_points[slice_id] = []
        rv_endo_points[slice_id].append(point_coords[endo_idx])

    for epi_idx in rv_epi_idx:
        # Epicardial points
        slice_id = point_slices[epi_idx]
        if slice_id not in rv_epi_points:
            rv_epi_points[slice_id] = []
        rv_epi_points[slice_id].append(point_coords[epi_idx])

    for insert_idx in rv_inserts_idx:
        # RV Insert points
        slice_id = point_slices[insert_idx]
        if slice_id not in rv_insert_points:
            rv_insert_points[slice_id] = []
        rv_insert_points[slice_id].append(point_coords[insert_idx])

    # Get centroid of RV points per slice
    rv_centroids = {}
    for slice_id in rv_endo_points.keys():
        all_points = rv_endo_points[slice_id] + rv_epi_points.get(slice_id, [])
        if len(all_points) == 0:
            continue
        centroid = np.mean(all_points, axis=0)
        rv_centroids[slice_id] = centroid
    
    # Get the pair of centroids that are furthest apart
    max_dist = 0
    limit_pair = [None, None]
    for sid1 in rv_centroids.keys():
        for sid2 in rv_centroids.keys():
            if sid1 >= sid2:
                continue
            dist = np.linalg.norm(rv_centroids[sid1] - rv_centroids[sid2])
            if dist > max_dist:
                max_dist = dist
                limit_pair = [sid1, sid2]

    if limit_pair[0] is None or limit_pair[1] is None:
        return dataset

    # Create a line between these two centroids
    point1 = rv_centroids[limit_pair[0]]
    point2 = rv_centroids[limit_pair[1]]
    line_vector = point2 - point1
    line_vector = line_vector / np.linalg.norm(line_vector)

    # Sort slices along this line
    slice_projections = {}
    for slice_id, centroid in rv_centroids.items():
        vec_to_centroid = centroid - point1
        projection = np.dot(vec_to_centroid, line_vector)
        slice_projections[slice_id] = projection
    sorted_slices = sorted(slice_projections.items(), key=lambda x: x[1])

    # Find any outlier slices based on distance from centroid trend
    slice_ids = [slice_id for slice_id, _ in sorted_slices]
    dists = {}
    for i in range(len(slice_ids)):
        previous_centroid = rv_centroids[slice_ids[i-1]] if i-1 >= 0 else None
        current_centroid = rv_centroids[slice_ids[i]]
        next_centroid = rv_centroids[slice_ids[i+1]] if i+1 < len(slice_ids) else None
        
        dist_prev = np.linalg.norm(current_centroid - previous_centroid) if previous_centroid is not None else 0
        dist_next = np.linalg.norm(current_centroid - next_centroid) if next_centroid is not None else 0
        dists[slice_ids[i]] = [dist_prev, dist_next]

    dist_avg = []
    for slice_id in slice_ids:
        dist_values = dists[slice_id]
        avg_dist = np.mean([d for d in dist_values if d > 0])
        dist_avg.append(avg_dist)

    dist_avg = np.array(dist_avg)
    dist_avg = np.mean(dist_avg)

    idx_to_del = []
    for i, slice_id in enumerate(slice_ids):
        if (dists[slice_id][0] > 2.0 * dist_avg and dists[slice_id][1] > 2.0 * dist_avg) or (dists[slice_id][0] > 2.0 * dist_avg) and (dists[slice_id][1] == 0) or (dists[slice_id][0] == 0 and dists[slice_id][1] > 2.0 * dist_avg):
            # Outlier detected, remove RV points from this slice
            my_logger.warning(f'RV SAX points on slice {slice_id} at frame {frame} are not aligned with the other SAX slices! Removing RV inserts...')
            # for endo_idx in rv_endo_idx:
            #     if point_slices[endo_idx] == slice_id:
            #         idx_to_del.append(endo_idx)
            # for epi_idx in rv_epi_idx:
            #     if point_slices[epi_idx] == slice_id:
            #         idx_to_del.append(epi_idx)
            for insert_idx in rv_inserts_idx:
                if point_slices[insert_idx] == slice_id:
                    idx_to_del.append(insert_idx)

    dataset.remove_data_points(idx_to_del)

    return dataset


def remove_misaligned_lv_sax_contours(dataset, my_logger, frame):
    point_coords = dataset.points_coordinates
    point_slices = dataset.slice_number
    point_labels = dataset.contour_type

    lv_endo_idx = np.where(point_labels == ContourType.SAX_LV_ENDOCARDIAL)[0]
    lv_epi_idx = np.where(point_labels == ContourType.SAX_LV_EPICARDIAL)[0]

    lv_endo_points = {}
    lv_epi_points = {}

    for endo_idx in lv_endo_idx:
        # Endocardial points
        slice_id = point_slices[endo_idx]
        if slice_id not in lv_endo_points:
            lv_endo_points[slice_id] = []
        lv_endo_points[slice_id].append(point_coords[endo_idx])

    for epi_idx in lv_epi_idx:
        # Epicardial points
        slice_id = point_slices[epi_idx]
        if slice_id not in lv_epi_points:
            lv_epi_points[slice_id] = []
        lv_epi_points[slice_id].append(point_coords[epi_idx])

    # Get centroid of LV points per slice
    lv_centroids = {}
    for slice_id in lv_endo_points.keys():
        all_points = lv_endo_points[slice_id] + lv_epi_points.get(slice_id, [])
        if len(all_points) == 0:
            continue
        centroid = np.mean(all_points, axis=0)
        lv_centroids[slice_id] = centroid

    # Get the pair of centroids that are furthest apart
    max_dist = 0
    limit_pair = [None, None]
    for sid1 in lv_centroids.keys():
        for sid2 in lv_centroids.keys():
            if sid1 >= sid2:
                continue
            dist = np.linalg.norm(lv_centroids[sid1] - lv_centroids[sid2])
            if dist > max_dist:
                max_dist = dist
                limit_pair = [sid1, sid2]

    if limit_pair[0] is None or limit_pair[1] is None:
        return dataset

    # Create a line between these two centroids
    point1 = lv_centroids[limit_pair[0]]
    point2 = lv_centroids[limit_pair[1]]
    line_vector = point2 - point1
    line_vector = line_vector / np.linalg.norm(line_vector)

    # Sort slices along this line
    slice_projections = {}
    for slice_id, centroid in lv_centroids.items():
        vec_to_centroid = centroid - point1
        projection = np.dot(vec_to_centroid, line_vector)
        slice_projections[slice_id] = projection
    sorted_slices = sorted(slice_projections.items(), key=lambda x: x[1])

    # Find any outlier slices based on distance from centroid trend
    slice_ids = [slice_id for slice_id, _ in sorted_slices]
    dists = {}
    for i in range(len(slice_ids)):
        previous_centroid = lv_centroids[slice_ids[i-1]] if i-1 >= 0 else None
        current_centroid = lv_centroids[slice_ids[i]]
        next_centroid = lv_centroids[slice_ids[i+1]] if i+1 < len(slice_ids) else None
        
        dist_prev = np.linalg.norm(current_centroid - previous_centroid) if previous_centroid is not None else 0
        dist_next = np.linalg.norm(current_centroid - next_centroid) if next_centroid is not None else 0
        dists[slice_ids[i]] = [dist_prev, dist_next]

    dist_avg = []
    for slice_id in slice_ids:
        dist_values = dists[slice_id]
        avg_dist = np.mean([d for d in dist_values if d > 0])
        dist_avg.append(avg_dist)

    dist_avg = np.array(dist_avg)
    dist_avg = np.mean(dist_avg)

    idx_to_del = []
    for i, slice_id in enumerate(slice_ids):
        if (dists[slice_id][0] > 2.0 * dist_avg and dists[slice_id][1] > 2.0 * dist_avg) or (dists[slice_id][0] > 2.0 * dist_avg) and (dists[slice_id][1] == 0) or (dists[slice_id][0] == 0 and dists[slice_id][1] > 2.0 * dist_avg):
            # Outlier detected, remove LV points from this slice
            my_logger.warning(f'LV SAX points on slice {slice_id} at frame {frame} are not aligned with the other SAX slices! Removing...')

            for endo_idx in lv_endo_idx:
                if point_slices[endo_idx] == slice_id:
                    idx_to_del.append(endo_idx)
            for epi_idx in lv_epi_idx:
                if point_slices[epi_idx] == slice_id:
                    idx_to_del.append(epi_idx)
    
    dataset.remove_data_points(idx_to_del)

    return dataset


def remove_misaligned_lv_lax_contours(dataset, my_logger, frame):
    point_coords = dataset.points_coordinates
    point_slices = dataset.slice_number
    point_labels = dataset.contour_type

    mv_idx = np.where(point_labels == ContourType.MITRAL_VALVE)[0]
    tv_idx = np.where(point_labels == ContourType.TRICUSPID_VALVE)[0]
    apex_idx = np.where(point_labels == ContourType.APEX_POINT)[0]

    mv_points = {}
    tv_points = {}
    apex_points = {}

    for idx in mv_idx:
        slice_id = point_slices[idx]
        mv_points[slice_id] = point_coords[idx]
    for idx in tv_idx:
        slice_id = point_slices[idx]
        tv_points[slice_id] = point_coords[idx]
    for idx in apex_idx:
        slice_id = point_slices[idx]
        apex_points[slice_id] = point_coords[idx]

    idx_to_del = []
    for slice_id in mv_points.keys():
        # Check non-4ch slices against 4ch
        if slice_id in mv_points and slice_id in apex_points:
            apex_point = np.array(apex_points[slice_id])

            try:
                tv_points_array = [point for sid, point in tv_points.items() if sid != slice_id]
                mv_points_array = [point for sid, point in mv_points.items() if sid != slice_id]
            except Exception as e:
                tv_points_array = []
                mv_points_array = []

            # If apex point is closer to mitral valve than the mitral valve to tricuspid valve distance, remove the slice
            if len(tv_points_array) == 0 or len(mv_points_array) == 0:
                continue
            mv_centroid = np.mean(np.array(mv_points_array), axis=0)
            tv_centroid = np.mean(np.array(tv_points_array), axis=0)
            dist_mv_apex = np.linalg.norm(mv_centroid - apex_point)
            dist_mv_tv = np.linalg.norm(mv_centroid - tv_centroid)

            if dist_mv_apex < dist_mv_tv:
                my_logger.warning(f'LAX landmarks on slice {slice_id} at frame {frame} are misaligned! Removing this slice...')
                for idx in range(len(point_labels)):
                    if point_slices[idx] == slice_id:
                        idx_to_del.append(idx)

    dataset.remove_data_points(idx_to_del)

    return dataset


def remove_misplaced_contours(dataset, my_logger, frame):
    point_coords = dataset.points_coordinates
    point_slices = dataset.slice_number
    point_labels = dataset.contour_type

    # Get sax points
    sax_points_idx = np.where((point_labels == ContourType.SAX_LV_ENDOCARDIAL) | (point_labels == ContourType.SAX_LV_EPICARDIAL) | (point_labels == ContourType.SAX_RV_FREEWALL) | (point_labels == ContourType.SAX_RV_SEPTUM) | (point_labels == ContourType.SAX_RV_EPICARDIAL))[0]
    sax_points = point_coords[sax_points_idx]

    # Get landmarks
    mv_points_idx = np.where(point_labels == ContourType.MITRAL_VALVE)[0]
    mv_points = point_coords[mv_points_idx]
    tv_points_idx = np.where(point_labels == ContourType.TRICUSPID_VALVE)[0]
    tv_points = point_coords[tv_points_idx]
    apex_points_idx = np.where(point_labels == ContourType.APEX_POINT)[0]
    apex_points = point_coords[apex_points_idx]

    # Define unique slices that contain points that are not SAX
    sax_slices = point_slices[sax_points_idx]
    non_sax_slices = set(point_slices) - set(sax_slices)
    unique_slices = set(non_sax_slices)

    # Concatenate SAX and landmark points
    roi_points = np.concatenate((sax_points, mv_points, tv_points, apex_points), axis=0)

    # Get xyz extremes of SAX and landmark points to get rough ROI for cardiac region
    max_x, min_x = np.max(roi_points[:, 0]), np.min(roi_points[:, 0])
    max_y, min_y = np.max(roi_points[:, 1]), np.min(roi_points[:, 1])
    max_z, min_z = np.max(roi_points[:, 2]), np.min(roi_points[:, 2])

    # Delete points outside of this ROI (with a 50% tolerance either side)
    tolerance = 0.5
    idx_to_del = []
    for slice_id in unique_slices:
        outside = False
        this_slice_points = point_coords[point_slices == slice_id]
        for idx, point in enumerate(this_slice_points):
            if point_labels[idx] not in [ContourType.LAX_LA, ContourType.LAX_RA, ContourType.MITRAL_VALVE, ContourType.AORTA_VALVE, ContourType.TRICUSPID_VALVE, ContourType.PULMONARY_VALVE, ContourType.APEX_POINT]: # no need to check atrial points or landmarks as they can be outside of the roi
                if point[0] < min_x - tolerance * (max_x - min_x) or point[0] > max_x + tolerance * (max_x - min_x) or \
                   point[1] < min_y - tolerance * (max_y - min_y) or point[1] > max_y + tolerance * (max_y - min_y) or \
                   point[2] < min_z - tolerance * (max_z - min_z) or point[2] > max_z + tolerance * (max_z - min_z):

                    outside = True
                    idx_to_del.append(idx)
                    
        if outside:
            my_logger.warning(f'Guide points on slice {slice_id} at frame {frame} are outside of the cardiac region! Removing the offending points...')

    dataset.remove_data_points(idx_to_del)

    return dataset


def postprocess_guidepoints(gpfilepath: str, metadatafilepath:str, case_name:str, frame:int, my_logger) -> None:
    """ Post-process guidepoints"""
    dataset = GPDataSet(gpfilepath, metadatafilepath, case=case_name, sampling=1, frame_num=frame)

    if dataset is None:
        my_logger.error(f'Could not load guide point file for frame {frame}. Skipping post-processing of guidepoints for this frame...')
        return
    
    # Check if there are any tricuspid valve points / mitral valve points / apex points - if not, skip post-processing as we won't be able to correctly identify misaligned slices without these landmarks
    point_labels = dataset.contour_type
    tv_idx = np.where(point_labels == ContourType.TRICUSPID_VALVE)[0]
    mv_idx = np.where(point_labels == ContourType.MITRAL_VALVE)[0]
    apex_idx = np.where(point_labels == ContourType.APEX_POINT)[0]
    if len(tv_idx) == 0:
        my_logger.warning(f'No tricuspid valve points detected at frame {frame}. Skipping post-processing of guidepoints for this frame...')
        return
    elif len(mv_idx) == 0:
        my_logger.warning(f'No mitral valve points detected at frame {frame}. Skipping post-processing of guidepoints for this frame...')
        return
    elif len(apex_idx) == 0:
        my_logger.warning(f'No apex points detected at frame {frame}. Skipping post-processing of guidepoints for this frame...')
        return

    # Post-process guidepoints
    # 1. Remove LV SAX contours that are not aligned with the LV axis
    dataset = remove_misaligned_lv_sax_contours(dataset, my_logger, frame)

    # 2. Remove RV inserts if contours are not aligned with the RV axis
    dataset = remove_misaligned_rv_sax_contours(dataset, my_logger, frame)

    # 3. Remove LAX slices if landmarks are upside down
    dataset = remove_misaligned_lv_lax_contours(dataset, my_logger, frame)

    # 4. Remove any points that are not in the cardiac region (e.g., due to breathing artefacts or bad segmentation)
    dataset = remove_misplaced_contours(dataset, my_logger, frame)

    # Save updated guidepoints back to file
    dataset.write_gpfile(gpfilepath, time_frame=frame, overwrite=True)
