import numpy as np
import cv2


LABEL_MAP = {"SAX":  { "LV_endo": 1,
                       "LV_myo": 2,
                       "RV_endo": 3,
                       "RV_myo": 4},
             "2ch":  { "LV_endo": 1,
                       "LV_myo": 2,
                       "LA_endo": 3},
             "3ch":  { "LV_endo": 1,
                       "LV_myo": 2,
                       "RV_endo": 3,
                       "LA_endo": 4,
                       "Aorta": 5,
                       "RV_myo": 6},
             "4ch":  { "LV_endo": 1,
                       "LV_myo": 2,
                       "RV_endo": 3,
                       "LA_endo": 4,
                       "RA_endo": 5,
                       "RV_myo": 6},
             "RVOT": { "RV_endo": 1,
                       "RV_myo": 2,
                       "PA": 3}
            }


def get_intersections(point_list1, point_list2, distance_cutoff=4.5):

    """ Finds the points that are within a given cutoff distance between two lists """

    a = range(len(point_list1))
    b = range(len(point_list2))
    [A, B] = np.meshgrid(a,b)
    c = np.concatenate([A.T, B.T], axis=0)
    pairs = c.reshape(2,-1).T
    try:
        dist = np.sqrt( ( ( point_list1[pairs[:,0],0] - point_list2[pairs[:,1],0] ) ** 2 ) + 
                        ( ( point_list1[pairs[:,0],1] - point_list2[pairs[:,1],1] ) ** 2 ) )
    except:
        return np.array([])
    
    pairs = pairs[np.where(dist < distance_cutoff)[0].tolist()]

    return pairs


def get_valve_points_from_intersections(segmentation, endolabel, superlabel, distance_cutoff=2):
    ## TODO: Pass in contours not segmentation
    endo = (segmentation == endolabel).astype(np.uint8)
    superior = (segmentation == superlabel).astype(np.uint8)
    # Get contours
    contours, hierarchy = cv2.findContours(cv2.inRange(endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        endopts = []
        for c in contours:
            endopts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        endopts = np.array(endopts, dtype=np.int64)
    else:
        endopts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(superior, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        suppts = []
        for c in contours:
            suppts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        suppts = np.array(suppts, dtype=np.int64)
    else:
        suppts = []
    
    # Get intersection points between endo and superior
    while distance_cutoff < 10:
        pairs = get_intersections(endopts, suppts, distance_cutoff)
        if len(pairs) > 2:
            break
        else:
            distance_cutoff += 0.5

    # Valve points will be extents of the intersection points
    # valveplane = suppts[pairs[:,1],:]
    valveplane = endopts[pairs[:,0],:] # Take the points corresponding to the first label (e.g. LV endo for mitral valve)
    # Find extent
    x = [v[0] for v in valveplane]
    y = [v[1] for v in valveplane]
    center = [np.mean(x), np.mean(y)]
    distances = [np.sqrt((v[0] - center[0])**2 + (v[1] - center[1])**2) for v in valveplane]
    valveplane = valveplane[np.argsort(distances)]
    radius = np.max(distances)
    
    max_dist = radius
    for i in range(len(valveplane)-len(valveplane//2), len(valveplane)):
        for j in range(i+1, len(valveplane)):
            dist = np.sqrt((valveplane[i][0] - valveplane[j][0])**2 + (valveplane[i][1] - valveplane[j][1])**2)
            if dist > radius:
                if dist > max_dist:
                    max_dist = dist
                    valvepts = np.array([valveplane[i], valveplane[j]], dtype=np.int64)

    return valvepts


def estimate_lva(epipts, mv1, mv2):
    mv_centroid = [np.mean([mv1[0], mv2[0]]), np.mean([mv1[1], mv2[1]])]
    distances = [np.sqrt((p[0] - mv_centroid[0])**2 + (p[1] - mv_centroid[1])**2) for p in epipts]
    lvepiapex = epipts[np.argmax(distances)]
    return lvepiapex


def find_contours(seg: np.ndarray, spec: str = "all") -> list[np.ndarray]:
    """Find the contours within the segmentation.

    Args:
        seg (np.ndarray): Segmentation.
        spec (str): Specification of the contours to find.

    Returns:
        contours (list[np.ndarray]): List of contours.
    """
    # Define the retrieval modes.
    retrieval_modes = {
        "all": cv2.RETR_LIST,
        "external": cv2.RETR_EXTERNAL,
        "tree": cv2.RETR_TREE,
    }

    # Find contours based on the specified mode.
    contours, _ = cv2.findContours(
        seg, retrieval_modes.get(spec, cv2.RETR_LIST), cv2.CHAIN_APPROX_NONE
    )

    return contours


def find_coordinates_of_holes(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the coordinates of the holes in a segmentation.

    Args:
        seg (np.ndarray): Segmentation.

    Returns:
        coordinates_holes (tuple[np.ndarray, np.ndarray]): Coordinates of the holes in the segmentation.
    """
    # Set all values larger than 0 to 1.
    seg_same_val = seg.copy()
    seg_same_val[seg_same_val > 0] = 1

    # Find the contours of the structures in full segmentation.
    contours = find_contours(seg_same_val, "external")

    coordinates_holes_x_all, coordinates_holes_y_all = np.array([]), np.array([])

    for contour in contours:
        # Create a mask from the contour.
        mask = cv2.drawContours(np.zeros_like(seg_same_val), [contour], 0, 255, -1)

        # Find the positions of all the zero pixels within the contour.
        coordinates_holes_contour = np.where((mask == 255) & (seg_same_val == 0))

        coordinates_holes_x, coordinates_holes_y = (
            coordinates_holes_contour[0],
            coordinates_holes_contour[1],
        )

        coordinates_holes_x_all = np.append(
            coordinates_holes_x_all, coordinates_holes_x
        )
        coordinates_holes_y_all = np.append(
            coordinates_holes_y_all, coordinates_holes_y
        )

    coordinates_holes = (
        coordinates_holes_x_all.astype("int64"),
        coordinates_holes_y_all.astype("int64"),
    )

    return coordinates_holes


def fill_holes_within_structure(seg: np.ndarray, label: int) -> np.ndarray:
    """Fill the holes within a structure.

    Args:
        seg (np.ndarray): Segmentation of structures.
        label (int): Label of structure.

    Returns:
        seg_no_holes (np.ndarray): Segmentation of structure with filled holes.
    """
    # Find coordinates of holes within structure.
    coordinates_holes = find_coordinates_of_holes(seg)

    seg_no_holes = seg.copy()

    # Fill the holes with in the structure.
    for row, col in zip(coordinates_holes[0], coordinates_holes[1]):
        seg_no_holes[row, col] = label

    return seg_no_holes


def qc_segmentation(segmentation, view):
    label_map_view = LABEL_MAP[view]
    
    # First operation: Apply hole filling within each label
    cleaned_segmentation = segmentation.copy()
    filled_structures = {}
    for name, structure in label_map_view.items():
        struct_mask = (segmentation == structure).astype(np.uint8)
        cleaned_struct_mask = fill_holes_within_structure(struct_mask, 1)
        filled_structures[name] = cleaned_struct_mask
        # cleaned_segmentation[cleaned_struct_mask == 1] = structure
    structure_priority = ["LV_endo", "LV_myo", "RV_endo", "RV_myo", "LA_endo", "RA_endo", "Aorta", "PA"]
    for name in reversed(structure_priority):
        if name in filled_structures.keys():
            cleaned_segmentation[filled_structures[name] == 1] = label_map_view[name]

    # Combine all labels, fill holes between, contour, and remove any stray labels not part of largest contour
    all_structures = np.zeros_like(segmentation).astype(np.uint8)
    for name, label in label_map_view.items():
        struct_mask = (cleaned_segmentation == label).astype(np.uint8)
        all_structures_mask = (all_structures > 0).astype(np.uint8)
        all_structures = (all_structures_mask | struct_mask).astype(np.uint8)
    
    cleaned_all_structures = fill_holes_within_structure(all_structures, 1)

    contours = find_contours(cleaned_all_structures, spec="external")
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        mask_largest = np.zeros_like(cleaned_all_structures)

        mask_largest = cv2.drawContours(mask_largest.copy(), [largest_contour], 0, 1, -1)

        # Remove any labels not in largest contour
        for name, label in label_map_view.items():
            cleaned_segmentation[(cleaned_segmentation == label) & (mask_largest == 0)] = 0

    if "RV_myo" in label_map_view.keys():
        # Remove any other labels between RV endo and RV myo
        rv_endo_mask = (cleaned_segmentation == label_map_view["RV_endo"]).astype(np.uint8)
        rv_myo_mask = (cleaned_segmentation == label_map_view["RV_myo"]).astype(np.uint8)
        rv_epi = (rv_endo_mask | rv_myo_mask).astype(np.uint8)

        cleaned_rv_epi = fill_holes_within_structure(rv_epi,1)
        cleaned_segmentation[cleaned_rv_epi == 1] = label_map_view["RV_endo"]
        cleaned_segmentation[rv_myo_mask == 1] = label_map_view["RV_myo"]

        # Remove RV myo that is not adjacent to RV endo
        rv_endo_mask = (cleaned_segmentation == label_map_view["RV_endo"]).astype(np.uint8)
        rv_myo_mask = (cleaned_segmentation == label_map_view["RV_myo"]).astype(np.uint8)
        rv_epi = (rv_endo_mask | rv_myo_mask).astype(np.uint8)
        
        epi_contours = find_contours(cleaned_rv_epi, spec="external")
        # Leave only the largest contour
        if len(epi_contours) > 1:
            largest_contour = max(epi_contours, key=cv2.contourArea)
            second_largest_contour = sorted(epi_contours, key=cv2.contourArea)[-2]

            if (cv2.contourArea(largest_contour) - cv2.contourArea(second_largest_contour)) < 0.25 * cv2.contourArea(largest_contour):
                # If the second largest contour is within 25% of the largest contour, compare and choose the one that is closer to the LV endo centroid
                lv_endo_mask = (cleaned_segmentation == label_map_view["LV_endo"]).astype(np.uint8)
                # Get 'center' of LV endo via average of segmentation coords
                lv_endo_coords = np.column_stack(np.where(lv_endo_mask == 1))
                lv_endo_centroid = np.mean(lv_endo_coords, axis=0)

                # Get center of largest contour
                largest_contour_center = np.mean(largest_contour[:,0,:], axis=0)
                second_largest_contour_center = np.mean(second_largest_contour[:,0,:], axis=0)
                dist_largest = np.sqrt((largest_contour_center[0] - lv_endo_centroid[0])**2 + (largest_contour_center[1] - lv_endo_centroid[1])**2)
                dist_second_largest = np.sqrt((second_largest_contour_center[0] - lv_endo_centroid[0])**2 + (second_largest_contour_center[1] - lv_endo_centroid[1])**2)
                if dist_second_largest < dist_largest:
                    largest_contour = second_largest_contour # If second largest is closer to LV endo, choose it instead

            mask_largest = np.zeros_like(cleaned_rv_epi)
            
            mask_largest = cv2.drawContours(mask_largest.copy(), [largest_contour], 0, 1, -1)

            # Remove any rv myo or rv endo not in largest contour
            cleaned_segmentation[(rv_endo_mask == 1) & (mask_largest == 0)] = 0
            cleaned_segmentation[(rv_myo_mask == 1) & (mask_largest == 0)] = 0

    if "LV_endo" in label_map_view.keys() and "LV_myo" in label_map_view.keys():
         # Remove any other labels between LV endo and LV myo
        lv_endo_mask = (cleaned_segmentation == label_map_view["LV_endo"]).astype(np.uint8)
        lv_myo_mask = (cleaned_segmentation == label_map_view["LV_myo"]).astype(np.uint8)
        lv_epi = (lv_endo_mask | lv_myo_mask).astype(np.uint8)

        cleaned_lv_epi = fill_holes_within_structure(lv_epi,1)
        cleaned_segmentation[cleaned_lv_epi == 1] = label_map_view["LV_endo"]
        cleaned_segmentation[lv_myo_mask == 1] = label_map_view["LV_myo"]

        # Remove LV myo that is not adjacent to LV endo
        lv_endo_mask = (cleaned_segmentation == label_map_view["LV_endo"]).astype(np.uint8)
        lv_myo_mask = (cleaned_segmentation == label_map_view["LV_myo"]).astype(np.uint8)
        lv_epi = (lv_endo_mask | lv_myo_mask).astype(np.uint8)
        epi_contours = find_contours(cleaned_lv_epi, spec="external")
        
        # Leave only the largest contour
        if len(epi_contours) > 1:
            largest_contour = max(epi_contours, key=cv2.contourArea)
            mask_largest = np.zeros_like(cleaned_lv_epi)
            
            mask_largest = cv2.drawContours(mask_largest.copy(), [largest_contour], 0, 1, -1)

            # Remove any lv myo or lv endo not in largest contour
            cleaned_segmentation[(lv_endo_mask == 1) & (mask_largest == 0)] = 0
            cleaned_segmentation[(lv_myo_mask == 1) & (mask_largest == 0)] = 0

        if view == "SAX":
            # If RV endo label is intruding into LV, remove RV endo
            lv_endo_mask = (cleaned_segmentation == label_map_view["LV_endo"]).astype(np.uint8)
            lv_myo_mask = (cleaned_segmentation == label_map_view["LV_myo"]).astype(np.uint8)
            rv_endo_mask = (cleaned_segmentation == label_map_view["RV_endo"]).astype(np.uint8)
            rv_myo_mask = (cleaned_segmentation == label_map_view["RV_myo"]).astype(np.uint8)

            combined_lv_rv_mask = (lv_endo_mask | rv_endo_mask).astype(np.uint8)

            rvlv_contours = find_contours(combined_lv_rv_mask, spec="external")
            if len(rvlv_contours) == 1:
                # Assume segmentation has failed, remove RV labels
                cleaned_segmentation[rv_endo_mask == 1] = 0
                cleaned_segmentation[rv_myo_mask == 1] = 0

    # Keep only the largest contour for each structure (except the LV and RV myo)
    for structure in ["LV_endo", "RV_endo", "LA_endo", "RA_endo", "Aorta", "PA"]:
        if structure not in label_map_view.keys():
            continue

        struct_mask = (cleaned_segmentation == label_map_view[structure]).astype(np.uint8)
        struct_contours = find_contours(struct_mask, spec="external")
        if len(struct_contours) > 1:
            largest_contour = max(struct_contours, key=cv2.contourArea)
            mask_largest = np.zeros_like(struct_mask)

            mask_largest = cv2.drawContours(mask_largest.copy(), [largest_contour], 0, 1, -1)

            # Remove any structure labels not in largest contour
            cleaned_segmentation[(struct_mask == 1) & (mask_largest == 0)] = 0

    return cleaned_segmentation


def contour_SAX(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    RV_endo = (segmentation == 3).astype(np.uint8)
    RV_myo = (segmentation == 4).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)

    # convert to contours
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        LV_myo_pts = []
        for c in contours:
            LV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        LV_myo_pts = np.array(LV_myo_pts, dtype=np.int64)
    else:
        LV_myo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    # Get intersection points between RV endo and LV epi to separate septal wall from free wall
    if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                    dtype=np.int64)
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,1])], 
                                    dtype=np.int64)
        else:
            RV_septal_pts = []
            RV_fw_pts = RV_endo_pts
    else:
        RV_septal_pts = []
        RV_fw_pts = RV_endo_pts
    

    # Get intersection points between RV epi and RV endo to remove extraneous RV epi points
    if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=8)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]
        
        
    # Get intersection points between RV epi and RV myo to keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Use RV epi to reassign any septal points that are actually free wall points
    if len(RV_epi_pts)>0 and len(RV_septal_pts)>0:
        pairs = get_intersections(RV_septal_pts, RV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            if len(RV_fw_pts) == 0:
                RV_fw_pts = RV_septal_pts[np.unique(pairs[:,0])]
            else:
                RV_fw_pts = np.vstack([RV_fw_pts, RV_septal_pts[np.unique(pairs[:,0])]])
            RV_septal_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_septal_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)

    # Get intersection points between LV epi and LV myo to keep only myocardial points
    if len(LV_epi_pts)>0 and len(LV_myo_pts)>0:
        pairs = get_intersections(LV_epi_pts, LV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = LV_epi_pts[np.unique(pairs[:,0])]

    # Remove intersection between RV epi and LV epi
    if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # # Compare centroids of LV endo and LV myo, if they are too far apart, segmentation is likely broken, so exclude both sets of points
    # if len(LV_endo_pts) > 0 and len(LV_myo_pts) > 0:
    #     lv_endo_centroid = np.mean(LV_endo_pts, axis=0)
    #     lv_myo_centroid = np.mean(LV_myo_pts, axis=0)

    #     if np.linalg.norm(lv_endo_centroid - lv_myo_centroid) > 10:
    #         LV_endo_pts = []
    #         LV_epi_pts = []
            
    # If there are no lv endo points, remove the lv epi points
    if len(LV_endo_pts) == 0:
        LV_epi_pts = []

    # If there are no rv endo points, remove the rv epi points
    if len(RV_endo_pts) == 0:
        RV_epi_pts = []
                
    return [LV_endo_pts, LV_epi_pts, RV_septal_pts, RV_fw_pts, RV_epi_pts]


def contour_RVOT(segmentation):
    RV_endo = (segmentation == 1).astype(np.uint8)
    RV_myo = (segmentation == 2).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)
    pa = (segmentation == 3).astype(np.uint8)

    # convert to contours
    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(pa, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        pa_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        pa_pts = []

    # Get intersection points between RV endo and pa to clean RV endo pts
    if len(RV_endo_pts)>0 and len(pa_pts)>0:
        pairs = get_intersections(RV_endo_pts, pa_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between RV epi and pa to clean RV epi pts
    if len(RV_epi_pts)>0 and len(pa_pts)>0:
        pairs = get_intersections(RV_epi_pts, pa_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)

            
    # Get intersection points between RV myo and RV endo to separate free wall from septal wall
    if len(RV_endo_pts)>0 and len(RV_myo_pts)>0:

        pairs = get_intersections(RV_endo_pts, RV_myo_pts, distance_cutoff = 5) # Relatively large cutoff because of tendency for RV myo to break

        if len(pairs) > 0:
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i in np.unique(pairs[:,0])], 
                                    dtype=np.int64)
            RV_s_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])],
                                    dtype=np.int64)
            
        else:
            RV_fw_pts = RV_endo_pts
            RV_s_pts = []

    else:
        RV_fw_pts = RV_endo_pts
        RV_s_pts = []
            
    # Get intersection points between RV epi and RV endo to remove extraneous RV epi points
    if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=8)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between RV epi and RV myo pts keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    return [RV_s_pts, RV_fw_pts, RV_epi_pts, pa_pts]


def contour_2ch(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    la = (segmentation == 3).astype(np.uint8)

    # convert to contours
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        LV_myo_pts = []
        for c in contours:
            LV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        LV_myo_pts = np.array(LV_myo_pts, dtype=np.int64)
    else:
        LV_myo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(la, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        la_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        la_pts = []

    # Get intersection points between LV endo and LA to clean LV endo pts 
    if len(LV_endo_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_endo_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            cleaned_la_pts = np.array([pnt.tolist() for i, pnt in enumerate(la_pts) if i not in np.unique(pairs[:,1])], 
                                dtype=np.int64)
            
    # Get intersection points between LV epi and LA to clean LV epi pts
    if len(LV_epi_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_epi_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
    
    # Get intersection points between LV epi and LV myo to keep only myocardial points
    if len(LV_epi_pts)>0 and len(LV_myo_pts)>0:
        pairs = get_intersections(LV_epi_pts, LV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = LV_epi_pts[np.unique(pairs[:,0])]
            
    la_pts = cleaned_la_pts if 'cleaned_la_pts' in locals() else la_pts
            
    return [LV_endo_pts, LV_epi_pts, la_pts]


def contour_3ch(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    RV_endo = (segmentation == 3).astype(np.uint8)
    la = (segmentation == 4).astype(np.uint8)
    aorta = (segmentation == 5).astype(np.uint8)
    RV_myo = (segmentation == 6).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)

    # convert to contours
    # left ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_endo_pts = [] 

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        LV_myo_pts = []
        for c in contours:
            LV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        LV_myo_pts = np.array(LV_myo_pts, dtype=np.int64)
    else:
        LV_myo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_epi_pts = []

    # right ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    # la
    contours, hierarchy = cv2.findContours(cv2.inRange(la, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        la_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        la_pts = []
    
    # aorta
    contours, hierarchy = cv2.findContours(cv2.inRange(aorta, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        aorta_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        aorta_pts = []

    # Get intersection points between RV endo and LV epi to separate septal wall from free wall
    if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:

        pairs = get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=1.5) 

        if len(pairs) > 0:
            RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,1])], 
                                dtype=np.int64)
        else:
            RV_septal_pts = []
            RV_fw_pts = RV_endo_pts
    else:
        RV_septal_pts = []
        RV_fw_pts = RV_endo_pts
            
    # Get intersection points between RV epi and RV endo to remove extraneous RV epi points
    if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=8)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between LV epi and LV myo to keep only myocardial points
    if len(LV_epi_pts)>0 and len(LV_myo_pts)>0:
        pairs = get_intersections(LV_epi_pts, LV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = LV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between LV endo and la to clean LV endo pts
    if len(LV_endo_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_endo_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            cleaned_la_pts = np.array([pnt.tolist() for i, pnt in enumerate(la_pts) if i not in np.unique(pairs[:,1])],
                                dtype=np.int64)
            
    # Get intersection points between LV epi and la to clean LV epi pts
    if len(LV_epi_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_epi_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    la_pts = cleaned_la_pts if 'cleaned_la_pts' in locals() else la_pts
    
    # Get intersection points between LV endo and aorta to clean LV endo pts
    if len(LV_endo_pts)>0 and len(aorta_pts)>0:
        pairs = get_intersections(LV_endo_pts, aorta_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between LV epi and aorta to clean LV epi pts
    if len(LV_epi_pts)>0 and len(aorta_pts)>0:
        pairs = get_intersections(LV_epi_pts, aorta_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    # Get intersection points between RV epi and RV myo to keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Use RV epi to reassign any septal points that are actually free wall points
    if len(RV_epi_pts)>0 and len(RV_septal_pts)>0:
        pairs = get_intersections(RV_septal_pts, RV_epi_pts, distance_cutoff=2)

        if len(pairs) > 0:
            if len(RV_fw_pts) == 0:
                RV_fw_pts = RV_septal_pts[np.unique(pairs[:,0])]
            else:
                RV_fw_pts = np.vstack([RV_fw_pts, RV_septal_pts[np.unique(pairs[:,0])]])

            RV_septal_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_septal_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)

    # Remove intersection between RV epi and LV epi
    if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    return [LV_endo_pts, LV_epi_pts, RV_septal_pts, RV_fw_pts, la_pts, aorta_pts, RV_epi_pts]


def contour_4ch(segmentation):
    # extract points
    LV_endo = (segmentation == 1).astype(np.uint8)
    LV_myo = (segmentation == 2).astype(np.uint8)
    LV_epi = (LV_endo | LV_myo).astype(np.uint8)
    RV_endo = (segmentation == 3).astype(np.uint8)
    la = (segmentation == 4).astype(np.uint8)
    ra = (segmentation == 5).astype(np.uint8)
    RV_myo = (segmentation == 6).astype(np.uint8)
    RV_epi = (RV_endo | RV_myo).astype(np.uint8)

    # convert to contours
    # left ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(LV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        LV_myo_pts = []
        for c in contours:
            LV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        LV_myo_pts = np.array(LV_myo_pts, dtype=np.int64)
    else:
        LV_myo_pts = []  

    contours, hierarchy = cv2.findContours(cv2.inRange(LV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        LV_epi_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        LV_epi_pts = []

    # right ventricle
    contours, hierarchy = cv2.findContours(cv2.inRange(RV_endo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        RV_endo_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        RV_endo_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_epi, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_epi_pts = []
        for c in contours:
            RV_epi_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_epi_pts = np.array(RV_epi_pts, dtype=np.int64)
    else:
        RV_epi_pts = []

    contours, hierarchy = cv2.findContours(cv2.inRange(RV_myo, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        RV_myo_pts = []
        for c in contours:
            RV_myo_pts += [x.tolist() for i,x in enumerate(c[:, 0, :])]
        RV_myo_pts = np.array(RV_myo_pts, dtype=np.int64)
    else:
        RV_myo_pts = []

    # la
    contours, hierarchy = cv2.findContours(cv2.inRange(la, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        la_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        la_pts = []

    # ra
    contours, hierarchy = cv2.findContours(cv2.inRange(ra, 1, 1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ra_pts = np.array([x.tolist() for i,x in enumerate(c[:, 0, :])], dtype=np.int64)
    else:
        ra_pts = []

    # Get intersection points between RV endo and LV epi to separate septal wall from free wall
    if len(RV_endo_pts)>0 and len(LV_epi_pts)>0:

        pairs = get_intersections(RV_endo_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_septal_pts = RV_endo_pts[np.unique(pairs[:,0])] # deletes intersection from RV endo pts
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,1])], 
                                dtype=np.int64)
            
        else:
            RV_septal_pts = []
            RV_fw_pts = RV_endo_pts
    else:
        RV_septal_pts = []
        RV_fw_pts = RV_endo_pts
            
    # Get intersection points between RV epi and RV fw to remove extraneous RV epi points
    if len(RV_epi_pts)>0 and len(RV_fw_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_fw_pts, distance_cutoff=8)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between LV epi and LV myo to keep only myocardial points
    if len(LV_epi_pts)>0 and len(LV_myo_pts)>0:
        pairs = get_intersections(LV_epi_pts, LV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = LV_epi_pts[np.unique(pairs[:,0])]

    # Get intersection points between LV endo and la to clean LV endo pts
    if len(LV_endo_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_endo_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_endo_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_endo_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            cleaned_la_pts = np.array([pnt.tolist() for i, pnt in enumerate(la_pts) if i not in np.unique(pairs[:,1])],
                                dtype=np.int64)
            
    # Get intersection points between LV epi and la to clean LV epi pts
    if len(LV_epi_pts)>0 and len(la_pts)>0:
        pairs = get_intersections(LV_epi_pts, la_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            LV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(LV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    la_pts = cleaned_la_pts if 'cleaned_la_pts' in locals() else la_pts 
    
    # Get intersection points between RV fw and ra to clean RV fw pts
    if len(RV_fw_pts)>0 and len(ra_pts)>0:
        pairs = get_intersections(RV_fw_pts, ra_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_fw_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_fw_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            ra_pts = np.array([pnt.tolist() for i, pnt in enumerate(ra_pts) if i not in np.unique(pairs[:,1])],
                                dtype=np.int64)
            
    # Get intersection points between RV epi and RV myo to keep only free wall points
    if len(RV_epi_pts)>0 and len(RV_myo_pts)>0:
        pairs = get_intersections(RV_epi_pts, RV_myo_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = RV_epi_pts[np.unique(pairs[:,0])]

    # Use RV epi to reassign any septal points that are actually free wall points
    if len(RV_epi_pts)>0 and len(RV_septal_pts)>0:
        pairs = get_intersections(RV_septal_pts, RV_epi_pts, distance_cutoff=2)

        if len(pairs) > 0:
            if len(RV_fw_pts) == 0:
                RV_fw_pts = RV_septal_pts[np.unique(pairs[:,0])]
            else:
                RV_fw_pts = np.vstack([RV_fw_pts, RV_septal_pts[np.unique(pairs[:,0])]])
                
            RV_septal_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_septal_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)

    # Remove intersection between RV epi and LV epi
    if len(RV_epi_pts)>0 and len(LV_epi_pts)>0:
        pairs = get_intersections(RV_epi_pts, LV_epi_pts, distance_cutoff=1.5)

        if len(pairs) > 0:
            RV_epi_pts = np.array([pnt.tolist() for i, pnt in enumerate(RV_epi_pts) if i not in np.unique(pairs[:,0])], 
                                dtype=np.int64)
            
    return [LV_endo_pts, LV_epi_pts, RV_septal_pts, RV_fw_pts, la_pts, ra_pts, RV_epi_pts]