import os
import nibabel as nib
import numpy as np

import bivme.preprocessing.dicom.src.contouring as contouring
import bivme.preprocessing.dicom.src.guidepointprocessing as guidepointprocessing
import bivme.preprocessing.dicom.src.utils as utils


class SliceViewer:
    def __init__(self, processed_folder, slice_info_df, view, sliceID, es_phase, num_phases, full_cycle=True, my_logger=None):
        
        self.slice_info_df = slice_info_df
        self.view = view
        self.sliceID = sliceID
        self.es_phase = int(es_phase)
        self.num_phases = num_phases
        if full_cycle:
            self.phases = np.arange(0,self.num_phases, dtype=int)
        else:
            self.phases = [0, self.es_phase]
        self.slice = self.slice_info_df[(self.slice_info_df['View'] == self.view) & (self.slice_info_df['Slice ID'] == self.sliceID)]
        self.segmentation_folder = os.path.join(processed_folder, 'segmentations')
        self.segmentations = self.get_segmentations()
        self.segmentations = self.qc_segmentations()  # QC segmentations
        self.get_contours()
        self.landmarks = None
        self.my_logger = my_logger
        
    def get_segmentations(self):
        segmentations = []

        segmentation = os.path.join(self.segmentation_folder, self.view, 'uncropped', f"{self.view}_3d_{self.sliceID}.nii.gz")
        segmentation = nib.load(segmentation).get_fdata()

        for i in range(0,segmentation.shape[2]):
            segmentations.append(segmentation[:,:,i])

        self.size = segmentations[0].shape
        
        return segmentations
    
    def qc_segmentations(self):
        corrected_segmentations = []
        new_segmentation = np.zeros((self.size[0], self.size[1], len(self.phases)))

        for i in range(len(self.phases)):
            seg = self.segmentations[i]
            corrected_seg = contouring.qc_segmentation(seg, self.view)
            corrected_segmentations.append(corrected_seg)
            new_segmentation[:,:,i] = corrected_seg

        new_segmentation = new_segmentation.astype(np.uint8)

        # new_segmentation = np.transpose(new_segmentation, (1, 0, 2)) # TODO: Check if transpose is needed

        # Save corrected segmentations to nii file
        if not os.path.exists(os.path.join(self.segmentation_folder, self.view, 'uncropped-corrected')):
            os.makedirs(os.path.join(self.segmentation_folder, self.view, 'uncropped-corrected'), exist_ok=True)

        segmentation_path = os.path.join(self.segmentation_folder, self.view, 'uncropped-corrected', f"{self.view}_3d_{self.sliceID}.nii.gz")
        nib.save(nib.Nifti1Image(new_segmentation.astype(np.uint8), affine=np.eye(4)), segmentation_path)

        # Update segmentations
        return corrected_segmentations
    
    def get_initial_landmarks(self):
        self.get_landmarks_from_intersections()

        self.landmarks = {
            'SAX': {'RVI': {}},
            '2ch': {'MV': {}, 'LVA': {}}, 
            '3ch': {'MV': {}, 'AV': {}, 'LVA': {}},
            '4ch': {'MV': {}, 'TV': {}, 'LVA': {}},
            'RVOT': {'PV': {}}
        }

        if self.view == 'SAX':
            for phase in self.phases:
                self.landmarks['SAX']['RVI'][f'{phase}'] = self.rvi[self.view][f'{phase}']

        elif self.view == '2ch':
            for phase in self.phases:
                self.landmarks['2ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['2ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']
        
        elif self.view == '3ch':
            for phase in self.phases:
                self.landmarks['3ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['3ch']['AV'][f'{phase}'] = self.av[self.view][f'{phase}']
                self.landmarks['3ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']

        elif self.view == '4ch':
            for phase in self.phases:
                self.landmarks['4ch']['MV'][f'{phase}'] = self.mv[self.view][f'{phase}']
                self.landmarks['4ch']['TV'][f'{phase}'] = self.tv[self.view][f'{phase}']
                self.landmarks['4ch']['LVA'][f'{phase}'] = self.lva[self.view][f'{phase}']

        elif self.view == 'RVOT':
            for phase in self.phases:
                self.landmarks['RVOT']['PV'][f'{phase}'] = self.pv[self.view][f'{phase}']

        return self.landmarks

    def get_landmarks_from_intersections(self):

        self.rvi = {'SAX': {}}
        self.mv = {'2ch': {}, '3ch': {}, '4ch': {}}
        self.av = {'3ch': {}}
        self.tv = {'4ch': {}}
        self.lva = {'2ch': {}, '3ch': {}, '4ch': {}}
        self.pv = {'RVOT': {}}

        if self.view == 'SAX':
            for i, phase in enumerate(self.phases):
                try:
                    self.rvi['SAX'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 2, 3, distance_cutoff=1)
                except:
                    self.rvi['SAX'][f'{phase}'] = None

        elif self.view == '2ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['2ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 3, distance_cutoff=1)
                except:
                    self.mv['2ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Mitral valve not found on 2ch slice {self.sliceID} phase {phase}')

                try:
                    self.lva['2ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['2ch'][f'{phase}'][0],  self.mv['2ch'][f'{phase}'][1])
                except:
                    self.lva['2ch'][f'{phase}'] = None
                    self.my_logger.warning(f'LVA not found on 2ch slice {self.sliceID} phase {phase}')
        
        elif self.view == '3ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['3ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 4, distance_cutoff=1)
                except:
                    self.mv['3ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Mitral valve not found on 3ch slice {self.sliceID} phase {phase}')
                
                try:
                    self.av['3ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 5, distance_cutoff=1)
                except:
                    self.av['3ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Aortic valve not found on 3ch slice {self.sliceID} phase {phase}')

                try:
                    self.lva['3ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['3ch'][f'{phase}'][0],  self.mv['3ch'][f'{phase}'][1])
                except:
                    self.lva['3ch'][f'{phase}'] = None
                    self.my_logger.warning(f'LVA not found on 3ch slice {self.sliceID} phase {phase}')

        elif self.view == '4ch':
            for i, phase in enumerate(self.phases):
                try:
                    self.mv['4ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 4, distance_cutoff=1)
                except:
                    self.mv['4ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Mitral valve not found on 4ch slice {self.sliceID} phase {phase}')
                
                try:
                    self.tv['4ch'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 3, 5, distance_cutoff=1)
                except:
                    self.tv['4ch'][f'{phase}'] = None
                    self.my_logger.warning(f'Tricuspid valve not found on 4ch slice {self.sliceID} phase {phase}')

                try:
                    self.lva['4ch'][f'{phase}'] = contouring.estimate_lva(self.contours[f'{phase}'][1], self.mv['4ch'][f'{phase}'][0],  self.mv['4ch'][f'{phase}'][1])
                except:
                    self.lva['4ch'][f'{phase}'] = None
                    self.my_logger.warning(f'LVA not found on 4ch slice {self.sliceID} phase {phase}')

        elif self.view == 'RVOT':
            for i, phase in enumerate(self.phases):
                try:
                    self.pv['RVOT'][f'{phase}'] = contouring.get_valve_points_from_intersections(self.segmentations[i], 1, 3, distance_cutoff=1)
                except:
                    self.pv['RVOT'][f'{phase}'] = None
                    self.my_logger.warning(f'Pulmonary valve not found on RVOT slice {self.sliceID} phase {phase}')
    
        else:
            self.my_logger.warning(f'Invalid view: {self.view}. Must be one of SAX, 2ch, 3ch, 4ch, or RVOT.')
        
    def get_contours(self):
        ## TODO: QC 
        contours = {}

        if self.view == 'SAX':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_SAX(self.segmentations[i])
        elif self.view == 'RVOT':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_RVOT(self.segmentations[i])
        elif self.view == '2ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_2ch(self.segmentations[i])
        elif self.view == '3ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_3ch(self.segmentations[i])
        elif self.view == '4ch':
            for i, phase in enumerate(self.phases):
                contours[f'{phase}'] = contouring.contour_4ch(self.segmentations[i])
        
        self.contours = contours

    def get_slice_info(self):
        self.imgPos = self.slice['ImagePositionPatient'].values[0]
        self.imgOrient = self.slice['ImageOrientationPatient'].values[0]
        self.ps = self.slice['Pixel Spacing'].values[0]

    def smooth_landmarks(self, landmarks):
        # Smooth landmarks by applying an low pass filter
        # Firstly, check if any of the landmarks are flipped across the cycle
        if np.any([x is None for x in landmarks]):
            return landmarks # No smoothing if there are missing landmarks
        
        l1 = np.array([x[0] for x in landmarks])
        l2 = np.array([x[1] for x in landmarks])

        true_l1 = []
        true_l2 = []

        # # Take the ED orientation as the true orientation
        l1_ed = l1[0]
        l2_ed = l2[0]

        for m, n in zip(l1, l2):
            if np.all(m == l1_ed) and np.all(n == l2_ed):
                # If the landmark is the same as the ED landmark, keep it
                true_l1.append(m)
                true_l2.append(n)

            else:
                # Check if the landmark is flipped
                if np.linalg.norm(m - previous_l1) > np.linalg.norm(n - previous_l1) or np.linalg.norm(n - previous_l2) > np.linalg.norm(m - previous_l2):
                    # If it is flipped, correct the orientation
                    true_l1.append(n)
                    true_l2.append(m)
                else:
                    true_l1.append(m)
                    true_l2.append(n)

            previous_l1 = true_l1[-1]
            previous_l2 = true_l2[-1]
        
        true_l1 = np.array(true_l1)
        true_l2 = np.array(true_l2)

        # Smooth the landmarks using a low pass filter
        harmonic_divisor = 4 # Low pass filter will keep num_phases/harmonic_divisor harmonics. E.g. if you have 30 frames and harmonic_divisor=4, it will keep the first 7 harmonics.

        l1_x = np.array([x[0] for x in true_l1])
        l1_y = np.array([x[1] for x in true_l1])
        l2_x = np.array([x[0] for x in true_l2])
        l2_y = np.array([x[1] for x in true_l2])

        l1_x_smooth = utils.apply_fft(l1_x, harmonic_divisor=harmonic_divisor)
        l1_y_smooth = utils.apply_fft(l1_y, harmonic_divisor=harmonic_divisor)
        l2_x_smooth = utils.apply_fft(l2_x, harmonic_divisor=harmonic_divisor)
        l2_y_smooth = utils.apply_fft(l2_y, harmonic_divisor=harmonic_divisor)

        l1_smooth = np.array([l1_x_smooth, l1_y_smooth]).T
        l2_smooth = np.array([l2_x_smooth, l2_y_smooth]).T

        # Reconstruct the landmarks
        smoothed_landmarks = []
        for i in range(len(l1_smooth)):
            if np.isnan(l1_smooth[i]).any() or np.isnan(l2_smooth[i]).any():
                smoothed_landmarks.append(None)
            else:
                smoothed_landmarks.append(np.array([l1_smooth[i], l2_smooth[i]]))

        return smoothed_landmarks
    
    def smooth_LVA(self, landmarks):
        if np.any([x is None for x in landmarks]):
            return landmarks # No smoothing if there are missing landmarks
        
        # No need to correct for orientation, so just smooth
        harmonic_divisor = 4 # Low pass filter will keep num_phases/harmonic_divisor harmonics. E.g. if you have 30 frames and harmonic_divisor=4, it will keep the first 7 harmonics.

        l_x = np.array([x[0] for x in landmarks])
        l_y = np.array([x[1] for x in landmarks])

        l_x_smooth = utils.apply_fft(l_x, harmonic_divisor=harmonic_divisor)
        l_y_smooth = utils.apply_fft(l_y, harmonic_divisor=harmonic_divisor)

        l_smooth = np.array([l_x_smooth, l_y_smooth]).T

        # Reconstruct the landmarks
        smoothed_landmarks = [l_smooth[i] if not np.isnan(l_smooth[i]).any() else None for i in range(len(l_smooth))]

        return smoothed_landmarks

    def export_slice(self, output_folder, smooth_landmarks):
        self.get_slice_info()
        os.makedirs(output_folder, exist_ok=True)

        if self.view == 'SAX':
            if smooth_landmarks:
                # Get all landmarks
                landmarks = [self.rvi[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the rvi dict
                self.rvi[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.rvi[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][4]

                RVI_pts = self.rvi[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                                RV_epi_pts,
                                RV_septal_pts,
                                LV_epi_pts,
                                LV_endo_pts,
                                RVI_pts]
                
                labels = ['SAX_RV_FREEWALL',
                            'SAX_RV_EPICARDIAL',
                            'SAX_RV_SEPTUM',
                            'SAX_LV_EPICARDIAL',
                            'SAX_LV_ENDOCARDIAL',
                            'RV_INSERT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue
                    
                    if len(points) == 0:
                        continue
                        
                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]

                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.sliceID, weight=1.0, phase=int(phase))

        elif self.view == 'RVOT':
            if smooth_landmarks:
                # Get all landmarks
                landmarks = [self.pv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the pv dict
                self.pv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.pv[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                RV_s_pts = self.contours[phase][0]
                RV_fw_pts = self.contours[phase][1]
                RV_epi_pts = self.contours[phase][2]
                pa_pts = self.contours[phase][3]

                PV_pts = self.pv[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                                RV_s_pts,
                                RV_epi_pts,
                                PV_pts]
                
                # labels = ['SAX_RV_FREEWALL',
                #             'SAX_RV_SEPTUM',
                #             'SAX_RV_EPICARDIAL',
                #             'PULMONARY_VALVE']

                # labels = ['LAX_RV_FREEWALL',
                #             'LAX_RV_SEPTUM',
                #             'LAX_RV_EPICARDIAL',
                #             'PULMONARY_VALVE']

                # OUTLET type contours are oblique to SAX and LAX contours, so we use a different label to distinguish them 
                labels = ['OUTLET_RV_FREEWALL',
                            'OUTLET_RV_SEPTUM',
                            'OUTLET_RV_EPICARDIAL',
                            'PULMONARY_VALVE']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                            for point in points]
                            
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.sliceID, weight=1.0, phase=int(phase))

        elif self.view == '2ch':
            if smooth_landmarks:
                ## Get MV landmarks
                landmarks = [self.mv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the mv dict
                self.mv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.mv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get LVA landmarks
                landmarks = [self.lva[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_LVA(landmarks)

                # Write the smoothed landmarks to the lva dict
                self.lva[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.lva[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                la_pts = self.contours[phase][2]

                MV_pts = self.mv[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [LV_endo_pts, 
                                LV_epi_pts,
                                MV_pts,
                                LVA_pts,
                                la_pts]
                
                labels = ['LAX_LV_ENDOCARDIAL',
                            'LAX_LV_EPICARDIAL',
                            'MITRAL_VALVE',
                            'APEX_POINT',
                            'LAX_LA']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    elif points.size == 2:
                        if points.shape != (2,):
                            points = points[0]

                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]

                    else:
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]
                    
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.sliceID, weight=1.0, phase=int(phase))

        elif self.view == '3ch':
            if smooth_landmarks:
                ## Get MV landmarks
                landmarks = [self.mv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the mv dict
                self.mv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.mv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get AV landmarks
                landmarks = [self.av[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the av dict
                self.av[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.av[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get LVA landmarks
                landmarks = [self.lva[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_LVA(landmarks)

                # Write the smoothed landmarks to the lva dict
                self.lva[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.lva[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][6]

                la_pts = self.contours[phase][4]

                MV_pts = self.mv[self.view][str(phase)]
                AV_pts = self.av[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                               RV_epi_pts,
                               RV_septal_pts,
                               LV_epi_pts,
                               LV_endo_pts,
                               RV_epi_pts,
                               la_pts,
                               MV_pts,
                               AV_pts,
                               LVA_pts]
                
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_RV_SEPTUM',
                            'LAX_LV_EPICARDIAL',
                            'LAX_LV_ENDOCARDIAL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_LA',
                            'MITRAL_VALVE',
                            'AORTA_VALVE',
                            'APEX_POINT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points)== 0:
                        continue

                    elif points.size == 2:
                        if points.shape != (2,):
                            points = points[0]

                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]

                    else:
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]

                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.sliceID, weight=1.0, phase=1.0)

        elif self.view == '4ch':
            if smooth_landmarks:
                ## Get MV landmarks
                landmarks = [self.mv[self.view][str(x)] for x in self.phases]
                
                # Smooth the landmarks
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the mv dict
                self.mv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.mv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get TV landmarks
                landmarks = [self.tv[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_landmarks(landmarks)

                # Write the smoothed landmarks to the tv dict
                self.tv[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.tv[self.view][str(phase)] = smoothed_landmarks[i]

                ## Get LVA landmarks
                landmarks = [self.lva[self.view][str(x)] for x in self.phases]
                smoothed_landmarks = self.smooth_LVA(landmarks)

                # Write the smoothed landmarks to the lva dict
                self.lva[self.view] = {}
                for i, phase in enumerate(self.phases):
                    self.lva[self.view][str(phase)] = smoothed_landmarks[i]

            for phase in self.phases:
                phase = str(phase)

                LV_endo_pts = self.contours[phase][0]
                LV_epi_pts = self.contours[phase][1]
                RV_septal_pts = self.contours[phase][2]
                RV_fw_pts = self.contours[phase][3]
                RV_epi_pts = self.contours[phase][6]

                la_pts = self.contours[phase][4]
                ra_pts = self.contours[phase][5]

                MV_pts = self.mv[self.view][str(phase)]
                TV_pts = self.tv[self.view][str(phase)]
                LVA_pts = self.lva[self.view][str(phase)]

                point_lists = [RV_fw_pts,
                               RV_epi_pts,
                               RV_septal_pts,
                               LV_epi_pts,
                               LV_endo_pts,
                               RV_epi_pts,
                                la_pts,
                                ra_pts,
                               MV_pts,
                               TV_pts,
                               LVA_pts]
                
                labels = ['LAX_RV_FREEWALL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_RV_SEPTUM',
                            'LAX_LV_EPICARDIAL',
                            'LAX_LV_ENDOCARDIAL',
                            'LAX_RV_EPICARDIAL',
                            'LAX_LA',
                            'LAX_RA',
                            'MITRAL_VALVE',
                            'TRICUSPID_VALVE',
                            'APEX_POINT']

                for i,points in enumerate(point_lists):
                    if points is None:
                        continue

                    if len(points) == 0:
                        continue

                    elif points.size == 2:
                        if points.shape != (2,):
                            points = points[0]

                        pts = [guidepointprocessing.inverse_coordinate_transformation(points, self.imgPos, self.imgOrient, self.ps)]

                    else:
                        pts = [guidepointprocessing.inverse_coordinate_transformation(point, self.imgPos, self.imgOrient, self.ps)
                                for point in points]
                        
                    # Write to file
                    guidepointprocessing.write_to_gp_file(output_folder + f'/GPFile_{int(phase):03}.txt', pts, labels[i], self.sliceID, weight=1.0, phase=int(phase))