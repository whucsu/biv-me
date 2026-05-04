import os
import shutil
import torch
import nibabel as nib
import numpy as np

# Set nnUNet environment variables so it doesn't scream at you with warnings
os.environ['nnUNet_raw'] = '.'
os.environ['nnUNet_preprocessed'] = '.'
os.environ['nnUNet_results'] = '.'

import nnunetv2 as nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from bivme.preprocessing.dicom.src.utils import write_nifti
from bivme.preprocessing.dicom.src.utils import write_sliceinfofile


def init_nnUNetv2(model_folder, my_logger):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")  # nnUNetv2 does not support MPS, so we use CPU instead
        my_logger.warning('MPS is available, but nnU-Net-v2 does not support it. Using CPU instead. This may be very slow!')
    else:
        device = torch.device("cpu")

    predictor = nnUNetPredictor(tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor


def predict_view(input_folder, output_folder, model, view, dataset, my_logger):
    # Define the trained model to use (Specified by the Task)
    model_folder_name = os.path.join(model,"Segmentation/{}/nnUNetTrainer__nnUNetPlans__3d_fullres/".format(dataset))
    
    view_input_folder = os.path.join(input_folder, view, 'resized-cropped-normalised')
    view_output_folder = os.path.join(output_folder, view, 'raw')
        
    if len(os.listdir(view_input_folder)) > 0:
        # Initialize nnUNet model
        predictor = init_nnUNetv2(model_folder_name, my_logger)

        # Make predictions
        predictor.predict_from_files(
            view_input_folder,
            view_output_folder,
            save_probabilities=False, overwrite=True, num_processes_preprocessing=2, 
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0
        )

        my_logger.info(f'Done with {view}')


def reassemble_full_segmentation(input_folder, output_folder, view, slice_info_df, my_logger):
    original_view_input_folder = os.path.join(input_folder, view, 'resized')
    view_output_folder = os.path.join(output_folder, view, 'raw')
    new_view_output_folder = os.path.join(output_folder, view, 'uncropped')

    os.makedirs(new_view_output_folder, exist_ok=True)

    img_seg_dict = {}
    original_images = os.listdir(original_view_input_folder)
    segmentations = [f for f in os.listdir(view_output_folder) if f.endswith('.nii.gz')]

    for img_file in original_images:
        img_id = img_file.split('_')[2]
        # Look for corresponding segmentation file
        seg_file = None
        for seg in segmentations:
            if seg.split('_')[2].replace('.nii.gz','') == img_id:
                seg_file = seg
                break

        img_seg_dict[img_id] = {'img_file': img_file, 'seg_file': seg_file}

    for img_id in img_seg_dict.keys():
        if img_seg_dict[img_id]['seg_file'] is None:
            my_logger.warning(f'No segmentation found for slice {img_id}, skipping...')
            continue

        # Read original image to get dimensions
        img_nii = nib.load(os.path.join(original_view_input_folder, img_seg_dict[img_id]['img_file']))
        img_data = img_nii.get_fdata()
        img_shape = img_data.shape

        # Read segmentation
        seg_nii = nib.load(os.path.join(view_output_folder, img_seg_dict[img_id]['seg_file']))
        seg_data = seg_nii.get_fdata()
        seg_shape = seg_data.shape

        # Create full segmentation array
        full_seg = np.zeros(img_shape, dtype=seg_data.dtype)
        start_x = (img_shape[0] - seg_shape[0]) // 2
        start_y = (img_shape[1] - seg_shape[1]) // 2

        # Pad segmentation back to original size
        full_seg[start_x:start_x+seg_shape[0], start_y:start_y+seg_shape[1], :] = seg_data
        # Save uncropped segmentation
        full_seg_nii = nib.Nifti1Image(full_seg, img_nii.affine, img_nii.header)
        nib.save(full_seg_nii, os.path.join(new_view_output_folder, f'{view}_3d_{img_id}.nii.gz'))


def segment_views(dst, model, slice_info_df, my_logger):
    # define I/O parameters for nnUnet segmentation
    input_folder = os.path.join(dst, 'images')
    output_folder = os.path.join(dst, 'segmentations')

    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    else:
        shutil.rmtree(input_folder)
        os.makedirs(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        
    # nnunet models / tasks
    datasets_3d = ["Dataset260_SAX_3D", "Dataset261_2ch_3D", "Dataset262_3ch_3D", "Dataset263_4ch_3D", "Dataset264_RVOT_3D"]

    views = ['SAX', '2ch', '3ch', '4ch', 'RVOT']

    for i, view in enumerate(views):
        if len(slice_info_df[slice_info_df['View'] == view]) == 0:
            my_logger.info(f'No {view} images found, skipping...')
            continue

        os.makedirs(os.path.join(input_folder, view), exist_ok=True)
        os.makedirs(os.path.join(output_folder, view), exist_ok=True)
        
        my_logger.info(f'Writing {view} images to nifti files...')

        view_rows = slice_info_df[slice_info_df['View'] == view]
        for j, row in view_rows.iterrows():
            slice_id = row['Slice ID']
            pixel_array = row['Img']
            pixel_spacing = row['Pixel Spacing']
            rescale_factor = write_nifti(slice_id, pixel_array, pixel_spacing, input_folder, view)

            if rescale_factor != 1:
                # Update pixel spacing
                idx = slice_info_df.index[slice_info_df['Slice ID'] == slice_id].tolist()[0]
                # Use idx to update the original slice_info_df
                slice_info_df.at[idx, 'Pixel Spacing'] = [pixel_spacing[0]*rescale_factor, pixel_spacing[1]*rescale_factor]

        my_logger.info(f'Segmenting {view} images...')
        
        dataset = datasets_3d[i]

        predict_view(input_folder, output_folder, model, view, dataset, my_logger) # generate segmentations for each view 

        reassemble_full_segmentation(input_folder, output_folder, view, slice_info_df, my_logger) # convert segmentations back to original dimensions by padding (i.e. 'uncropping')

    # Write updated slice info file with new pixel spacings
    write_sliceinfofile(dst, slice_info_df)