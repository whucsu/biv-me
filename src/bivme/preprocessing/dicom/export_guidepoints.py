import os
import glob
import shutil

from bivme.preprocessing.dicom.src.guidepointprocessing import postprocess_guidepoints
from bivme.preprocessing.dicom.src.sliceviewer import SliceViewer

def export_guidepoints(dst, output, slice_dict, smooth_landmarks):
    # check if files in output folder already exist
    if os.path.exists(output):
        existing_files = os.listdir(output)
        for file in existing_files:
            if file.endswith('.txt'):
                os.remove(os.path.join(output, file))
            
    for s in slice_dict.values():
        s.export_slice(output, smooth_landmarks)

    # Copy sliceinfo file to output folder
    shutil.copyfile(os.path.join(dst, 'SliceInfoFile.txt'), os.path.join(output, 'SliceInfoFile.txt'))
    
def apply_guidepoint_postprocessing(output, case_name, my_logger):
    gpfiles = glob.glob(os.path.join(output, 'GPFile*.txt'))

    gpfiles = sorted(gpfiles, key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.txt','')))
    my_logger.info(f'Post-processing guidepoint files...')
    for gpfile in gpfiles:
        frame_num = int(os.path.basename(gpfile).split('_')[-1].replace('.txt',''))
        postprocess_guidepoints(gpfile, os.path.join(output, 'SliceInfoFile.txt'), case_name, frame_num, my_logger)

