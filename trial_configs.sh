#### Global Parameters ####
## Applied to every trial ##

# Root directory of project. Should be same as the root path specified in configuration.py! 
root_dir="/scratch/alpine/maha7624/herbfishCV/2023_FF"

# Path to the CSV with SAM2 errors for this project
errors_csv_filepath="/scratch/alpine/maha7624/herbfishCV/stationary/Stationary Annotation Data - SAM2_errors.csv"

# Specify the rounded FPS at which videos were annotated: e.g., 3, 24, 120, 3. 
original_fps=3

# Specify the FPS at which frames were extracted 
extracted_fps=3

# Desired output FPS (can subsample frames, if desired)
final_fps=3

# Run image re-numbering? Will ensure images always start with 0001.jpg
rename_images=False

# Auto cleaning desired? If set to False, will simply do manual cleaning from errors.csv and COCO conversion
auto_clean=False

#### Batch Parameters  ####
## List a value for every trial you want to run through Automatic Mask Cleaning ##

# Observation or trial ID (should match what's in the SAM2errors.csv if that can be used)
errors_obs_ids=(
  "SR_063023_29_L"
  "SR_063023_29_R"
  "EH_070323_40_L"
  "MM_ER_062623_9_L"
  "MM_ER_062623_9_R"
  "ER_070323_36_L"
  )

# Directory of images to process 
image_dirs=(
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/frames/GZ_SR_063023_29_L"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/frames/GZ_SR_063023_29_R"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/frames/MM_EH_070323_40_L"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/frames/MM_ER_062623_9_L"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/frames/MM_ER_062623_9_R"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/frames/MM_ER_070323_36_L"

    )

# Annotation file location
annot_filepaths=(
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/annotations/LG_GZ_SR_063023_29_L_annotations.npy"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/annotations/LG_GZ_SR_063023_29_R_annotations.npy"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/annotations/LG_EH_070323_40_L_annotations.npy"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/annotations/LG_ ER_062623_9_L_annotations.npy"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/annotations/LG_ ER_062623_9_R_annotations.npy"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/annotations/LG_ER_070323_36_L_annotations.npy"

    )

# SAM2 mask location
SAM2_masks=(
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/masks/LG_SR_063023_29_L_masks.pkl"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/masks/LG_SR_063023_29_R_masks.pkl"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/masks/LG_EH_070323_40_L_masks.pkl"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/masks/LG_ER_062623_9_L_masks.pkl"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/masks/LG_ER_062623_9_R_masks.pkl"
    "/scratch/alpine/maha7624/herbfishCV/2023_FF/masks/LG_ER_070323_36_L_masks.pkl"
    )

# SAM2 start frames: Used to match the annotation point to the correct frame for LabelMe
sam2_start_frames=(
    0
    0
    0
    0
    0
    0
    )
    
#### OPTIONAL ####

## Rclone transfer parameters

# Set to False if rclone transfer is not desired, True if you do want automatic transfer (can take some extra time)
rclone_transfer=False

# Name of your rclone remote to the shared google drive. Check rclone config if unsure.
remote_name="shared_gil"

# Path to the folder that data should be transferred to
dest_dir="Gil Lab/All_Projects/Auto_Tracking/LabelMe/2023FF_data"