#### Global Parameters ####
## Applied to every trial ##

# Path to the CSV with SAM2 errors for this project
errors_csv_filepath="/insert/path/to/SAM2_errors.csv"

# Specify the rounded FPS at which videos were recorded: e.g., 24, 120.
original_fps=24

# Specify the FPS at which frames were extracted 
extracted_fps=3

# Desired output FPS (can subsample frames, if desired)
final_fps=3




#### Batch Parameters  ####
## List a value for every trial you want to run through Automatic Mask Cleaning ##

# Observation or trial ID (should match what's in the SAM2errors.csv if that can be used)
errors_obs_ids=(
  "JGL_042724_site1_east_B_Right_GX110093"
  "trial2_observation_id"
  )

# Directory of images to process 
image_dirs=(
    "/scratch/alpine/maha7624/SAM2.1/stationary/raw_vids/GX110093_frames"
    "path/to/trial2/frames"
    )

# Annotation file location
annot_filepaths=(
    "/scratch/alpine/maha7624/SAM2.1/stationary/annotations/JGL_04-27-2024_site1_east_B_Right_GX110093_annotations1.npy"
    "path/to/trial2/annotations.npy"
    )

# SAM2 mask location
SAM2_masks=(
    "/scratch/alpine/maha7624/SAM2.1/stationary/masks/JGL_042724_site1_east_B_Right_GX110093_masks.pkl"
    "path/to/trial2/masks.pkl"
    )

# SAM2 start frames: Used to match the annotation point to the correct frame for LabelMe
sam2_start_frames=(
    0
    0
    )