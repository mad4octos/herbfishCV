#!/bin/bash -l
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --time=5:00:00 # Budget ~1hr per trial
#SBATCH --output=./logs/%j.out 
#SBATCH --error=./logs/%j.err
#SBATCH --qos=normal
#SBATCH --account=ucb689_peak1
#SBATCH --mail-user="youremail@colorado.edu"

module purge
module load miniforge
mamba activate /projects/maha7624/software/anaconda/envs/herbfishCV

# Read from configs
source trial_configs.sh


# First, check that the correct lengths of essential config values are provided
arrays_to_check=(
  errors_obs_ids
  image_dirs
  annot_filepaths
  SAM2_masks
  sam2_start_frames
)
expected_len=${#errors_obs_ids[@]}

for arr_name in "${arrays_to_check[@]}"; do
    # indirect expansion to get array length dynamically
    arr_len=$(eval "echo \${#${arr_name}[@]}")

    if [[ "$arr_len" -ne "$expected_len" ]]; then
        echo "Length mismatch:"
        echo "  trials     = $expected_len"
        echo "  $arr_name = $arr_len"
        exit 1
    fi
done

for i in "${!errors_obs_ids[@]}"; do
    errors_obs_id=${errors_obs_ids[$i]}
    images_dirpath=${image_dirs[$i]}
    annot_filepath=${annot_filepaths[$i]}
    masks_filepath=${SAM2_masks[$i]}
    sam2_start=${sam2_start_frames[$i]}
    
    echo "======================================"
    echo "Cleaning trial: $errors_obs_id"
    echo "======================================"
    
    python multi_dataset_builder.py --manual --ignore-missing-observation-ids --extracted-fps=$extracted_fps --final-fps=$final_fps \
      --original-fps=$original_fps \
      --errors-csv-filepath="$errors_csv_filepath" --sam2-start=$sam2_start --errors-obs-id="$errors_obs_id" \
      --annot-filepath="$annot_filepath" \
      --masks-filepath="$masks_filepath"\
      --images-dirpath="$images_dirpath"
done
      