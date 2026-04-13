#!/bin/bash -l
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks=60
#SBATCH --job-name=batch_AMC
#SBATCH --time=5:00:00 # Budget ~1hr per trial
#SBATCH --output=./logs/%j.out 
#SBATCH --error=./logs/%j.err
#SBATCH --qos=normal
#SBATCH --account=ucb689_peak1
#SBATCH --mail-user=maha7624@colorado.edu

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
    
    if [[ "rename_images" == True ]]; then
        python renumber_images.py "${images_dirpath}"
    fi
    
    cmd=(
        python multi_dataset_builder.py 
        --manual 
        --ignore-missing-observation-ids 
        --extracted-fps=$extracted_fps 
        --final-fps=$final_fps 
        --original-fps=$original_fps 
        --errors-csv-filepath="$errors_csv_filepath" 
        --sam2-start=$sam2_start 
        --errors-obs-id="$errors_obs_id" 
        --annot-filepath="$annot_filepath" 
        --masks-filepath="$masks_filepath"
        --images-dirpath="$images_dirpath"
        )
    if [[ "$auto_clean" == "False" ]]; then
        cmd+=(--no-auto)
    fi
    
    printf 'Running command:\n'
    printf '  %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
    done

# Rclone transfer
if [[ "$rclone_transfer" == True ]]; then
    module load rclone
    for i in "${!errors_obs_ids[@]}"; do
        errors_obs_id=${errors_obs_ids[$i]}
        images_dirpath=${image_dirs[$i]}
        annot_filepath=${annot_filepaths[$i]}
        masks_filepath=${SAM2_masks[$i]}
        sam2_start=${sam2_start_frames[$i]}
    
        export_dir="${root_dir}/exports/${errors_obs_id}"
        latest_run_dir=""
        for d in $(find "$export_dir" -maxdepth 1 -type d -name "run_*" | sort -V -r); do
            if [ -f "$d/${errors_obs_id}/annotations/instances_train.json" ]; then
                latest_run_dir="$d"
                break
            fi
        done
        if [ -z "$latest_run_dir" ]; then
            echo "ERROR: No valid run with instances_train.json in $export_dir"
        else
            echo "${errors_obs_id} from ${latest_run_dir} transferring to ${dest_dir}."
            rclone copy "${latest_run_dir}/${errors_obs_id}" "${remote_name}":"${dest_dir}/${errors_obs_id}"
            echo "${errors_obs_id} transfered to ${dest_dir}."
        fi
        
    done
fi

    
      