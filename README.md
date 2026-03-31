
### Instructions

0) Prepare environment
```
mamba create -n ENV_NAME python=3.11
```

Activate the environment:
```
activate ENV_NAME
```

1) Install Rust, which is needed to compile the CVAT's Datumaro fork:

```
conda install conda-forge::rust
```

2) Install requirements:
```
pip install -r requirements.txt
```


1) Fill the <FILL_ME> placeholders in the `configuration.py` file.
    - `DATA_ROOT_PATH`: the base directory from which all other relative paths are resolved:
        - `DATA_ROOT_PATH / "SAM2_errors.csv"`
        - `DATA_ROOT_PATH / "location_annotations"`
        - `DATA_ROOT_PATH / "SAM2_masks"`
        - `DATA_ROOT_PATH / "exports"`
    - `ClassifierConfig.model_weights_path` the full path to the model's weight file used by the masks classifier.

2) To add a new observation and specify its location on disk, add a new `ParsedObservationID : <path>` entry under `Config.obsId_to_folder_map` in the `configuration.py` file, for example:
    ```
      ParsedObservationID(
          observer="MLM",
          date="04-27-2024",
          site="site1",
          direction="east",
          ab="B",
          side="Right",
          videoname="GX040093",
      ): DATA_ROOT_PATH / "frames" / "GX040093",
    ```
    

3) Run the `multi_dataset_builder.py` file. This will:
    - Verify that all required input data for all observations is available before starting.
    - Run sequentially for all the specified observations.
    - Load the masks, annotations and errors.
    - Iterate through all the frames and masks, rejecting masks using a trained classifier and time series rules.
    - Export the filtered object detection bounding boxes into CVAT XML and Ultralytics YOLO (object-detection) formats. 
    - Export a video for debug, to be found in the `Config.output_path` folder corresponding for the current ObservationID.

    **Optional flags:**

    - `--no-auto` — Disable automatic mask cleaning (blob filters, classifier, anomaly detection). When this flag is passed, only frames manually specified in the errors CSV will have their masks removed; all other masks are accepted as-is, keeping only the largest blob per object.
      ```bash
      python multi_dataset_builder.py --no-auto
      ```

    - `--ignore-missing-observation-ids` — Do not stop execution when an observation ID is missing from `SAM2_errors.csv`.
      ```bash
      python multi_dataset_builder.py --ignore-missing-observation-ids
      ```

    - `--extracted-fps` and `--final-fps` — Control frame subsampling by specifying the FPS at which frames were extracted and the desired output FPS. Both must be provided together. When omitted, all frames are processed.
      ```bash
      # Keep every 3rd frame (30fps extracted, 10fps desired)
      python multi_dataset_builder.py --extracted-fps 30 --final-fps 10

      # Keep every 6th frame (30fps extracted, 5fps desired)
      python multi_dataset_builder.py --extracted-fps 30 --final-fps 5
      ```

    - `--original-fps` and `--sam2-start` — Attach ground-truth annotation data to exported COCO annotations. These flags map extracted frame indices back to the original video frame numbers (as stored in the `.npy` annotations), then look up the closest known annotation location for each object mask. Both must be provided together, and require `--extracted-fps` (and `--final-fps`).
      - `--original-fps`: Native frame rate of the original video (before ffmpeg extraction).
      - `--sam2-start`: Zero-indexed frame number in the original video from which ffmpeg began extracting frames.

      Each exported annotation will include `gt_location`, `gt_obj_id`, `gt_frame_original`, and `gt_frame_extracted` attributes.

      ```bash
      python multi_dataset_builder.py \
          --extracted-fps 3 --final-fps 1 \
          --original-fps 23.997 --sam2-start 100
      ```

**Prefixed image filenames:**

Image files with prefixes (e.g., `left_0001.jpg`, `cam1_0001.jpg`) are supported automatically. When loading a frame, the system first tries an exact filename match (e.g., `0001.jpg`). If not found, it searches for any file ending with the expected filename (e.g., `*0001.jpg`) and uses the first match. Non-prefixed filenames continue to work as before. The expected filename is constructed using the `number_of_zeros` setting in `configuration.py` (e.g., 5 produces `00001.jpg`, 4 produces `0001.jpg`).

### Observation ID Naming Conventions (Stationary Project)

The stationary project uses `ParsedObservationID` to automatically format observation IDs from constituent parts. Below are the naming conventions used across different file types.

**SAM2_Errors.csv format:**
```
<monopod>_<date-recorded>_<site>_<direction>_<A/B>_<Left/Right>_<videoname>
```

**Masks and annotations format:**
```
<observer>_<date>_<site>_<direction>_<A/B>_<Left/Right>_<videoname>_<colorcorrection>
```

> **Note:** Variations exist for masks files, such as:
> - `<observer>_monopod_<date><...>`
> - `<videoname>`
> - `<videoname>_synced`

**Date format variations:**
- `MM-DD-YYYY` (e.g., `05-30-2024`)
- `MMDDYY` (e.g., `053024`)

**Color correction:**
For some videos, color correction methods (LACC or retinex) were applied to the frames. This is specified in the `Enhancement` column of the SAM2_Errors.csv. If color correction was used, the corresponding frames folder will be named `<videoname>_LACC` or `<videoname>_ret`. If no color correction was used, it will be blank in the observation ID.

**Example files:**
```
JGL_monopod_05302024_site5_east_B_Right_GX030843_masks.pkl
JGL_monopod_05302024_site5_east_B_Right_GX030843_annotations.npy
```

### Using ManualObservationID (Focal Follow Project)

The focal follow project uses `ManualObservationID` to specify exact observation ID strings when file naming conventions don't follow the standard `ParsedObservationID` format.

**Parameters:**
- `errors_obs_id`: The observation ID as it appears in the SAM2_errors.csv file
- `masks_filename`: The exact filename of the masks `.pkl` file
- `annotations_filename`: The exact filename of the annotations `.npy` file

**Example usage in `configuration.py`:**
```python
ManualObservationID(
    errors_obs_id="JM_060724_152_playa_largu_scuba_TPScv_L",
    masks_filename="CR_JM_060724_152_playa_largu_scuba_TPScv_L_mask.pkl",
    annotations_filename="CR_JM_060724_152_playa_largu_scuba_TPScv_L_annotations.npy",
): Path("/path/to/images/MH_JM_060724_152_L"),
```

**Example usage via command line:**
```bash
python multi_dataset_builder.py --manual \
    --errors-obs-id="JM_060724_152_playa_largu_scuba_TPScv_L" \
    --errors-csv-filepath="/path/to/SAM2_errors.csv" \
    --masks-filepath="/path/to/CR_JM_060724_152_playa_largu_scuba_TPScv_L_mask.pkl" \
    --annot-filepath="/path/to/CR_JM_060724_152_playa_largu_scuba_TPScv_L_annotations.npy" \
    --images-dirpath "/path/to/JM_060724"
```

### Indexing Conventions

| Data source                  | Indexing |
|------------------------------|----------|
| Annotations (`.npy` file)   | 0-indexed |
| Masks (`.pkl` file)         | 0-indexed |
| Errors CSV                   | 0-indexed |
| Image filenames              | 1-indexed |
| Image IDs in COCO annotations | 1-indexed |

### `configuration.py` explanation

- The `Config` class:
  - Defines the relative paths towards the annotations, masks and errors files.
  - Has the `obsId_to_folder_map` where observations of interest are defined. Each `ParsedObservationID : <path>` pair defines the observation of interest and its location on disk (the ParsedObservationID class is used to input the constituent fields of the ObservationID, and it's used to test different versions of the ObservationID string).
  - Has a `number_of_zeros` field that controls the zero-padding width used when constructing frame filenames (e.g., 5 → `00001.jpg`, 4 → `0001.jpg`). Use 4 for focal follow data and 5 for stationary data.
  - Has an `anomaly_rules` field, where all the anomaly rules acting on the timeseries of blobs properties are defined.

- The `ClassifierConfig` class:
  - Defines in `model_weights_path` the path towards the classifier model used to reject incorrectly masked fish.


### Scripts

##### scripts/extract_crops.py

Extracts blob crops from SAM2 segmentation masks and classifies them as `correct` or `incorrect` based on a manually annotated errors CSV. Crops are saved into subdirectories named after the observation ID. By default crops are RGBA PNGs; pass `--overlay` to save masked overlay images instead.

Note: see the "bulk_extract_crops.py" script description below to run this in bulk!

```bash
python extract_crops.py \
  --images-dirpath "/path/to/frames/MH_JM_060624_146_L" \
  --masks-filepath "/path/to/masks/CR_JM_060624_146_playa_largu_scuba_IPScv_L_mask.pkl" \
  --errors-csv-filepath "/path/to/SAM2_errors_ff_2024 - SAM2_errors.csv" \
  --errors-obs-id "JM_060624_146_playa_largu_scuba_IPScv_L" \
  --output-folder "/path/to/automatic_mask_cleaner_data_overlay" \
  --overlay
```

**Arguments:**

| Argument | Description |
|---|---|
| `--images-dirpath` | Absolute path to the directory containing the image frames. |
| `--masks-filepath` | Absolute path to the masks `.pkl` file. |
| `--errors-csv-filepath` | Absolute path to the errors CSV file. |
| `--errors-obs-id` | Observation ID string exactly as it appears in the errors CSV. |
| `--output-folder` | Root output folder. Crops are saved under `<output-folder>/<obs-id>/correct\|incorrect/`. |
| `--filename-num-zeros` | Zero-padding width for frame filenames (default: `4` for focal follow, use `5` for stationary). |
| `--area-threshold` | Minimum blob area in pixels (default: from `Config`). |
| `--size-threshold` | Minimum blob width/height in pixels (default: from `Config`). |
| `--overlay` | Save masked overlay crops instead of RGBA crops. |

---

##### scripts/bulk_extract_crops.py

Runs `extract_crops.py` in bulk for every `(frames_folder, mask_file)` pair listed in a matched CSV. Edit the constants at the top of the file to point to your data paths and CSV before running.

```bash
python bulk_extract_crops.py
```

**Constants to configure (top of file):**

| Constant | Description |
|---|---|
| `FRAMES_BASE` | Root directory containing all frame folders. |
| `MASKS_BASE` | Root directory containing all mask `.pkl` files. |
| `ERRORS_CSV` | Path to the errors CSV file. |
| `OUTPUT_FOLDER` | Root output folder for all crops. |
| `CSV_PATH` | Path to the matched CSV (`frames_masks_matched_subset.csv`). |

The script derives the `--errors-obs-id` from the mask filename by stripping the annotator prefix (e.g. `CR_`) and the `_mask.pkl` suffix.

---

##### scripts/organize_classifier_dataset.py

Reorganizes cropped fish images (produced by `extract_crops.py` / `bulk_extract_crops.py`) into a `train` / `val` / `test` classifier dataset. Folders are kept intact across splits (no folder is split between train and val), and stereo pairs (L/R folders sharing the same ID) are always assigned together. Train and val pools are stratified by correct-ratio before greedy size-based filling, so both splits span the full easy-to-hard range.

```bash
python organize_classifier_dataset.py \
  /path/to/crops \
  /path/to/output_dataset \
  --test-folders test_folders.txt \
  --train-fraction 0.8
```

**Arguments:**

| Argument | Description |
|---|---|
| `src_dir` | Source directory containing labelled crop folders (`<folder_id>_<L\|R>/correct\|incorrect/`). |
| `target_dir` | Destination directory where `train/`, `val/`, and `test/` splits will be written. |
| `--test-folders` | Text file listing folder names (one per line) to reserve for the test split. Stereo pair integrity is enforced: both L and R must be listed together or neither. |
| `--train-fraction` | Target fraction of total images for the train split (default: `0.8`). |

Output filenames are prefixed with `<folder_id>_<L|R>_` to avoid collisions when images from different folders share the same base name.

---

##### scripts/convert_coco_to_yolo.py

Converts one or more COCO detection datasets into a single Ultralytics YOLO dataset. Input folders and their target splits are specified via a CSV file (columns `dir_path`, `split`, and `observation_id`). Each folder must contain `annotations/` and `images/train/` subdirectories. When multiple versioned annotation files exist (`instances_train_v1.json`, `instances_train_v2.json`, ...) the highest-versioned file is used automatically.

```bash
python scripts/convert_coco_to_yolo.py \
  --csv /path/to/dirs.csv \
  --output-dir /path/to/yolo_dataset
```

**Arguments:**

| Argument | Description |
|---|---|
| `--csv` | CSV file with columns `dir_path`, `split`, and `observation_id`. Each row points to an observation subfolder, its YOLO split, and the unique identifier used to prefix output filenames. |
| `--output-dir` | Destination directory for the converted YOLO dataset. |

**CSV format** (`--csv`):

```
dir_path,observation_id,split
/path/to/240101_01_label_L,240101_01_L,train
/path/to/240101_02_label_R,240101_02_R,val
```

**Output layout:**

```
<output-dir>/
├── images/
│   └── <split>/
├── labels/
│   └── <split>/
└── data.yaml
```

Image filenames in the output are prefixed with the `observation_id` value from the CSV to avoid collisions when images from different observation folders share the same base name.

---

##### scripts/coco_to_sam2_masks.py

Converts COCO instance annotations (as output by the [modified Labelme](https://github.com/mad4octos/LabelMe)) to SAM2 frame masks in the MOSE/DAVIS dataset format. Each frame is saved as a palette-indexed PNG where each pixel value corresponds to an object ID (background = 0, void/invalid = 254).

```bash
python scripts/coco_to_sam2_masks.py \
    --coco-file path/to/instances_train.json \
    --output-dir path/to/output \
    --video-name my_video
```

**Arguments:**

| Argument | Description |
|---|---|
| `--coco-file` | Path to the COCO annotations file (`instances_train.json`). |
| `--output-dir` | Root output directory. Masks are saved under `<output-dir>/Annotations/<video-name>/`. |
| `--video-name` | Video/sequence name used as the subfolder under `Annotations/`. |

**Output layout:**

```
<output-dir>/
└── Annotations/
    └── <video-name>/
        ├── 00001.png
        ├── 00002.png
        └── ...
```

Each annotation must have an `ObjID` attribute (integer ≥ 1). If multiple annotations on the same frame have overlapping masks, the annotation with the higher `ObjID` takes precedence.

---

### Notebooks

##### convert_masks.ipynb
Notebook to convert SAM2 predicted masks and annotations (stored as `.pkl` and `.npy`) into common annotation datasets: CVAT XML and Ultralytics YOLO (object-detection) format. 

##### visualize_yolo.ipynb
Notebook to load an exported YOLO object-detection dataset and visualize the images with their annotations (bounding boxes and class labels). Useful for quick QA of exports created by `convert_masks.ipynb`.
