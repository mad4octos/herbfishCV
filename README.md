
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

### `configuration.py` explanation

- The `Config` class:
  - Defines the relative paths towards the annotations, masks and errors files.
  - Has the `obsId_to_folder_map` where observations of interest are defined. Each `ParsedObservationID : <path>` pair defines the observation of interest and its location on disk (the ParsedObservationID class is used to input the constituent fields of the ObservationID, and it's used to test different versions of the ObservationID string).
  - Has an `anomaly_rules` field, where all the anomaly rules acting on the timeseries of blobs properties are defined.

- The `ClassifierConfig` class:
  - Defines in `model_weights_path` the path towards the classifier model used to reject incorrectly masked fish.


### Notebooks

##### convert_masks.ipynb
Notebook to convert SAM2 predicted masks and annotations (stored as `.pkl` and `.npy`) into common annotation datasets: CVAT XML and Ultralytics YOLO (object-detection) format. 

##### visualize_yolo.ipynb
Notebook to load an exported YOLO object-detection dataset and visualize the images with their annotations (bounding boxes and class labels). Useful for quick QA of exports created by `convert_masks.ipynb`.
