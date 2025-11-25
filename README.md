
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

2) To add a new observation and specify its location on disk, add a new `ParsedObservationID : <path>` entry under `Config.obsId_to_folder_map`, for example:
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
