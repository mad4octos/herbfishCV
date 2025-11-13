
### Instructions

- Fill the <FILL_ME> placeholders in the `configuration.py` file.
  - `DATA_ROOT_PATH`: defines where all the data will be found.
  - `ClassifierConfig.model_weights_path` defines where the model weights will be located.

- Run the `multi_dataset_builder.py` file. It will verify that all the input data is available before it starts running.

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
