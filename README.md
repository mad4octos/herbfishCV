

### Notebooks

##### convert_masks.ipynb
Notebook to convert SAM2 predicted masks and annotations (stored as `.pkl` and `.npy`) into common annotation datasets: CVAT XML and Ultralytics YOLO (object-detection) format. 

##### visualize_yolo.ipynb
Notebook to load an exported YOLO object-detection dataset and visualize the images with their annotations (bounding boxes and class labels). Useful for quick QA of exports created by `convert_masks.ipynb`.