from pathlib import Path

DATA_ROOT_PATH = Path(<<FILL_ME>>)


class Config:
    # Path to classifier weights for detecting correctly masked fish
    model_weights_path = Path(<<FILL_ME>>)

    # Path to CSV errors file
    errors_csv_filepath = DATA_ROOT_PATH / "SAM2_errors.csv"

    # Path towards directory containing all the annotation files
    annot_path = Path(<<FILL_ME>>)

    # Path towards directory containing all the masks files
    masks_path = Path(<<FILL_ME>>)

    # Minimum bounding box area (in pixels) required for a blob to be considered valid.
    # Blobs with a smaller area are ignored as likely noise or irrelevant detections.
    area_threshold = 200  # px

    # Minimum acceptable dimension (either width or height, in pixels) for a blob.
    # Blobs smaller than this in width or height are discarded.
    min_size_threshold = 20  # px

    # Use 4 for focal follow data, 5 for stationary data.
    number_of_zeros = 5

    # # Stationary files
    # RIGHT GX050100
    # data_path = DATA_ROOT_PATH / "stationary" / "GX050100"
    # annotations_filepath = (DATA_ROOT_PATH / "stationary" / "MLM_051524_site3_east_A_Right_GX050100_annotations.npy")
    # masks_filepath = (DATA_ROOT_PATH / "stationary" / "MLM_051524_site3_east_A_Right_GX050100_masks.pkl")

    # LEFT GX056267_frames
    data_path = DATA_ROOT_PATH / "stationary" / "GX056267_frames2"
    annotations_filepath = (
        DATA_ROOT_PATH
        / "stationary"
        / "MLM_051524_site3_east_B_Left_GX056267_annotations.npy"
    )
    masks_filepath = (
        DATA_ROOT_PATH
        / "stationary"
        / "MLM_051524_site3_east_B_Left_GX056267_masks.pkl"
    )

    # Focal follow files
    # data_path = ROOT_PATH / "focal_follow" / "MH_SR_062523_6_L"
    # annotations_filepath = (ROOT_PATH / "focal_follow" / "MH_SR_062523_6_L_annotations.npy")
    # masks_filepath = ROOT_PATH / "focal_follow" / "MH_SR_062523_6_L_masks.pkl"

    # NOTE:
    # The files *_masks.pkl and *_annotations.npy must match the keys.
    # For example:
    #   - ABC123_masks.pkl and ABC123_annotations.npy
    #   - DEF456_masks.pkl and DEF456_annotations.npy
    obsId_to_folder_map: dict[str, Path] = {
        "MLM_051524_site3_east_B_Left_GX056267": DATA_ROOT_PATH
        / "stationary"
        / "GX056267_frames2",
    }
