from pathlib import Path

DATA_ROOT_PATH = Path(r"F:\DATASETS\GIL LAB\Example SAM2")


class Config:
    # Minimum bounding box area required for a blob to be considered
    # TODO: Calculate this threshold based on data
    area_threshold = 200  # px

    # Use 4 for focal follow data, 5 for stationary data.
    # TODO: standardize the use of N zeros when extracting data, maybe 8?
    number_of_zeros = 8

    file_extension = ".jpg"

    errors_csv_filepath = DATA_ROOT_PATH / "SAM2_errors.csv"

    # # Stationary files
    # RIGHT GX050100
    # DATA_PATH = DATA_ROOT_PATH / "stationary" / "GX050100"
    # annotations_filepath = (
    #     DATA_ROOT_PATH
    #     / "stationary"
    #     / "MLM_051524_site3_east_A_Right_GX050100_annotations.npy"
    # )
    # masks_filepath = (
    #     DATA_ROOT_PATH / "stationary" / "MLM_051524_site3_east_A_Right_GX050100_masks.pkl"
    # )

    # LEFT GX056267_frames
    DATA_PATH = DATA_ROOT_PATH / "stationary" / "GX056267_frames2"
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

    classifier_path = Path(r"herbfishCV\runs\classify\train5\weights\best.pt")
