# Standard Library imports
from pathlib import Path

# External imports
import cv2
import numpy as np
from tqdm import tqdm

# Local imports
from convert_utils import load_masks, _get_frame_filename, get_blobs_from_mask, BlobInfo
from common import sparse_mask_tensor_to_dense_numpy
from multi_dataset_builder import find_masks
from configuration import Config, ClassifierConfig, ParsedObservationID, ManualObservationID


def get_filtered_blobs(
    dense_object_mask: np.ndarray,
    obj_id: int,
    frame_idx: int,
    min_size_threshold,
    area_threshold,
) -> list[BlobInfo]:
    """Get filtered blobs from dense mask.
    The blobs are filtered based on a minimal area, height and width."""

    valid_blobs: list[BlobInfo] = []
    for blob in get_blobs_from_mask(dense_object_mask, obj_id, frame_idx):
        if blob.area < area_threshold:
            continue
        if blob.w < min_size_threshold or blob.h < min_size_threshold:
            continue

        valid_blobs.append(blob)

    return valid_blobs


def extract_blobs(
    object_mask,
    frame_idx,
    obj_id,
    input_image: np.ndarray,
    area_threshold=200,
    min_size_threshold=20,
    output_folder: Path | str = "output_blobs",
    save=False,
    do_mask=False,
):
    """ """

    # Create output folder if required
    output_dirpath = Path(output_folder)
    output_dirpath.mkdir(exist_ok=True, parents=True)

    filtered_blobs = get_filtered_blobs(
        object_mask,
        obj_id,
        frame_idx,
        area_threshold=area_threshold,
        min_size_threshold=min_size_threshold,
    )

    for blob in filtered_blobs:
        observation_id = f"frame_{frame_idx}_obj_{obj_id}_blob_{blob.blob_num}"
        if do_mask:
            crop = blob.mask_and_crop_blob(input_image, remove_background=False)
        else:
            crop = blob.crop_from_image(input_image)

        if save:
            blob_filename = Path(output_folder, f"{observation_id}.png")
            cv2.imwrite(str(blob_filename), crop)


def main(obsId_to_folder_map: dict[ParsedObservationID | ManualObservationID, Path], output_folder: Path):
    """ """

    filename_num_zeros = Config.number_of_zeros
    area_threshold = Config.area_threshold
    masks_path = Config.masks_path

    for obs_id_object, images_path in obsId_to_folder_map.items():
        masks_filepath = find_masks(masks_path, obs_id_object)
        if masks_filepath is None:
            print(f"Couldn't find masks for {obs_id_object.to_str()}")
            continue

        masks = load_masks(masks_filepath)
        print(f"Loading masks from {masks_path}")

        for extracted_frame_idx in tqdm(
            masks.keys(), desc="Extracting frames", unit="frame"
        ):
            # Load masks available for this frame
            frame_masks = masks[extracted_frame_idx]

            # The number of zeros is important. For the focal follow data is 4, for the stationary data is 5
            filename = _get_frame_filename(extracted_frame_idx, filename_num_zeros)
            image_filepath = images_path / filename

            # Load frame
            input_frame = cv2.imread(str(image_filepath), cv2.IMREAD_COLOR)
            assert input_frame is not None

            for obj_id in frame_masks.keys():
                # Load mask corresponding to one object (can be multiple blobs!)
                sparse_mask = frame_masks[obj_id]
                dense_object_mask = sparse_mask_tensor_to_dense_numpy(sparse_mask)

                # TODO: extract to a folder like:
                #     <ROOT>\exports\JGL_05-30-2024_site5_east_B_Right_GX030843\extracted_crops
                # instead of:
                #     <ROOT>\exports\extracted_crops\JGL_05-30-2024_site5_east_B_Right_GX030843
                extract_blobs(
                    dense_object_mask,
                    extracted_frame_idx,
                    obj_id,
                    input_frame,
                    area_threshold=area_threshold,
                    output_folder=output_folder / obs_id_object.to_str(),
                    save=True,
                    do_mask=True,
                )


if __name__ == "__main__":
    output_folder = ClassifierConfig.data_path
    main(Config.obsId_to_folder_map, output_folder)
