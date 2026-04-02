# Standard Library imports
from pathlib import Path
import multiprocessing as mp
from typing import Optional
import argparse

# External imports
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd

# Local imports
from convert_utils import (
    get_frame_chunks_df,
    load_masks,
    load_annotations,
    load_categories,
    load_errors_df,
    extract_error_frames,
)
from blob_filter_rules import MinAreaRule, MinSizeRule
from anomaly_rules import create_anomaly_rules
from configuration import Config, ParsedObservationID, ManualObservationID, ClassifierConfig
from dataset_builder import DatumaroDatasetBuilder
from convert_utils import (
    find_annot,
    find_masks,
    find_obsId_in_errors_file,
    ObservationIdSimilarity,
    next_run_dir,
)


def parse_args():
    """ """
    parser = argparse.ArgumentParser(
        description="Build datasets from SAM2 masks and annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use configuration from Config class (default)
  python multi_dataset_builder.py

  # Pass a ManualObservationID directly via CLI
  python multi_dataset_builder.py --manual \\
      --errors-csv-filepath "/path/to/SAM2_errors.csv" \\
      --errors-obs-id "JM_060724_152_playa_largu_scuba_TPScv_L" \\
      --masks-filepath "/path/to/CR_JM_060724_152_playa_largu_scuba_TPScv_L_mask.pkl" \\
      --annot-filepath "/path/to/CR_JM_060724_152_playa_largu_scuba_TPScv_L_annotations.npy" \\
      --images-dirpath "/path/to/images/MH_JM_060724_152_L"

  python multi_dataset_builder.py \\
      --original-fps 23.997 --extracted-fps 3 --final-fps 1 --sam2-start 100
        """
    )
    parser.add_argument(
        "--ignore-missing-observation-ids",
        action="store_true",
        help="Do not stop execution when an observation ID is missing from SAM2_errors.csv.",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Disable automatic mask cleaning (blob filters, classifier, anomaly detection). "
             "Only frames manually specified in the errors CSV will have their masks removed.",
    )
    parser.add_argument(
        "--extracted-fps",
        type=int,
        default=None,
        help="FPS at which frames were extracted from the video by ffmpeg. "
        "Used together with --final-fps to subsample frames.",
    )
    parser.add_argument(
        "--final-fps",
        type=int,
        default=None,
        help="Desired FPS for the output dataset. "
        "Used together with --extracted-fps to subsample frames (e.g., --extracted-fps 3 --final-fps 1 keeps every 3rd frame).",
    )
    parser.add_argument(
        "--original-fps",
        type=float,
        default=None,
        help="Native frame rate of the original video (before ffmpeg extraction). "
        "The frame numbers in the .npy annotations file are in this FPS space. "
        "Used together with --extracted-fps and --sam2-start to map extracted frame numbers "
        "back to original video frame numbers for annotation lookup.",
    )
    parser.add_argument(
        "--sam2-start",
        type=int,
        default=None,
        help="Zero-indexed frame number in the original video from which ffmpeg began extracting frames. "
        "Used together with --original-fps and --extracted-fps to map extracted frame numbers "
        "back to original video frame numbers for annotation lookup.",
    )

    # Manual observation ID arguments
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual observation ID mode instead of Config.obsId_to_folder_map.",
    )
    parser.add_argument(
        "--errors-obs-id",
        type=str,
        help="The exact observation ID string as it appears in the errors CSV file.",
    )
    parser.add_argument(
        "--errors-csv-filepath",
        type=Path,
        help="The absolute file path for the errors csv file (e.g., '/path/to/SAM2_errors.csv').",
    )
    parser.add_argument(
        "--masks-filepath",
        type=Path,
        help="The absolute file path for the masks file (e.g., '/path/to/video_name_mask.pkl').",
    )
    parser.add_argument(
        "--annot-filepath",
        type=Path,
        help="The absolute file path for the annotations file (e.g., '/path/to/video_name_annotations.npy').",
    )
    parser.add_argument(
        "--images-dirpath",
        type=Path,
        help="The absolute path to the directory containing the image frames.",
    )

    args = parser.parse_args()

    # Validate FPS arguments: both must be provided together
    if (args.extracted_fps is None) != (args.final_fps is None):
        parser.error("--extracted-fps and --final-fps must be used together.")
    if args.extracted_fps is not None:
        if (args.extracted_fps <= 0) or (args.final_fps <= 0):
            parser.error("--extracted-fps and --final-fps must be positive integers.")
        if args.final_fps > args.extracted_fps:
            parser.error("--final-fps cannot be greater than --extracted-fps.")

    # Validate original-fps / sam2-start: both must be provided together, and require --extracted-fps
    if (args.original_fps is None) != (args.sam2_start is None):
        parser.error("--original-fps and --sam2-start must be used together.")
    if args.original_fps is not None:
        if args.original_fps <= 0:
            parser.error("--original-fps must be a positive number.")
        if args.sam2_start < 0:
            parser.error("--sam2-start must be a non-negative integer.")
        if args.extracted_fps is None:
            parser.error(
                "--original-fps and --sam2-start require --extracted-fps (and --final-fps) to be set."
            )

    # Validate that all manual args are provided when --manual is used
    if args.manual:
        required_manual_args = ["errors_obs_id", "errors_csv_filepath", "masks_filepath", "annot_filepath", "images_dirpath"]
        missing = [arg for arg in required_manual_args if getattr(args, arg) is None]
        if missing:
            parser.error(
                f"--manual requires all of: --errors-obs-id, --errors-csv-filepath, --masks-filepath, --annot-filepath, --images-dirpath. "
                f"Missing: {', '.join('--' + arg.replace('_', '-') for arg in missing)}"
            )

    return args


class MultiBuilder:
    def __init__(
        self,
        errors_csv_filepath: Path,
        obsId_to_folder_map: dict[ParsedObservationID | ManualObservationID, Path],
        masks_path: Path,
        annot_path: Path,
        ignore_missing_observation_ids: bool = False,
        no_auto: bool = False,
        extracted_fps: int | None = None,
        final_fps: int | None = None,
        original_fps: float | None = None,
        sam2_start: int | None = None,
    ) -> None:
        self.processes: list[mp.Process] = []
        self.obsId_to_folder_map = obsId_to_folder_map
        self.errors_csv_filepath = errors_csv_filepath
        self.masks_path = masks_path
        self.annot_path = annot_path
        self.ignore_missing_observation_ids = ignore_missing_observation_ids
        self.no_auto = no_auto
        self.extracted_fps = extracted_fps
        self.final_fps = final_fps
        self.original_fps = original_fps
        self.sam2_start = sam2_start

    def load_error_frames(self, obs_id: ParsedObservationID | ManualObservationID) -> list[int]:
        """ """
        errors_df = load_errors_df(self.errors_csv_filepath, obs_id)
        return extract_error_frames(errors_df)

    def verify_existence(self):
        """
        Verify the existence of masks (.pkl), annotations (.npy) files, the images folder and the observation in the
        errors file.
        """

        for obs_id_object, images_path in self.obsId_to_folder_map.items():
            # Get display name for logging (supports both ParsedObservationID and ManualObservationID)
            display_name = (
                obs_id_object.videoname
                if isinstance(obs_id_object, ParsedObservationID)
                else obs_id_object.to_str()
            )
            print(f"* Checking existence of assets for video {display_name}")

            ############################################################################################################
            # Assert existence of images directory
            ############################################################################################################
            assert images_path.exists(), f"{images_path} doesn't exist"
            assert images_path.is_dir(), f"{images_path} is not a directory"
            print(" - Found frames dir ")

            ############################################################################################################
            # Assert existence of annotations file
            ############################################################################################################
            assert find_annot(self.annot_path, obs_id_object) is not None, (
                f"annotations file for {display_name} doesn't exist"
            )
            print(" - Found annotations file ")

            ############################################################################################################
            # Assert existence of masks file
            ############################################################################################################
            assert find_masks(self.masks_path, obs_id_object) is not None, (
                f"masks file for {display_name} doesn't exist "
            )
            print(" - Found masks file")

            ############################################################################################################
            # Check for the existence of the observation ID on the errors file
            ############################################################################################################
            errors_df = pd.read_csv(self.errors_csv_filepath)
            result = find_obsId_in_errors_file(obs_id_object, errors_df)
            
            if isinstance(result, ObservationIdSimilarity):
                print("Couldn't find observation id in errors file.")
                print(result)
                if not self.ignore_missing_observation_ids:
                    raise ValueError(
                        f"Observation ID '{obs_id_object.to_str()}' not found in errors file.\n"
                        "Run with --ignore-missing-observation-ids flag to ognore missing Observation IDs."
                    )
            elif isinstance(result, pd.DataFrame):
                print(" - Found observation id in errors file")

    def build_all(self, output_path: Path):
        """ """

        total_jobs = len(self.obsId_to_folder_map)
        pbar = tqdm(total=total_jobs, desc="Processing observations", unit="obs")

        try:
            for obs_id_object, images_path in self.obsId_to_folder_map.items():
                # It's assumed that the existence of annotations and masks has been previously asserted
                masks_filepath = find_masks(self.masks_path, obs_id_object)
                annot_filepath = find_annot(self.annot_path, obs_id_object)
                obsId_error_frames = self.load_error_frames(obs_id_object)

                obs_id_str = obs_id_object.to_str()

                self.run_process(
                    obs_id_str,
                    images_path,
                    masks_filepath,
                    annot_filepath,
                    obsId_error_frames,
                    output_path / obs_id_str,
                    no_auto=self.no_auto,
                    extracted_fps=self.extracted_fps,
                    final_fps=self.final_fps,
                    original_fps=self.original_fps,
                    sam2_start=self.sam2_start,
                )
                pbar.update(1)

            pbar.close()

        except Exception as e:
            print("Exception", str(e))

    @staticmethod
    def run_process(
        obs_id: str,
        images_path: Path,
        masks_filepath: Path,
        annot_filepath: Path,
        obsId_error_frames: list[int],
        export_root_path: Path,
        no_auto: bool = False,
        extracted_fps: int | None = None,
        final_fps: int | None = None,
        original_fps: float | None = None,
        sam2_start: int | None = None,
    ):
        """ """
        print(f"Creating job for observation '{obs_id}'")

        try:
            print(f"Loading model weights from {ClassifierConfig.model_weights_path}")
            classifier = YOLO(ClassifierConfig.model_weights_path)

            masks = load_masks(masks_filepath)
            annotations_df = load_annotations(annot_filepath)
            label_categories = load_categories(annotations_df)
            chunked_df = get_frame_chunks_df(annotations_df)

            # Blobs will be pre-filtered according to these rules
            blob_filter_rules = [
                MinAreaRule(Config.area_threshold),
                MinSizeRule(Config.min_size_threshold),
            ]

            # Anomalies across time in blob properties will be detected using these rules
            anomaly_rules = create_anomaly_rules(Config.anomaly_rules)

            run_dir = next_run_dir(export_root_path)

            builder = DatumaroDatasetBuilder(
                obs_id=obs_id,
                masks=masks,
                error_frames=obsId_error_frames,
                chunked_df=chunked_df,
                annotations_df=annotations_df,
                label_categories=label_categories,
                images_path=images_path,
                export_root_path=run_dir,
                classifier=classifier,
                blob_rules=blob_filter_rules,
                anomaly_rules=anomaly_rules,
                incorrect_cls_conf_thresh=ClassifierConfig.incorrect_cls_conf_thresh,
                correct_class=ClassifierConfig.correct_class,
                incorrect_class=ClassifierConfig.incorrect_class,
                start_frame=Config.start_frame,
                max_frames=Config.max_frames,
                filename_num_zeros=Config.number_of_zeros,
                verbose=False,
                notebook_debug=False,
                window_size=Config.window_size,
                no_auto=no_auto,
                extracted_fps=extracted_fps,
                final_fps=final_fps,
                original_fps=original_fps,
                sam2_start=sam2_start,
            )
            dataset = builder.build()

            coco_out = run_dir / obs_id
            dataset.export(str(coco_out), format="coco", save_media=True, tasks="instances")

            print(f"Finished: '{obs_id}', results written to '{export_root_path}'.")

        except Exception as e:
            print(f"[ERROR] Exception in '{mp.current_process().name}': {e}")

    def _check_process_health(self):
        """Check health of all child processes."""
        healthy = True
        for p in self.processes:
            if not p.is_alive() and p.exitcode not in (None, 0):
                print(f"Process '{p.name}' failed with exit code {p.exitcode}")
                healthy = False
        return healthy


if __name__ == "__main__":
    args = parse_args()

    # Build obsId_to_folder_map based on mode
    obsId_to_folder_map: dict[ParsedObservationID | ManualObservationID, Path]
    if args.manual:

        # Validate errors CSV file
        if not args.errors_csv_filepath.is_absolute():
            raise ValueError(
                f"Errors CSV path must be absolute: {args.errors_csv_filepath}"
            )
        if not args.errors_csv_filepath.exists():
            raise FileNotFoundError(
                f"Errors CSV file not found: {args.errors_csv_filepath}"
            )
        if not args.errors_csv_filepath.is_file():
            raise ValueError(
                f"Errors CSV path is not a file: {args.errors_csv_filepath}"
            )
        errors_csv_filepath = args.errors_csv_filepath

        # Validate masks file
        if not args.masks_filepath.is_absolute():
            raise ValueError(f"Masks file path must be absolute: {args.masks_filepath}")
        if not args.masks_filepath.exists():
            raise FileNotFoundError(f"Masks file not found: {args.masks_filepath}")
        if not args.masks_filepath.is_file():
            raise ValueError(f"Masks file path is not a file: {args.masks_filepath}")
        masks_path = args.masks_filepath.parent

        # Validate annotations file
        if not args.annot_filepath.is_absolute():
            raise ValueError(
                f"Annotations file path must be absolute: {args.annot_filepath}"
            )
        if not args.annot_filepath.exists():
            raise FileNotFoundError(
                f"Annotations file not found: {args.annot_filepath}"
            )
        if not args.annot_filepath.is_file():
            raise ValueError(
                f"Annotations file path is not a file: {args.annot_filepath}"
            )
        annot_path = args.annot_filepath.parent

        # Validate images directory
        if not args.images_dirpath.is_absolute():
            raise ValueError(
                f"Images directory path must be absolute: {args.images_dirpath}"
            )
        if not args.images_dirpath.exists():
            raise FileNotFoundError(
                f"Images directory not found: {args.images_dirpath}"
            )
        if not args.images_dirpath.is_dir():
            raise ValueError(
                f"Images path is not a directory: {args.images_dirpath}"
            )

        # Create ManualObservationID from CLI arguments
        manual_obs_id = ManualObservationID(
            errors_obs_id=args.errors_obs_id,
            masks_filename=args.masks_filepath.name,
            annotations_filename=args.annot_filepath.name,
        )
        obsId_to_folder_map = {manual_obs_id: args.images_dirpath}

    else:
        # Use configuration from Config class
        obsId_to_folder_map = Config.obsId_to_folder_map
        errors_csv_filepath = Config.errors_csv_filepath
        masks_path = Config.masks_path
        annot_path = Config.annot_path

    mb = MultiBuilder(
        errors_csv_filepath,
        obsId_to_folder_map,
        masks_path,
        annot_path,
        args.ignore_missing_observation_ids,
        no_auto=args.no_auto,
        extracted_fps=args.extracted_fps,
        final_fps=args.final_fps,
        original_fps=args.original_fps,
        sam2_start=args.sam2_start,
    )
    mb.verify_existence()
    mb.build_all(output_path=Config.output_path)
