# Standard Library imports
from pathlib import Path
import multiprocessing as mp
from typing import Optional

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
from configuration import Config, ParsedObservationID, ClassifierConfig
from dataset_builder import DatumaroDatasetBuilder
from convert_utils import (
    find_annot,
    find_masks,
    find_obsId_in_errors_file,
    ObservationIdSimilarity,
)


class MultiBuilder:
    def __init__(
        self,
        errors_csv_filepath: Path,
        obsId_to_folder_map: dict[ParsedObservationID, Path],
        masks_path: Path,
        annot_path: Path,
    ) -> None:
        self.processes: list[mp.Process] = []
        self.obsId_to_folder_map = obsId_to_folder_map
        self.errors_csv_filepath = errors_csv_filepath
        self.masks_path = masks_path
        self.annot_path = annot_path

    def load_error_frames(self, obs_id: ParsedObservationID):
        """ """
        errors_df = load_errors_df(self.errors_csv_filepath, obs_id)
        return extract_error_frames(errors_df)

    def verify_existence(self):
        """
        Verify the existence of masks (.pkl), annotations (.npy) files, the images folder and the observation in the
        errors file.
        """

        for obs_id_object, images_path in self.obsId_to_folder_map.items():
            print(f"* Checking existence of assets for video {obs_id_object.videoname}")

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
                f"annotations file for video {obs_id_object.videoname} doesn't exist"
            )
            print(" - Found annotations file ")

            ############################################################################################################
            # Assert existence of masks file
            ############################################################################################################
            assert find_masks(self.masks_path, obs_id_object) is not None, (
                f"masks file for video {obs_id_object.videoname} doesn't exist "
            )
            print(" - Found masks file")

            ############################################################################################################
            # Check for the existence of the observation ID on the errors file
            ############################################################################################################
            errors_df = pd.read_csv(self.errors_csv_filepath)
            result = find_obsId_in_errors_file(obs_id_object, errors_df)
            if isinstance(result, ObservationIdSimilarity):
                print(result)
            else:
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
                )

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

            builder = DatumaroDatasetBuilder(
                obs_id=obs_id,
                masks=masks,
                error_frames=obsId_error_frames,
                chunked_df=chunked_df,
                label_categories=label_categories,
                images_path=images_path,
                export_root_path=export_root_path,
                classifier=classifier,
                blob_rules=blob_filter_rules,
                anomaly_rules=anomaly_rules,
                classifier_conf=ClassifierConfig.classifier_conf,
                target_class=["correct_fish_mask"],
                start_frame=Config.start_frame,
                max_frames=Config.max_frames,
                filename_num_zeros=Config.number_of_zeros,
                verbose=False,
                notebook_debug=False,
                window_size=Config.window_size,
            )
            dataset = builder.build()

            cvat_out = export_root_path / "dataset_cvat"
            yolo_out = export_root_path / "dataset_yolo"

            dataset.export(str(cvat_out), format="cvat")
            dataset.export(
                str(yolo_out),
                format="yolo_ultralytics_detection",
                add_path_prefix=False,
                save_media=True,
            )

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
    mb = MultiBuilder(
        Config.errors_csv_filepath,
        Config.obsId_to_folder_map,
        Config.masks_path,
        Config.annot_path,
    )
    mb.verify_existence()
    mb.build_all(output_path=Config.output_path)
