# Standard Library imports
from pathlib import Path
import multiprocessing as mp

# External imports
from ultralytics import YOLO

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
from anomaly_rules import (
    AreaChangeAnomaly,
    LargeDisplacementAnomaly,
    AreaZScoreAnomaly,
    CompactnessChangeAnomaly,
    SolidityChangeAnomaly,
)
from configuration import Config, DATA_ROOT_PATH
from dataset_builder import DatumaroDatasetBuilder


class MultiBuilder:
    def __init__(
        self,
        errors_csv_filepath: Path,
        obsId_to_folder_map: dict[str, Path],
        masks_path: Path,
        annot_path: Path,
    ) -> None:
        self.processes: list[mp.Process] = []
        self.obsId_to_folder_map = obsId_to_folder_map
        self.errors_csv_filepath = errors_csv_filepath
        self.masks_path = masks_path
        self.annot_path = annot_path

    def load_error_frames(self, obs_id: str):
        """ """
        errors_df = load_errors_df(self.errors_csv_filepath, obs_id)
        return extract_error_frames(errors_df)

    def build_all(self):
        """ """

        try:
            for obs_id, images_path in self.obsId_to_folder_map.items():
                masks_filepath = self.masks_path / f"{obs_id}_masks.pkl"
                annot_filepath = self.annot_path / f"{obs_id}_annotations.npy"

                obsId_error_frames = self.load_error_frames(obs_id)

                export_root_path = DATA_ROOT_PATH / "exports" / obs_id
                export_root_path.mkdir(parents=True, exist_ok=True)

                process = mp.Process(
                    name=obs_id,
                    target=self.run_process,
                    args=(
                        obs_id,
                        images_path,
                        masks_filepath,
                        annot_filepath,
                        obsId_error_frames,
                        export_root_path,
                    ),
                    kwargs=({}),
                )

                self.processes.append(process)
                process.start()

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
            config = Config()
            classifier = YOLO(config.model_weights_path)

            masks = load_masks(masks_filepath)
            annotations_df = load_annotations(annot_filepath)
            label_categories = load_categories(annotations_df)
            chunked_df = get_frame_chunks_df(annotations_df)

            # Blobs will be pre-filtered according to these rules
            blob_filter_rules = [
                MinAreaRule(config.area_threshold),
                MinSizeRule(config.min_size_threshold),
            ]

            # Anomalies across time in blob properties will be detected using these rules
            anomaly_rules = [
                AreaChangeAnomaly(),
                LargeDisplacementAnomaly(),
                SolidityChangeAnomaly(),
                AreaZScoreAnomaly(),
                CompactnessChangeAnomaly(),
            ]

            builder = DatumaroDatasetBuilder(
                obs_id=obs_id,
                masks=masks,
                error_frames=obsId_error_frames,
                chunked_df=chunked_df,
                label_categories=label_categories,
                classifier=classifier,
                blob_rules=blob_filter_rules,
                anomaly_rules=anomaly_rules,
                classifier_conf=0.25,
                target_class="correct_fish_mask",
                start_frame=0,
                max_frames=None,
                images_path=images_path,
                filename_num_zeros=config.number_of_zeros,
                verbose=False,
                notebook_debug=False,
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

            print(f"[OK] Finished: {export_root_path}")

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
    # # On Windows, spawn is required for safety with CUDA/torch
    # try:
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass

    mb = MultiBuilder(
        Config.errors_csv_filepath,
        Config.obsId_to_folder_map,
        Config.masks_path,
        Config.annot_path,
    )
    mb.build_all()
