#!/usr/bin/env python
"""
Execution script for network's threshold-free cluster
enhancement (TFCE) analysis. This script implements TFCE
using nilearn's second-level non_parametric_inference
function. From the command line, it takes in which task
and contrast you would like to run TFCE on and then
executes the pipeline and saves out the figures to an
output directory.
"""

import argparse
import glob
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image
from nilearn.maskers import NiftiMapsMasker
from nilearn.glm.second_level import non_parametric_inference
from nilearn.plotting import plot_stat_map
from nilearn.image import resample_to_img, load_img
from templateflow import api as tf

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DataProcessor:
    """Main processor for TFCE analysis workflow."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        task_name: str,
        contrast_name: str,
        n_permutations: int = 10000,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.task_name = task_name
        self.contrast_name = contrast_name
        self.n_permutations = n_permutations
        self.logger = self._setup_logging()

        # Initialize component classes
        self.file_globber = FileGlobber(self.logger)
        self.mni_fetcher = MNIBackgroundFetcher(self.logger)
        self.tfce_analyzer = TFCEAnalyzer(self.logger, n_permutations)

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure logging for the processor."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def validate_inputs(self) -> None:
        """Validate input directory and parameters."""
        self.logger.info("Validating input parameters...")

        # Check input directory
        if not self.data_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.data_dir}")
            sys.exit(1)
        if not self.data_dir.is_dir():
            self.logger.error(f"Input path is not a directory: {self.data_dir}")
            sys.exit(1)

    def process(self) -> None:
        """Main processing pipeline."""
        self.logger.info("Starting TFCE analysis pipeline")
        self.logger.info("Using the following configuration...")
        for attr, value in self.__dict__.items():
            if (
                attr != "logger"
                and not attr.startswith("_")
                and not callable(getattr(self, attr, None))
            ):
                self.logger.info(f"  {attr}: {value}")

        # Validate inputs
        self.validate_inputs()

        # Create output directory
        self.logger.info(f"Creating output directory: {self.output_dir.absolute()}")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Output directory created/verified: {self.output_dir.absolute()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

        # Get NIfTI files using FileGlobber
        try:
            nifti_files = self.file_globber.get_files(
                self.data_dir, self.task_name, self.contrast_name
            )
            if not nifti_files:
                self.logger.error(
                    "No contrast files found matching the specified pattern"
                )
                sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error getting contrast files: {e}")
            sys.exit(1)

        # Fetch MNI background image
        try:
            mni_bg = self.mni_fetcher.fetch_background()
            self.logger.info("MNI background image ready for use")
        except Exception as e:
            self.logger.error(f"Error fetching MNI background: {e}")
            sys.exit(1)

        # Run TFCE analysis
        try:
            self.logger.info("Starting TFCE analysis...")
            results = self.tfce_analyzer.run_analysis(nifti_files)

            # Log data statistics for different maps
            self.tfce_analyzer.log_data_statistics(results, "tfce")
            self.tfce_analyzer.log_data_statistics(results, "logp_max_tfce")
            self.tfce_analyzer.log_data_statistics(results, "t")

            # Save TFCE results
            output_filename = f"tfce_{self.task_name}_{self.contrast_name}.nii.gz"
            output_path = self.output_dir / output_filename
            nib.save(results["tfce"], str(output_path))
            self.logger.info(f"TFCE results saved to: {output_path}")

            # Save t-map as well
            t_output_filename = f"tmap_{self.task_name}_{self.contrast_name}.nii.gz"
            t_output_path = self.output_dir / t_output_filename
            nib.save(results["t"], str(t_output_path))
            self.logger.info(f"T-map saved to: {t_output_path}")

            self.logger.info("TFCE analysis completed successfully")

        except Exception as e:
            self.logger.error(f"Error in TFCE analysis: {e}")
            sys.exit(1)

        # Save the t-stat plot as PNG
        try:
            plot_title = f"Second Level Analysis: {self.task_name} - T map (contrast: {self.contrast_name})"
            plt = plot_stat_map(
                results["t"],
                title=plot_title,
                threshold=0,
                # TODO: Consider fixing or dynamically setting scale
                # vmax=
                # vmin=
                cmap="cold_hot",
                bg_img=mni_bg,
            )

            # Create filename
            png_filename = f"tmap_plot_{self.task_name}_{self.contrast_name}.png"
            png_save_path = self.output_dir / png_filename

            # Save the plot
            plt.savefig(str(png_save_path))
            self.logger.info(f"T-map plot saved to: {png_save_path}")

            # Close the plot to free up memory
            plt.close()
        except Exception as e:
            self.logger.error(f"Error saving T-map plot: {e}")

        # Save the TFCE plot as PNG
        try:
            tfce_plot_title = f"Second Level Analysis: {self.task_name} - TFCE raw map (contrast: {self.contrast_name})"
            tfce_plot = plot_stat_map(
                results["tfce"],
                title=tfce_plot_title,
                threshold=0,
                # TODO: Consider fixing or dynamically setting scale
                # vmax=
                # vmin=
                cmap="cold_hot",
                bg_img=mni_bg,
            )

            # Create filename
            tfce_png_filename = f"tfce_plot_{self.task_name}_{self.contrast_name}.png"
            tfce_png_save_path = self.output_dir / tfce_png_filename

            # Save the plot
            tfce_plot.savefig(str(tfce_png_save_path))
            self.logger.info(f"TFCE plot saved to: {tfce_png_save_path}")

            # Close the plot to free up memory
            tfce_plot.close()
        except Exception as e:
            self.logger.error(f"Error saving TFCE plot: {e}")


class FileGlobber:
    """Handles file globbing for NIfTI files based on path patterns."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def get_files(
        self, base_dir: Path, task_name: str, contrast_name: str
    ) -> List[Path]:
        """Get all files matching the specified pattern.

        Args:
            base_dir: Base directory path
            task_name: Task name to search for
            contrast_name: Contrast name to search for

        Returns:
            List of Path objects for matching files
        """
        pattern = f"{base_dir}/sub-*/{{task_name}}/fixed_effects/*{{contrast_name}}*fixed-effects.nii.gz"
        pattern = pattern.format(task_name=task_name, contrast_name=contrast_name)

        self.logger.info(f"Searching for files with pattern: {pattern}")

        matching_files = glob.glob(pattern)
        matching_paths = [Path(f) for f in matching_files]

        self.logger.info(f"Found {len(matching_paths)} matching files")

        return matching_paths


class MNIBackgroundFetcher:
    """Fetches and manages MNI background image data."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._mni_bg = None

    def fetch_background(self):
        """Fetch MNI152NLin2009cAsym background image."""
        self.logger.info("Fetching MNI152NLin2009cAsym background image...")
        self._mni_bg = tf.get("MNI152NLin2009cAsym", resolution=2, suffix="T1w", desc=None) 
        self.logger.info("MNI152NLin2009cAsym background image fetched successfully")

        return self._mni_bg

    @property
    def background_image(self):
        """Get the MNI background image."""
        if self._mni_bg is None:
            return self.fetch_background()
        return self._mni_bg


class TFCEAnalyzer:
    """Handles TFCE analysis with non-parametric inference."""

    def __init__(self, logger: logging.Logger, n_permutations: int = 10000):
        self.logger = logger
        self.n_permutations = n_permutations

    def run_analysis(self, nifti_files: List[Path]) -> Dict:
        """Run TFCE analysis on the provided NIfTI files.

        Args:
            nifti_files: List of NIfTI file paths

        Returns:
            Dictionary containing TFCE analysis results
        """
        if not nifti_files:
            raise ValueError("No NIfTI files provided for analysis")

        self.logger.info(f"Running TFCE analysis on {len(nifti_files)} files")
        self.logger.info(f"Using {self.n_permutations} permutations")

        # Create design matrix for one-sample t-test
        design_matrix = pd.DataFrame([1] * len(nifti_files), columns=["intercept"])
        self.logger.info(f"Created design matrix with shape: {design_matrix.shape}")

        # Convert Path objects to strings for nilearn
        file_paths = [str(f) for f in nifti_files]

        # Run non-parametric inference
        self.logger.info("Starting non-parametric inference...")
        second_level_map = non_parametric_inference(
            second_level_input=file_paths,
            design_matrix=design_matrix,
            second_level_contrast="intercept",
            n_perm=self.n_permutations,
            tfce=True,
            two_sided_test=True,
            verbose=1,
        )

        self.logger.info("TFCE analysis completed successfully")
        return second_level_map

    def log_data_statistics(self, data_map: Dict, map_name: str):
        """Log min, max, and unique values for a data map.

        Args:
            data_map: Dictionary containing analysis results
            map_name: Name of the map to analyze
        """
        if map_name in data_map:
            data = data_map[map_name].get_fdata()
            self.logger.info(
                f"{map_name} - min: {np.min(data)}, max: {np.max(data)}, unique: {len(np.unique(data))} values"
            )


def get_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run TFCE analysis on contrast images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=("Base directory containing lev1 outputs directory in MNI space "
              "(e.g., /oak/stanford/groups/russpold/data/network_grant/"
              "discovery_BIDS_20250402/derivatives/output_lev1_mni)"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for TFCE results and figures",
    )

    parser.add_argument(
        "--task-name", type=str, required=True, help="Task name (e.g., nBack)"
    )

    parser.add_argument(
        "--contrast-name",
        type=str,
        required=True,
        help="Contrast name (e.g., twoBack-oneBack)",
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations for non-parametric inference",
    )

    return parser


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()

    # Initialize processor
    processor = DataProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task_name=args.task_name,
        contrast_name=args.contrast_name,
        n_permutations=args.n_permutations,
    )

    # Run TFCE analysis
    try:
        processor.process()

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
