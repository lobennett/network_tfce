# Network TFCE Analysis

A threshold-free cluster enhancement (TFCE) analysis pipeline for neuroimaging data using nilearn's non-parametric inference. This package performs TFCE analysis on contrast images to identify significant clusters without arbitrary cluster-forming thresholds.

## Features

- Object-oriented design with clean separation of concerns
- Threshold-free cluster enhancement (TFCE) analysis
- Non-parametric statistical inference with permutation testing
- Support for first-level contrast images in MNI space
- Automated file discovery and validation
- SLURM cluster integration with apptainer containers
- Detailed logging and error handling

## Prerequisites

- SLURM cluster environment
- Apptainer/Singularity container runtime
- First-level contrast images in MNI space (NIfTI format)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lobennett/network_tfce.git 
cd network_tfce
```

2. **Pull the required apptainer image** (this step is required before running the pipeline):

   **Option A - Direct execution:**
   ```bash
   ./pull_image.sh
   ```
   
   **Option B - SLURM batch submission:**
   ```bash
   sbatch pull_image.sh
   ```
   
   This will download the fMRI processing environment container to `./apptainer/fmri_env_latest.sif`.

3. Ensure scripts are executable:
```bash
chmod +x run_tfce.sh pull_image.sh
```

## Usage

### Basic Usage

```bash
./run_tfce.sh [email] [additional arguments for run_tfce.py]
```

### Command Line Arguments

The TFCE script supports the following arguments:

- `--data-dir`: Base directory containing first-level outputs in MNI space (required)
- `--output-dir`: Output directory for TFCE results (default: `results`)
- `--task-name`: Task name to analyze (required, e.g., `nBack`)
- `--contrast-name`: Contrast name to analyze (required, e.g., `twoBack-oneBack`)
- `--n-permutations`: Number of permutations for non-parametric inference (default: 10000)

### Examples

```bash
# Basic usage with required parameters
./run_tfce.sh user@stanford.edu --data-dir /path/to/first_level_outputs --task-name nBack --contrast-name twoBack-oneBack

# Custom parameters
./run_tfce.sh user@stanford.edu --data-dir /path/to/data --task-name nBack --contrast-name twoBack-oneBack --n-permutations 5000 --output-dir my_tfce_results

# Run locally (without SLURM submission)
python run_tfce.py --data-dir /path/to/first_level_outputs --task-name nBack --contrast-name twoBack-oneBack
```

## Input Data Format

The input data should be first-level contrast images in NIfTI format organized in the following directory structure:

```
data_directory/
├── sub-001/
│   └── task_name/
│       └── fixed_effects/
│           └── contrast_name*fixed-effects.nii.gz
├── sub-002/
│   └── task_name/
│       └── fixed_effects/
│           └── contrast_name*fixed-effects.nii.gz
...
```

The script automatically discovers files matching the pattern:
`{data_dir}/sub-*/{{task_name}}/fixed_effects/*{{contrast_name}}*fixed-effects.nii.gz`

All images must be in MNI space and have consistent dimensions.

## Output Structure

The analysis generates the following outputs in the specified output directory:

```
results/
├── tfce_taskname_contrastname.nii.gz    # TFCE statistical map
└── tmap_taskname_contrastname.nii.gz    # T-statistic map
```

Where `taskname` and `contrastname` are the values provided via command line arguments.

## Class Structure

The pipeline uses an object-oriented design with four main classes:

### `DataProcessor`
Main orchestrator class that coordinates the entire workflow:
- Input validation and directory management
- Output directory creation
- Workflow coordination and logging

### `FileGlobber`
Handles file discovery and validation:
- Pattern-based file searching
- NIfTI file validation
- Path management

### `MNIBackgroundFetcher`
Manages MNI background image data:
- Downloads and caches MNI152NLin2009aAsym template
- Provides background images for visualization

### `TFCEAnalyzer`
Handles TFCE analysis and statistics:
- Non-parametric inference with permutation testing
- TFCE statistical map generation
- Data statistics logging and validation

## SLURM Configuration

The script submits jobs with the following default settings:
- **Time limit**: 2 days
- **CPUs per task**: 8
- **Memory**: 32GB
- **Partitions**: russpold, hns, normal
- **Single job**: (not array-based since TFCE processes all subjects together)

## Logs

Job logs are saved in the `log/` directory:
- `log/run_tfce-{job_id}.out`
- `log/run_tfce-{job_id}.err`

## Requirements

### Python Dependencies

All dependencies are managed within the apptainer container, including:
- pandas
- numpy
- nibabel
- nilearn
- All other required packages

### Data Requirements

- First-level contrast images in NIfTI format
- Images must be in MNI space with consistent dimensions
- Valid file organization following the expected directory structure

## Methodology

### Threshold-Free Cluster Enhancement (TFCE)
The pipeline uses TFCE to identify significant activation clusters:
1. Performs one-sample t-tests across all subjects
2. Applies TFCE to enhance cluster-like signals without arbitrary thresholds
3. Uses non-parametric permutation testing for statistical inference
4. Generates corrected p-value maps accounting for multiple comparisons

### Statistical Analysis
- Uses nilearn's non_parametric_inference for robust statistics
- Performs two-sided tests with permutation-based correction
- Default 10,000 permutations for reliable p-value estimation
- Saves both TFCE-enhanced maps and raw t-statistic maps

## Troubleshooting

### Common Issues

1. **"Input directory does not exist"**: Verify the --data-dir path is correct
2. **"No contrast files found"**: Check task-name and contrast-name match your file structure
3. **"Images have inconsistent dimensions"**: Ensure all images are properly normalized to MNI space

### Checking Job Status

```bash
# Check job status
squeue -u $USER

# View job output
cat log/run_tfce-{job_id}.out

# View job errors
cat log/run_tfce-{job_id}.err
```

### Advanced Customization

To modify analysis parameters, you can:

1. **Edit default parameters** in `run_tfce.py`
2. **Modify SLURM settings** in `run_tfce.sh`
3. **Adjust class behavior** by modifying the respective class methods

## Example Workflow

```bash
# 1. Ensure container is available
./pull_image.sh

# 2. Run TFCE with required parameters
./run_tfce.sh user@stanford.edu --data-dir /path/to/first_level_outputs --task-name nBack --contrast-name twoBack-oneBack

# 3. Check results
ls results/

# 4. Run with custom parameters
./run_tfce.sh user@stanford.edu --data-dir /path/to/data --task-name nBack --contrast-name twoBack-oneBack --n-permutations 5000 --output-dir custom_results
```

## License

MIT License