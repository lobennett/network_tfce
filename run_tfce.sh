#!/bin/bash

# Usage: run_tfce.sh [email] [additional args for run_tfce.py]
# Example: run_tfce.sh user@email.com --data-dir /path/to/data --task-name nBack --contrast-name twoBack-oneBack

# Get script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
EMAIL="${1:-logben@stanford.edu}"
shift  # Remove email from args, pass rest to run_tfce.py

# Create log directory
mkdir -p "${SCRIPT_DIR}/log"

# Check if apptainer image exists
APPTAINER_IMAGE="$SCRIPT_DIR/apptainer/fmri_env_latest.sif"
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image not found: $APPTAINER_IMAGE"
    echo "Please run the pull_image.sh script first to download the required image:"
    echo "  $SCRIPT_DIR/pull_image.sh"
    exit 1
fi

echo "Using apptainer image: $APPTAINER_IMAGE"
echo "Submitting TFCE analysis job to SLURM..."

# Submit single TFCE job to SLURM
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=run_tfce
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=russpold,hns,normal
#SBATCH --output=${SCRIPT_DIR}/log/%x-%j.out
#SBATCH --error=${SCRIPT_DIR}/log/%x-%j.err
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=END

echo "Starting TFCE analysis..."
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Working directory: \$PWD"

# Change to script directory
cd "$SCRIPT_DIR"

# Run TFCE analysis using apptainer
apptainer exec "$APPTAINER_IMAGE" python3 run_tfce.py $@

echo "Completed TFCE analysis"
EOF