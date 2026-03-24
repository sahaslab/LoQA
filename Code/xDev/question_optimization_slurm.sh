#!/bin/bash -l
#SBATCH --job-name=qo
#SBATCH --output=Outputs/slurm_logs/question_optimization_%j.out
# # SBATCH --error=Outputs/slurm_logs/question_optimization_%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# # SBATCH --mail-type=END,FAIL
# # SBATCH --mail-user=omar.sharif.gr@dartmouth.edu

echo "=========================================="
echo "  Question Optimization - SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Change to repo directory (run sbatch from repo root)
REPO_ROOT="/dartfs-hpc/rc/home/j/f006f3j/lab/omar/LoQA"
cd "$REPO_ROOT" || exit 1

# Create logs directory if needed
mkdir -p Outputs/slurm_logs

# Load environment (API keys, etc.)
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
else
    echo "WARNING: .env file not found"
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate loqaGpu

# Print environment info (helpful for debugging)
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo ""

# Run the question optimization script
bash Code/scripts/question_optimization.sh

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
