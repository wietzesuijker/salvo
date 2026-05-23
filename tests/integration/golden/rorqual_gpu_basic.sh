#!/bin/bash
#SBATCH --job-name=gpu_basic
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=32768M
#SBATCH --time=120
#SBATCH --export=NONE
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/tmp/.salvo/runs/JOBID/stdout.log
#SBATCH --error=/tmp/.salvo/runs/JOBID/stderr.log
#SBATCH --signal=SIGUSR1@90

set -euo pipefail
export SALVO_ARTIFACT_DIR=/tmp/.salvo/runs/JOBID
export SALVO_HOP=${SALVO_HOP:-0}/5
if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
  trap 'python -m salvo.runtime.epilog --artifact-dir "$SALVO_ARTIFACT_DIR" --reason preempt' SIGUSR1
fi
python train.py --seed 1
