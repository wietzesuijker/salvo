#!/bin/bash
#SBATCH --job-name=cpu_basic
#SBATCH --account=mila
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384M
#SBATCH --time=60
#SBATCH --export=NONE
#SBATCH --output=/tmp/.salvo/runs/JOBID/stdout.log
#SBATCH --error=/tmp/.salvo/runs/JOBID/stderr.log
#SBATCH --signal=SIGUSR1@90

set -euo pipefail
export SALVO_ARTIFACT_DIR=/tmp/.salvo/runs/JOBID
export SALVO_HOP=${SALVO_HOP:-0}/5
if [[ "${SLURM_JOB_ID:-}" != "" ]]; then
  trap 'python -m salvo.runtime.epilog --artifact-dir "$SALVO_ARTIFACT_DIR" --reason preempt' SIGUSR1
fi
python -c 'print(1)'
