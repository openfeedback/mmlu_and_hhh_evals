#!/usr/bin/bash
#SBATCH --job-name=run_mmlu_eval
#SBATCH --output=batch_outputs/test_job.%j.out
#SBATCH --error=batch_outputs/test_job.%j.err
#SBATCH --account=nlp
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=sphinx8
#SBATCH --partition=sphinx

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

export NCCL_P2P_DISABLE=1 # look at https://github.com/microsoft/DeepSpeed/issues/2176
export HF_DATASETS_CACHE="/nlp/scr/fongsu/.cache"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate superhf
python hhh_create_dataset.py
python run_hhh_eval.py --models \
peterchatain/rlhf-v3-lr-5.0e-6-batch-16@gold-run
