#!/bin/bash

#SBATCH --job-name=mnist_mini_batch
#SBATCH --output=slurm/slurm-%x-%A_%a.out
#SBATCH --time=1-00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p besteffort
# SBATCH --nodelist=titanic-1
# SBATCH --array=[1-2]


if [ "${SLURM_ARRAY_JOB_ID}" ] ; then
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    JOB_ID="${SLURM_JOB_ID}"
fi
echo -e "JOB ID = ${JOB_ID}"
echo -e "NODE NAME = ${SLURMD_NODENAME}"

python src/gromo/main_graph_network.py --exp_name MNIST --neurons 10 --iters 50 --job_id "${JOB_ID}" --node_name "${SLURMD_NODENAME}"
