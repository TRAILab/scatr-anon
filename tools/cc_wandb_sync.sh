#!/usr/bin/env bash

SING_IMG=/home/$USER/projects/rrg-swasland/$USER/singularity/sparse4d-lidar-apptainer-0223.sif
# HOST_DIR=/home/cheongb2/job_artifacts/Sparse4D-L/artifacts/sparse4dv3-temporal_lidar_1x4_bs6-10e_wacv-narval-lidar-4g-topk_rand-qc/wandb/offline-run-20250616_170243-o3ojwr0m/
HOST_DIR=$1 # Pass the host directory as the first argument to the script
CONTAINER_DIR=/wandb

module load StdEnv/2020 apptainer

date
echo "running wandb sync $HOST_DIR"
CONTAINER_CMD="apptainer --silent exec --nv -c -e --pwd /
--env "WANDB_API_KEY=$WANDB_API_KEY"
--bind=$HOST_DIR/:$CONTAINER_DIR/
$SING_IMG
wandb sync $CONTAINER_DIR"
eval $CONTAINER_CMD
echo "Finished syncing $HOST_DIR"