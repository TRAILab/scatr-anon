#!/usr/bin/env bash

SING_IMG=/home/$USER/projects/rrg-swasland/$USER/singularity/sparse4d-lidar-apptainer-0223.sif
SYNC_DIR=/home/$USER/job_artifacts/Sparse4D-L/artifacts/sparse4dv3-temporal_lidar_1x4_bs6-10e_lidar-narval-nogroup/wandb
BIND_DIR=/wandb
module load apptainer

while :; do
    date
    # echo "running wandb sync $RUN_DIR"
    CONTAINER_CMD="apptainer --silent exec --nv -e --pwd /
    --env "WANDB_API_KEY=$WANDB_API_KEY"
    --bind=$SYNC_DIR:$BIND_DIR
    $SING_IMG
    wandb sync $BIND_DIR/offline-run-20250227_124901-7jkhtp0u" # --project JDT3D --entity trailab --job_type cc_narval
    echo $CONTAINER_CMD
    eval $CONTAINER_CMD
    if [ $? -ne 0 ]; then
        echo "Error syncing $RUN_DIR"
    else
        echo "done syncing $RUN_DIR"
    fi
    echo "sleeping for 600 seconds"
    sleep 600
done
