#!/usr/bin/env bash

SING_IMG=/home/$USER/projects/rrg-swasland/$USER/singularity/sparse4d-lidar-apptainer-0223.sif
SYNC_DIR=/home/$USER/job_artifacts/Sparse4D-L/artifacts
BIND_DIR=/wandb
module load apptainer/1.3.5

while :; do
    date
    # Sort experiment dirs (wandb subdirs) from newest to oldest
    for EXPERIMENT_DIR in $(ls -td "$SYNC_DIR"/*/wandb 2>/dev/null); do
        if [ -d "$EXPERIMENT_DIR" ]; then
            # Sort run dirs from newest to oldest
            for RUN_DIR in $(ls -td "$EXPERIMENT_DIR"/* 2>/dev/null); do
                if [ -d "$RUN_DIR" ]; then
                    echo "Syncing $RUN_DIR"
                    CONTAINER_CMD="apptainer --silent exec --nv -e --pwd /
                    --env \"WANDB_API_KEY=$WANDB_API_KEY\"
                    --bind=$RUN_DIR:$BIND_DIR
                    $SING_IMG
                    wandb sync $BIND_DIR" # --project JDT3D --entity trailab --job_type cc_narval
                    echo $CONTAINER_CMD
                    eval $CONTAINER_CMD
                    if [ $? -ne 0 ]; then
                        echo "Error syncing $RUN_DIR"
                    else
                        echo "done syncing $RUN_DIR"
                    fi
                fi
            done
        fi
    done
    echo "sleeping for 600 seconds"
    sleep 600
done
