#!/usr/bin/env bash

SING_IMG=/home/$USER/projects/rrg-swasland/$USER/singularity/jdt3d.sif
SYNC_DIR=/home/$USER/job_artifacts/PF-Track2/train_jdt3d_f1_e10_d5/20240921_052650/vis_data/wandb
BIND_DIR=/wandb
module load apptainer

while :; do
    date
    # echo "running wandb sync $RUN_DIR"
    CONTAINER_CMD="apptainer --silent exec --nv -e --pwd /
    --env "WANDB_API_KEY=$WANDB_API_KEY"
    --bind=$SYNC_DIR:$BIND_DIR
    $SING_IMG
    wandb sync $BIND_DIR/offline-run-20240921_052702-8cg9n97u" # --project JDT3D --entity trailab --job_type cc_narval
    echo $CONTAINER_CMD
    eval $CONTAINER_CMD
    if [ $? -ne 0 ]; then
        echo "Error syncing $RUN_DIR"
    else
        echo "done syncing $RUN_DIR"
    fi
    echo "sleeping for 600 seconds"
    break
    sleep 600
done
