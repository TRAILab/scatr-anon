#!/usr/bin/env bash

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