#!/usr/bin/env bash

module load python/3.13.2
source ~/projects/rrg-swasland/cheongb2/ENV/bin/activate

HOST_DIR=$1 # Pass the host directory as the first argument to the script
echo "Host dir $HOST_DIR"
date
CONTAINER_CMD="wandb sync $HOST_DIR --verbose"
echo $CONTAINER_CMD
eval $CONTAINER_CMD
echo "Finished syncing $HOST_DIR"
