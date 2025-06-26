#!/bin/bash
#SBATCH --array=1-1%1     # Step size of 2, 1,3,5,7
#SBATCH --job-name=sparse4d-L-baseline    # Job name
#SBATCH --account=rrg-swasland
#SBATCH --ntasks=1                    # number of MPI processes
#SBATCH --mem=256G                     # Job CPU memory request
#SBATCH --time=60:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/cheongb2/projects/rrg-swasland/cheongb2/job_artifacts/Sparse4D-L/slurm_logs/%x-%j.log   # Standard output and error log
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4           # gpu:t4:4 (graham) or gpu:a100:1 (narval)
#SBATCH --mail-user="g1j5i4u0v5b4y4x4@trail-utias.slack.com"
#SBATCH --mail-type=ALL

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."

echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"

# Parameters
DATASET=nuscenes # nuscenes_mini
NUM_GPUS=4

# Host paths
HOME_DIR=/home/$USER
TMP_DATA_DIR=$SLURM_TMPDIR/data
# TMP_DATA_DIR=/home/$USER/scratch/temp_data # Slurm unzip alternative
PROJ_DIR=$HOME_DIR/repos/Sparse4D-LiDAR-mirror
OUT_DIR=$HOME_DIR/projects/rrg-swasland/$USER/job_artifacts/Sparse4D-L/work_dirs/
SING_IMG=/home/$USER/projects/rrg-swasland/$USER/singularity/sparse4d-lidar-apptainer-0223.sif
DATA_DIR=/home/$USER/projects/rrg-swasland/$USER/nuscenes # use a symlink to the actual data, may be different on each server
PKL_DIR=/home/$USER/projects/rrg-swasland/$USER/nuscenes_pkls/sparse4dL
CKPT_DIR=/home/$USER/projects/rrg-swasland/$USER/ckpts/sparse4d

mkdir -p $OUT_DIR
mkdir -p $WANDB_ARTIFACT_DIR
mkdir -p $WANDB_DATA_DIR
mkdir -p $WANDB_CACHE_DIR

# Container paths
# THERE SHOULD BE NO SPACES AFTER THE \
PROJECT_NAME=sparse4d-l
CONTAINER_PATH=/workspace/$PROJECT_NAME # path to main workspace
VOLUMES="--bind=$PROJ_DIR:$CONTAINER_PATH \
--bind=$TMP_DATA_DIR:$CONTAINER_PATH/data/nuscenes \
--bind=$OUT_DIR:$CONTAINER_PATH/work_dirs \
--bind=$WANDB_ARTIFACT_DIR:$WANDB_ARTIFACT_DIR \
--bind=$WANDB_DATA_DIR:$WANDB_DATA_DIR \
--bind=$WANDB_CACHE_DIR:$WANDB_CACHE_DIR \
--bind=$SLURM_TMPDIR:/tmp \
--bind=$CKPT_DIR:$CONTAINER_PATH/ckpts
"
CFG_FILE=projects/configs/sparse4dv3-temporal_lidar.py

# Command
WANDB_MODE='offline'
BASE_CMD="bash ./tools/dist_train.sh $CFG_FILE $NUM_GPUS"
CONTAINER_CMD="apptainer exec --nv -c -e --writable-tmpfs --pwd $CONTAINER_PATH \
--env "WANDB_API_KEY=$WANDB_API_KEY"
--env "WANDB_MODE=$WANDB_MODE"
--env "CUDA_LAUNCH_BLOCKING=1"
--env "TORCH_USE_CUDA_DSA=1"
--env "TORCH_NCCL_ENABLE_MONITORING=0"
--env "WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR"
--env "WANDB_DATA_DIR=$WANDB_DATA_DIR"
--env "WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
$VOLUMES \
$SING_IMG \
$BASE_CMD
"

# Start script
SECONDS=0
echo "SLURM_JOB_ID=$SLURM_JOB_ID
CFG_FILE=$CFG_FILE
NUM_GPUS=$NUM_GPUS
"

# Extract dataset
echo "Extracting data"
mkdir $TMP_DATA_DIR
if [ "$DATASET" = "nuscenes_mini" ]; then
    nuscenes_zips=()
    nuscenes_pkls=(
        "nuscenes_sparse4d_mmlabv2_11-18_mini_infos_train.pkl" \
        "nuscenes_sparse4d_mmlabv2_11-18_mini_infos_val.pkl" \
        "nuscenes_track_dbinfos_train.pkl"
    )
    nuscenes_tgz=(
        "v1.0-mini.tgz" \
        "nuscenes_track_gt_database.tar.gz"
    )
fi
if [ "$DATASET" = "nuscenes" ]; then
    nuscenes_zips=("sweeps.zip" "samples.zip" "v1.0-trainval.zip" "lidarseg.zip" "maps.zip")
    nuscenes_pkls=(
        "nuscenes_sparse4d_mmlabv2_11-06_infos_train.pkl" \
        "nuscenes_sparse4d_mmlabv2_11-06_infos_val.pkl" \
        "nuscenes_track_dbinfos_train.pkl")
    nuscenes_tgz=(
        "nuscenes_track_gt_database.tar.gz"
    )
fi

for file in "${nuscenes_zips[@]}"; do
    duration=$SECONDS
    echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
    unzip -qq $DATA_DIR/$file -d $TMP_DATA_DIR
done
for file in "${nuscenes_pkls[@]}"; do
    duration=$SECONDS
    echo "[$((duration/3600))h$(((duration%3600)/60))m]: Copying $file to $TMP_DATA_DIR"
    cp $PKL_DIR/$file $TMP_DATA_DIR
done
for file in "${nuscenes_tgz[@]}"; do
    duration=$SECONDS
    echo "[$((duration/3600))h$((duration%3600/60))m]: Unzipping $file to $TMP_DATA_DIR"
    tar -xf $DATA_DIR/$file -C $TMP_DATA_DIR
done

duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done extracting data"

# Run command
# echo "Debug mode: sleep engaged" && sleep 5d
module load StdEnv/2020 apptainer
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Running command"
echo "$CONTAINER_CMD"
eval $CONTAINER_CMD
duration=$SECONDS
echo "[$((duration/3600))h$(((duration%3600)/60))m]: Done"