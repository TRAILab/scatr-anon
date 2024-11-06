WORK_DIR=${PWD}
PROJECT=sparse4d-lidar
DOCKER_IMAGE=bcheong/${PROJECT}:latest
DOCKER_FILE=Docker/Dockerfile
DATA_ROOT_LOCAL_MINI=/media/brian/Data/nuscenes
DATA_ROOT_LOCAL=/media/brian/Data/nuscenes
WORK_DIR_LOCAL=/media/brian/Data/job_artifacts/Sparse4D-Lidar/work_dirs
CKPTS_ROOT_LOCAL=/media/brian/Data/ckpts/sparse4d

DATA_ROOT_APOLLO=/scratch/hpc_nas/datasets/nuscenes/v1.0-trainval
DATA_ROOT_APOLLO_MINI=/scratch/hpc_nas/datasets/nuscenes/v1.0-mini
OUTPUT_APOLLO=/home/bcheong/job_artifacts

DOCKER_OPTS = \
	-it \
	--rm \
	-e DISPLAY=${DISPLAY} \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-v /tmp:/tmp \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /mnt/fsx:/mnt/fsx \
	-v ~/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws \
	-v ${WORK_DIR}:/workspace/${PROJECT} \
	--shm-size=1G \
	--ipc=host \
	--network=host \
	--pid=host \
	--privileged \
	--gpus 'all,"capabilities=compute,utility,graphics"'

DOCKER_BUILD_ARGS = \
	--build-arg AWS_ACCESS_KEY_ID \
	--build-arg AWS_SECRET_ACCESS_KEY \
	--build-arg AWS_DEFAULT_REGION \
	--build-arg WANDB_ENTITY \
	--build-arg WANDB_API_KEY \

docker-build:
	docker image build \
	-f $(DOCKER_FILE) \
	-t $(DOCKER_IMAGE) \
	$(DOCKER_BUILD_ARGS) .

docker-dev-local-mini:
	docker run \
	--runtime=nvidia \
	--name $(PROJECT) \
	-v ${DATA_ROOT_LOCAL_MINI}:/workspace/${PROJECT}/data/nuscenes \
	-v ${WORK_DIR_LOCAL}:/workspace/${PROJECT}/work_dirs \
	-v ${CKPTS_ROOT_LOCAL}:/workspace/${PROJECT}/ckpts \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

docker-dev-local:
	docker run \
	--runtime=nvidia \
	--name $(PROJECT) \
	-v ${DATA_ROOT_LOCAL}:/workspace/${PROJECT}/data/nuscenes \
	-v ${WORK_DIR_LOCAL}:/workspace/${PROJECT}/work_dirs \
	-v ${CKPTS_ROOT_LOCAL}:/workspace/${PROJECT}/ckpts \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

docker-dev-apollo-mini:
	docker run \
	--runtime=nvidia \
	--name $(PROJECT) \
	-v ${DATA_ROOT_APOLLO_MINI}:/workspace/${PROJECT}/data/nuscenes \
	-v ${OUTPUT_APOLLO}:/workspace/${PROJECT}/work_dirs \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

docker-dev-apollo:
	docker run \
	--runtime=nvidia \
	--name $(PROJECT) \
	-v ${DATA_ROOT_APOLLO}:/workspace/${PROJECT}/data/nuscenes \
	-v ${OUTPUT_APOLLO}:/workspace/${PROJECT}/work_dirs \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

clean:
	find . -name '"*.pyc' | xargs sudo rm -f && \
	find . -name '__pycache__' | xargs sudo rm -rf