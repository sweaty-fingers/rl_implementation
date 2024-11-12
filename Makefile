# SHELL := /bin/zsh
SHELL := /bin/zsh -l
# SHELL:=/bin/zsh
# Variables
WORKDIR := $(shell pwd)
DOCKERFILE_DIRNAME = ubuntu2204
DOCKERFILE = $(WORKDIR)/docker/Dockerfile

DOCKER_IMAGE_NAME = "rl_implement-$(DOCKERFILE_DIRNAME)"
DOCKER_CONTAINER_NAME = "rl_implement-$(DOCKERFILE_DIRNAME)-test"

WORKSPACE_DIR := $(shell pwd)
SAVE_DATA_DIR := /Users/yunhyeongeun/Documents/dataset/codes/rl_implement
SAVE_DATA_DIR_CONTAINER := /mnt/dataset

PORT_JUPYTER = 8891
PORT_STREAMLIT = 8892
PORT_TENSORBOARD = 8893
PORT_MLFLOW = 8894
PORT_SUB = 8895

# Phony targets
.PHONY: all build run stop clean

# Default target
all: build run
# Build the Docker image
build:
	@echo "Building Docker image..."
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f $(DOCKERFILE) . 

# Run the Docker container 
# -v $(DATASET_DIR):/dataset
run:
	@echo "Running Docker container..."
	docker run -it --name $(DOCKER_CONTAINER_NAME) -v $(WORKSPACE_DIR):/workspace -v $(SAVE_DATA_DIR):$(SAVE_DATA_DIR_CONTAINER) -p $(PORT_JUPYTER):$(PORT_JUPYTER) -p $(PORT_STREAMLIT):$(PORT_STREAMLIT) -p $(PORT_MLFLOW):$(PORT_MLFLOW) -p $(PORT_TENSORBOARD):$(PORT_TENSORBOARD) -p $(PORT_SUB):$(PORT_SUB) $(DOCKER_IMAGE_NAME)

# Stop the Docker container

# 종료된 docker container 재실헹
start:
	@echo "Start Docker container..."
	sudo docker start $(DOCKER_CONTAINER_NAME)
# Docker container 종료 및 삭제
stop:
	@echo "Stopping Docker container..."
	sudo docker stop $(DOCKER_CONTAINER_NAME) || true
	sudo docker rm $(DOCKER_CONTAINER_NAME) || true

# 살행 중인 docker container에 접속
attach:
	@echo "Attach Docker container..."
	sudo docker exec -it $(DOCKER_CONTAINER_NAME) /bin/zsh

# Clean up the Docker image and container
clean: stop
	@echo "Cleaning up Docker image and container..."
	docker rmi $(DOCKER_IMAGE_NAME) || true
	rm -rf $(WORKSPACE_DIR)

# Build and run the Docker container
rebuild: clean build run

jupyter-notebook-pyenv:
	@echo "Make jupter kernel and start notebook"
	python -m ipykernel install --user --name=$(shell pyenv version-name) --display-name $(shell pyenv version-name)
	jupyter notebook --allow-root --port=$(PORT_JUPYTER) --no-browser --ip=0.0.0.0

jupyter-notebook:
	@echo "Make jupter kernel and start notebook"
	poetry shell
	poetry run python -m ipykernel install --user --name=$(shell echo $$(poetry env info --path | sed 's|.*/||')) --display-name "python (poetry)"
	jupyter notebook --allow-root --port=$(PORT_JUPYTER) --no-browser --ip=0.0.0.0

streamlit-run:
	@echo "Run Streamlit dashboard"
	streamlit run dashboard/app/main.py --server.port $(PORT_STREAMLIT) --server.address 0.0.0.0