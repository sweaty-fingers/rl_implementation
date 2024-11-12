#!/bin/zsh
alias docker="/Applications/Docker.app/Contents/Resources/bin/docker"
# 변수 설정
WORKDIR=$(pwd)
DOCKERFILE_DIRNAME="ubuntu2204"
DOCKERFILE="${WORKDIR}/Dockerfile"

DOCKER_IMAGE_NAME="rl_implement-${DOCKERFILE_DIRNAME}"
DOCKER_CONTAINER_NAME="rl_implement-${DOCKERFILE_DIRNAME}"

WORKSPACE_DIR=$(pwd)
SAVE_DATA_DIR="/Users/yunhyeongeun/Documents/dataset/rl_implement"
SAVE_DATA_DIR_CONTAINER="/mnt/dataset"

PORT_JUPYTER=8891
PORT_STREAMLIT=8892
PORT_TENSORBOARD=8893
PORT_MLFLOW=8894
PORT_SUB=8895

# 빌드 함수
build_image() {
    echo "Building Docker image..."
    docker build -t "${DOCKER_IMAGE_NAME}" -f "${DOCKERFILE}" .
}

# 컨테이너 실행 함수
run_container() {
    echo "Running Docker container..."
    docker run -it --name "${DOCKER_CONTAINER_NAME}" \
        -v "${WORKSPACE_DIR}":/workspace \
        -v "${SAVE_DATA_DIR}":"${SAVE_DATA_DIR_CONTAINER}" \
        -p "${PORT_JUPYTER}":"${PORT_JUPYTER}" \
        -p "${PORT_STREAMLIT}":"${PORT_STREAMLIT}" \
        -p "${PORT_MLFLOW}":"${PORT_MLFLOW}" \
        -p "${PORT_TENSORBOARD}":"${PORT_TENSORBOARD}" \
        -p "${PORT_SUB}":"${PORT_SUB}" \
        "${DOCKER_IMAGE_NAME}"
}

# 컨테이너 시작 함수
start_container() {
    echo "Starting Docker container..."
    docker start "${DOCKER_CONTAINER_NAME}"
}

# 컨테이너 중지 및 삭제 함수
stop_container() {
    echo "Stopping Docker container..."
    docker stop "${DOCKER_CONTAINER_NAME}" || true
    docker rm "${DOCKER_CONTAINER_NAME}" || true
}

# 컨테이너 접속 함수
attach_container() {
    echo "Attaching to Docker container..."
    docker exec -it "${DOCKER_CONTAINER_NAME}" /bin/zsh
}

# 이미지 및 컨테이너 정리 함수
clean_up() {
    stop_container
    echo "Cleaning up Docker image and container..."
    docker rmi "${DOCKER_IMAGE_NAME}" || true
    rm -rf "${WORKSPACE_DIR}"
}

# 이미지 재빌드 및 컨테이너 실행 함수
rebuild_container() {
    clean_up
    build_image
    run_container
}

# 메인 함수
case $1 in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    attach)
        attach_container
        ;;
    clean)
        clean_up
        ;;
    rebuild)
        rebuild_container
        ;;
    all)
        build_image
        run_container
        ;;
    *)
        echo "Usage: $0 {build|run|start|stop|attach|clean|rebuild|all}"
        ;;
esac