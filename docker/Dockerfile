# Use the official NVIDIA CUDA base image with Ubuntu 22.04
#FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
FROM ubuntu:22.04
# FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

# Set environment variables to non-interactive for automatic installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
    nano python3-pip python3-mock libpython3-dev \
    libpython3-all-dev python-is-python3 wget curl cmake \
    software-properties-common sudo pkg-config libhdf5-dev \
    python3 \
    python3-dev \
    build-essential \
    vim \
    git \
    libopenblas-dev \
    libomp-dev \
    zsh \
    unzip\
    fonts-powerline \
    make \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    lsof\
    tree\
    xvfb\
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set the powerlevel10k theme
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/themes/powerlevel10k \
    && sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' /root/.zshrc

# Install pyenv and pyenv-virtualenv
RUN curl https://pyenv.run | zsh
# Set environment variables for pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
# Set up pyenv in .zshrc
RUN echo 'export PYENV_ROOT="/root/.pyenv"' >> /root/.zshrc \
    && echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.zshrc \
    && echo 'eval "$(pyenv init --path)"' >> /root/.zshrc \
    && echo 'eval "$(pyenv init -)"' >> /root/.zshrc \
    && echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.zshrc \
    && echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.zshrc

# Install Python 3.10.12 using pyenv
RUN pyenv install 3.10.12
RUN pyenv global 3.10.12

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and poetry.lock to the container
COPY pyproject.toml /workspace/

# Set the working directory inside the container
WORKDIR /workspace

# Install Python packages using poetry
# RUN poetry install
# RUN zsh -c "source /root/.zshrc && pyenv global 3.10.12 && poetry env use 3.10.12"

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH
ENV PATH="/workspace/scripts:${PATH}"

ENV PYTHONPATH=/workspace
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"
RUN echo "bindkey -v" >> /root/.zshrc

# (Optional) Add your own Python scripts or other files here
# COPY your_script.py /workspace/
# Set the default command to bash
# CMD ["bash"]
# Set the default command to run when starting the container
CMD ["zsh"]