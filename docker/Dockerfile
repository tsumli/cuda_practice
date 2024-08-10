FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

# install libraries
RUN apt update
RUN apt upgrade -y
RUN apt install -y \
    file \
    flex \
    git \
    wget \
    curl \
    cmake \
    software-properties-common \
    unzip \
    vim \
    ninja-build

# install gcc/gdb
RUN wget https://github.com/gcc-mirror/gcc/archive/refs/tags/releases/gcc-12.4.0.tar.gz &&\
    tar xvf gcc-12.4.0.tar.gz
WORKDIR /gcc-releases-gcc-12.4.0
RUN ./contrib/download_prerequisites
RUN mkdir build && \
    cd build && \
    ../configure --enable-languages=c,c++ --prefix=/usr/local --disable-bootstrap --disable-multilib && \
    make > /dev/null && \
    make install > /dev/null
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install --only-upgrade -y libstdc++6
RUN apt install -y gdb

# update cmake
# https://apt.kitware.com/
RUN apt-get update
RUN apt-get install ca-certificates gpg wget
RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | \
    tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | \
    tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update
RUN apt-get install -y cmake

# modify permission
ARG USER
ARG GROUP=${USER}
ARG UID=1001
ARG GID=1001
# Check if the group exists before creating it
RUN getent group ${GID} || groupadd -g ${GID} ${GROUP}
# Create the user 
RUN useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}

RUN echo "$USER:$GID" | chpasswd && \
    usermod --shell /bin/bash $USER && \
    usermod -aG sudo $USER && \
    mkdir /etc/sudoers.d && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER && \
    usermod  --uid $UID $USER && \
    groupmod --gid $GID $USER

RUN apt install -y sudo 
USER ${USER}
ENV HOME=/home/${USER}

WORKDIR /workspace
