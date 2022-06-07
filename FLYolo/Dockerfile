FROM carlasim/carla:0.9.10 as carla
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ARG MIRROR='https://mirrors.aliyun.com'
ARG PYPI_MIRROR='https://mirrors.aliyun.com/pypi/simple'
RUN sed -i '/bintray/d' /etc/apt/sources.list && \
    sed -i "s@http://.*archive.ubuntu.com@$MIRROR@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@$MIRROR@g" /etc/apt/sources.list && \  
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev python3-pip \
        git \ 
        vim \
        curl \
        sudo \
        wget \
        libgl1-mesa-glx && \
    pip3 install --no-cache-dir pip -U -i ${PYPI_MIRROR} --trusted-host ${MIRROR} && \
    pip3 config set global.index-url ${PYPI_MIRROR} && \
    pip3 config set install.trusted-host ${MIRROR} && \
    rm -rf /var/lib/apt/lists/* 

ENV TZ='Asia/Shanghai'
RUN echo ${TZ} > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*


RUN mkdir -p /home/carla && \
    useradd -m carla

COPY --from=carla --chown=root /home/carla/PythonAPI /home/carla/PythonAPI

WORKDIR /home/carla/PythonAPI

# RUN git clone https://github.com/zijianzhang/CARLA_INVS.git --branch=main --depth=1
COPY --chown=root . /home/carla/PythonAPI/CARLA_INVS

WORKDIR /home/carla/PythonAPI/CARLA_INVS

RUN apt update && \
    apt-get install -y --no-install-recommends \
        libxerces-c3.2 libjpeg8 && \
    pip3 install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*
 
WORKDIR /home/carla/PythonAPI/CARLA_INVS/PCDet

# Pre-setup OpenPCDet build environment
RUN apt update && \
    apt-get install -y --no-install-recommends \
        libboost-all-dev ninja-build && \
    pip3 install cmake --upgrade && \
    pip3 install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*


# sed -i "s@\${CUDNN_INCLUDE_PATH}/cudnn.h@\${CUDNN_INCLUDE_PATH}/cudnn_version.h@g" \
        # /usr/local/lib/python3.8/dist-packages/torch/share/cmake/Caffe2/public/cuda.cmake && \
# PROBLEM_FILE=/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake && \
# sed -i 's/-Wall;-Wextra;-Wno-unused-parameter;-Wno-missing-field-initializers;-Wno-write-strings;-Wno-unknown-pragmas;-Wno-missing-braces;-fopenmp//g' $PROBLEM_FILE && \
# sed -i 's/-Wall;-Wextra;-Wno-unused-parameter;-Wno-missing-field-initializers;-Wno-write-strings;-Wno-unknown-pragmas;-Wno-missing-braces//g' $PROBLEM_FILE && \  

# Build SPCONV
RUN sed -i "s@AT_CHECK@TORCH_CHECK@g" pcdet/ops/iou3d_nms/src/iou3d_nms.cpp && \
    sed -i "s@AT_CHECK@TORCH_CHECK@g" pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp && \
    git clone https://github.com/traveller59/spconv.git build/spconv --recursive && \
	cd build/spconv && \
    python3 setup.py bdist_wheel && \
    cd dist && \ 
    pip3 install *.whl 

# Build and Install PCDet
RUN python3 setup.py develop

RUN chown -R carla:carla /home/carla

RUN echo "carla:carla" | chpasswd && \
    usermod -aG sudo carla

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt-get install -y --no-install-recommends \
        libsdl2-2.0 xserver-xorg libvulkan1 && \
    pip3 install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*    

USER carla

RUN sed -i "s@~/CARLA_0.9.10@/home/carla@g" /home/carla/PythonAPI/CARLA_INVS/params.py

WORKDIR  /home/carla/PythonAPI/CARLA_INVS
# export PYTHONPATH=$PYTHONPATH:/home/carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Example: 
# docker run -it --rm --network=host --gpus all --runtime=nvidia -e DISPLAY harbor.isus.tech/carla1s/carla-invs:0.9.10 bash