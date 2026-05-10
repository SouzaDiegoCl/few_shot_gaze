FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG VIDEO_GID=39
ARG RENDER_GID=105

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    gfortran \
    git \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgtk2.0-0 \
    liblapack-dev \
    libopenblas-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    software-properties-common \
    python3-dev \
    python3-venv \
    unzip \
    v4l-utils \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt-get update \
 && apt-get install -y --no-install-recommends gcc-7 g++-7 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -g ${GROUP_ID} demo \
 && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash demo \
 && (getent group video >/dev/null && groupmod -g ${VIDEO_GID} video || groupadd -g ${VIDEO_GID} -o video) \
 && (getent group render >/dev/null && groupmod -g ${RENDER_GID} render || groupadd -g ${RENDER_GID} -o render) \
 && usermod -aG video,render demo

WORKDIR /work

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir cmake==3.18.4.post1

RUN python -m pip install --no-cache-dir \
    numpy==1.19.5 \
    h5py==2.10.0 \
    imageio==2.8.0 \
    moviepy==1.0.1 \
    opencv-python==4.2.0.34 \
    'tqdm>=4.40.0' \
    yacs \
    pandas==0.24.2 \
    scipy==1.0.0 \
    hdf5storage \
    Pillow

RUN CC=/usr/bin/gcc-7 CXX=/usr/bin/g++-7 python -m pip install --no-cache-dir eos-py==1.1.2

COPY demo/calibrate_camera.py /work/demo/
COPY demo/camera.py /work/demo/
COPY demo/face.py /work/demo/
COPY demo/frame_processor.py /work/demo/
COPY demo/head.py /work/demo/
COPY demo/KalmanFilter1D.py /work/demo/
COPY demo/landmarks.py /work/demo/
COPY demo/monitor.py /work/demo/
COPY demo/normalization.py /work/demo/
COPY demo/person_calibration.py /work/demo/
COPY demo/run_demo.py /work/demo/
COPY demo/undistorter.py /work/demo/
COPY demo/ext/eos/share/ /work/demo/ext/eos/share/
COPY demo/ext/HRNet-Facial-Landmark-Detection/lib/ /work/demo/ext/HRNet-Facial-Landmark-Detection/lib/
COPY demo/ext/HRNet-Facial-Landmark-Detection/experiments/wflw/face_alignment_wflw_hrnet_w18.yaml /work/demo/ext/HRNet-Facial-Landmark-Detection/experiments/wflw/
COPY demo/ext/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained/HR18-WFLW.pth /work/demo/ext/HRNet-Facial-Landmark-Detection/hrnetv2_pretrained/
COPY demo/ext/mtcnn-pytorch/src/ /work/demo/ext/mtcnn-pytorch/src/
COPY src/models/ /work/src/models/
COPY src/losses/ /work/src/losses/

WORKDIR /work/demo

RUN wget -q https://files.ait.ethz.ch/projects/faze/demo_weights.zip \
 && unzip -o demo_weights.zip \
 && rm demo_weights.zip \
 && chown -R demo:demo /work

USER demo

CMD ["/bin/bash"]