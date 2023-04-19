FROM determinedai/environments:cuda-11.3-pytorch-1.10-tf-2.8-gpu-0.19.4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && \
  apt-get install -y \
  libgl1-mesa-glx \
  libsm6 \
  libxext6 \
  sudo \
  vim \
  wget \
  unzip \
  net-tools &&\
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


# Only add the library shared by local and determined training.
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    deepdish==0.3.7 \
    omegaconf==2.2.3 \
    scikit-learn==0.24.2 \
    pytest==6.2.5 \
    determined==0.19.4 \
    loguru==0.6.0 \
    h5py==3.7.0 \
    opencv-contrib-python==4.5.5.62 \
    nvidia-pyindex==1.0.9 \
    opencv-contrib-python==4.5.5.62 \
    albumentations==1.3.0 \
    prettytable==3.4.1 \
    kornia==0.6.8 \
    pose-transform==0.3.1 \
    setuptools==59.5.0 \
    accelerate==0.12.0 \
    pycocotools==2.0.6

RUN pip install -i  https://developer.download.nvidia.com/compute/redist \
    nvidia-dali-cuda110==1.20.0

ENV DET_MASTER="https://determined.corp.deepmirror.com:443"
