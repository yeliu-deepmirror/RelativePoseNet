#!/bin/bash

# The directory contains current script.
DIR=$(dirname $(realpath "$BASH_SOURCE"))
# The directory contains .git directory.
REPO_DIR=$DIR/../../

# Start qemu emulator
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

ARCH="amd64"
echo "build" $ARCH "image"
docker build -t relative-pose-net-dev \
    -f $REPO_DIR/artifacts/docker/dev.dockerfile \
    --build-arg ARCH=$ARCH \
    $REPO_DIR/artifacts/docker
