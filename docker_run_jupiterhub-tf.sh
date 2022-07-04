#!/bin/bash

cur_dir=$(basename $PWD)

if which podman > /dev/null; then
    DOCKER_TOOL=podman
else
    DOCKER_TOOL=docker
fi

echo "Using $DOCKER_TOOL"
$DOCKER_TOOL run -it --rm \
    -p 11888:8888 \
    -v "$PWD":"/home/jovyan/$cur_dir" \
    docker.io/jupyter/tensorflow-notebook 
