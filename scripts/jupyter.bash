#!/usr/bin/env bash

if [ "$DOCKER_RUNNING" == true ] 
then
    echo "Inside docker instance"
    jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
    
else
    echo "Starting up docker instance..."

    cmp_volumes="--volume=$(pwd):/app/:rw"

    docker run --rm -ti \
        $cmp_volumes \
        -it \
        --gpus all \
        --ipc host \
        -p 8889:8889 \
        adrianorenstein/hypertension \
        jupyter-lab --ip 0.0.0.0 --port 8889 --no-browser --allow-root
fi