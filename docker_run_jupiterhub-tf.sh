cur_dir=$(basename $PWD)
docker run -it \
    -p 11888:8888 \
    -v "$PWD":"/home/jovyan/$cur_dir" \
    --memory "4g" \
    jupyter/tensorflow-notebook