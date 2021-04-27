sudo docker build - < ../docker/Dockerfile2

sudo docker run -w /home/peterahn/Workspace/substruct-emebedding/src -v /home/peterahn/Workspace/substruct-emebedding:/home/peterahn/Workspace/substruct-emebedding --gpus all -i -t lg