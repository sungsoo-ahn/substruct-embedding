FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN conda install -y tqdm
RUN conda install -y -c conda-forge neptune-client
RUN conda install -y -c conda-forge rdkit
RUN conda install -y -c conda-forge scikit-learn 

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
RUN pip install torch-geometric
RUN pip install faiss-gpu