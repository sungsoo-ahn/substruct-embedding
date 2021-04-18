import random
import numpy as np
import torch
from torch_geometric.data import Data
import faiss

from model import GraphEncoder
from util import compute_accuracy

class SharpeningGraphClusteringScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, num_clusters, transform, temperature):
        self._transform = transform
        self.temperature = temperature
        self.num_clusters = num_clusters
        self.num_neg_protos = 16000
        self.centroids = None
        self.g2cluster = None
        self.density = None
        
    @staticmethod
    def collate_fn(data_list):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = Data()
        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_num_nodes = []
        batch.batch_size = len(data_list)
        batch.num_views = 2

        cumsum_node = 0
        for i, data in enumerate(data_list):
            num_nodes = data.x.size(0)
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.batch_num_nodes.append(torch.LongTensor([num_nodes]))

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node

                batch[key].append(item)

            cumsum_node += num_nodes

        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        return batch.contiguous()

    def transform(self, data):
        x, edge_index, edge_attr = self._transform(
            data.x.clone(), data.edge_index.clone(), data.edge_attr.clone()
        )
        if x.size(0) == 0:
            return None

        data0 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data0.dataset_idx = data.dataset_idx

        return data0

    def get_models(self, num_layers, emb_dim, drop_rate):
        encoder = GraphEncoder(num_layers, emb_dim, drop_rate)
        head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim),
        )
        models = torch.nn.ModuleDict({"encoder": encoder, "head": head})
        return models

    def assign_cluster(self, loader, models, device):
        models.eval()
        features = torch.zeros(len(loader.dataset), models["encoder"].emb_dim).to(device)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = models["encoder"](
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                features[batch.dataset_idx] = models["head"](out)
                
        features = features.cpu().numpy()        
        
        d = features.shape[1]
        clus = faiss.Clustering(d, self.num_clusters)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = 0
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = device
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(features, index)

        # for each sample, find cluster distance and assignments
        D, I = index.search(features, 1)
        g2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(self.num_clusters, d)

        # convert to cuda Tensors for broadcast
        self.centroids =  torch.Tensor(centroids).to(device)        
        self.g2cluster = torch.LongTensor(g2cluster).to(device)
        
        obj = clus.iteration_stats.at(clus.iteration_stats.size()-1).obj
        bincount = torch.bincount(self.g2cluster)
        statistics = {"obj": obj, "bincount": bincount}
        
        return statistics
        
    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)

        out = models["encoder"](batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        features = models["head"](out)
        proto_logits = -torch.cdist(features, self.centroids, p=2) ** 2
        proto_probs = torch.softmax(proto_logits, dim=1)
        proto_labels = self.g2cluster[batch.dataset_idx]
        loss_proto = -torch.mean(torch.sum(proto_probs * torch.log(proto_probs), dim=1))
        acc_proto = compute_accuracy(proto_logits, proto_labels) 
        loss = loss_proto

        optim.zero_grad()
        loss.backward()
        optim.step()

        statistics = {
            "loss": loss,
            "loss_proto": loss_proto.detach(),
            "acc_proto": acc_proto,
            }
        
        return statistics