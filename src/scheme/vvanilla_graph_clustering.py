import random
import numpy as np
import torch
from torch_geometric.data import Data
import faiss

from model import GraphEncoder
from util import compute_accuracy

class VVanillaGraphClusteringScheme:
    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, num_clusters, transform, temperature):
        self._transform = transform
        self.temperature = temperature
        self.num_clusters = num_clusters
        self.num_neg_protos = 16000
        self.centroids = None
        self.g2cluster = None
        self.density = None
        self.cluster_temperature = 0.2
        
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
            
            features = torch.nn.functional.normalize(features, p=2, dim=1)
                
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

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(self.num_clusters)]
        for im, i in enumerate(g2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(self.num_clusters)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        # clamp extreme values for stability
        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))

        # scale the mean to temperature
        density = self.cluster_temperature * density / density.mean()

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(device)
        self.centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

        self.g2cluster = torch.LongTensor(g2cluster).to(device)
        self.density = torch.Tensor(density).to(device)
        
        obj = clus.iteration_stats.at(clus.iteration_stats.size()-1).obj
        bincount = torch.bincount(self.g2cluster)
        statistics = {"obj": obj, "bincount": bincount}
        return statistics
        
    def train_step(self, batch, models, optim, device):
        models.train()
        batch = batch.to(device)

        out = models["encoder"](batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out = models["head"](out)
        features = torch.nn.functional.normalize(out, p=2, dim=1)

        proto_logits, proto_labels = self.get_proto_logits_and_labels(
            batch.dataset_idx, features, device
            )

        loss_proto = self.criterion(proto_logits, proto_labels)
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

    def get_proto_logits_and_labels(self, dataset_idx, features, device):
        # get positive prototypes
        pos_proto_id = self.g2cluster[dataset_idx]
        pos_prototypes = self.centroids[pos_proto_id]    
        
        # sample negative prototypes
        all_proto_id = list(range(self.num_clusters))       
        neg_proto_id = list(set(all_proto_id)-set(pos_proto_id.tolist()))
        if self.num_neg_protos < len(neg_proto_id):
            neg_proto_id = random.sample(neg_proto_id, self.num_neg_protos)
        
        neg_proto_id = torch.LongTensor(neg_proto_id).to(device)
        
        neg_prototypes = self.centroids[neg_proto_id]    

        proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
        
        # compute prototypical logits
        logits_proto = torch.mm(features, proto_selected.T)
        
        # targets for prototype assignment
        labels_proto = torch.arange(features.size(0), dtype=torch.long).to(device)
        
        # scaling temperatures for the selected prototypes
        temp_proto = self.density[
            torch.cat([pos_proto_id, neg_proto_id.to(device)], dim=0)
            ]
        logits_proto /= temp_proto
        
        return logits_proto, labels_proto