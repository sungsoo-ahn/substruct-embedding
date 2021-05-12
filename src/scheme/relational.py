import torch
import numpy as np
from model import GNN
from torch_geometric.nn import global_mean_pool


class Model(torch.nn.Module):
    def __init__(self, aggr, use_relation):
        super(Model, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.proto_temperature = 0.01
        self.contrastive_temperature = 0.04
        self.criterion = torch.nn.CrossEntropyLoss()
        self.aggr = aggr
        self.use_relation = use_relation

        if self.aggr == "cat":
            self.feat_dim = 2*self.emb_dim
        else:
            self.feat_dim = self.emb_dim
                        
        self.encoder = GNN(self.num_layers, self.emb_dim, drop_ratio=self.drop_rate)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),            
            torch.nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.feat_dim, self.emb_dim),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.ReLU(),            
            torch.nn.Linear(self.emb_dim, (5 if self.use_relation else 2)),
        )
        
        self.pos_labels = torch.full((6, 6), 0, dtype=torch.long)    
        self.neg_labels = torch.full((6, 6), 1, dtype=torch.long)

        self.neg_labels = self.neg_labels.cuda()
        
    def compute_features(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = global_mean_pool(out, batch.batch)
        return out
        
    def compute_logits_and_labels(self, batch):
        batch = batch.to(0)
        features = self.compute_features(batch)        
        batch_size = int(features.size(0) / 6)

        features = features.view(batch_size, 6, self.emb_dim)
        features = torch.transpose(features, 0, 1)
        
        if self.aggr in ["plus", "minus", "max"]:
            features0 = features.unsqueeze(0)
            features1 = features.unsqueeze(1)
            features2 = torch.roll(features1, 1, dims=2)
            if self.aggr == "plus":
                pos_features = features0 + features1
                neg_features = features0 + features2      
            elif self.aggr == "minus":
                pos_features = features0 - features1
                neg_features = features0 - features2      
            if self.aggr == "max":
                pos_features = torch.max(features0, features1)
                neg_features = torch.max(features0, features2)
        
        elif self.aggr == "cat":
            features0 = features.unsqueeze(0).expand(6, 6, batch_size, self.emb_dim)
            features1 = features.unsqueeze(1).expand(6, 6, batch_size, self.emb_dim)
            features2 = torch.roll(features1, 1, dims=2)
            
            pos_features = torch.cat([features0, features1], dim=6)
            neg_features = torch.cat([features0, features2], dim=6)
        
        
        features = torch.cat([pos_features, neg_features], dim=2)
        logits = self.classifier(features.view(-1, self.feat_dim))
        
        pos_labels = self.pos_labels.view(6, 6, 1).expand(6, 6, batch_size)
        neg_labels = self.neg_labels.view(6, 6, 1).expand(6, 6, batch_size)
        labels = torch.cat([pos_labels, neg_labels], dim=2).view(-1)

        if self.use_relation:
            relation_logits = self.relation_classifier(pos_features.view(-1, self.feat_dim))
            relation_logits = relation_logits.view(6, 6, batch_size, 3)
            relation_logits = torch.cat(
                [relation_logits[:3][:, :3], relation_logits[3:][:, 3:]], dim=2
                )
            relation_logits = relation_logits.view(-1, 3)
            relation_labels = (
                self.relation_labels.view(3, 3, 1).expand(3, 3, 2*batch_size).contiguous().view(-1)
            )

            return logits, labels, relation_logits, relation_labels

        else:
            return logits, labels
    
    def compute_pos_label(self, frag_mask):
        # 6 x batch_size x 100
        frag_mask0 = frag_mask.unsqueeze(1)
        frag_mask1 = frag_mask.unsqueeze(0)
        intersect_frag_mask = ((frag_mask0.long() + frag_mask1.long()) == 2)
        
        cond0 = (intersect_frag_mask == frag_mask0).sum(dim=0) == frag_mask0.sum(dim=0)
        cond1 = (intersect_frag_mask == frag_mask1).sum(dim=1) == frag_mask1.sum(dim=1)
        cond2 = (cond0.long() + cond1.long()) == 2
        cond3 = 
        
        label = torch.full((6, 6, frag_mask.size(1), frag_mask.size(2)), 2, dtype=torch.long)
        label[cond0] = 1
        label[cond1] = 1
        label[cond2] = 0
        
        