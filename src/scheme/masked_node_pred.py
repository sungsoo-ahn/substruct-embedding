import random
import numpy as np
import torch
import torch.nn as nn

from model import NodeEncoder
from scheme.util import compute_accuracy

NUM_ATOM_TYPES = 120

class MaskedNodePredModel(nn.Module):
    def __init__(self):
        super(MaskedNodePredModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_lstm_layers = 1
        self.lstm_drop_rate = 0.0

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.classifier = nn.Linear(self.emb_dim, NUM_ATOM_TYPES-1)
    
    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        out = out[batch.node_mask]
        logits = self.classifier(out)
        labels = (batch.y[batch.node_mask] - 1)
        
        return logits, labels
    
class MaskedNodePredScheme():
    def __init__(self):
        pass
     
    def train_step(self, batch, model, optim):
        model.train()
        batch = batch.to(0)
        
        logits, labels = model.compute_logits_and_labels(batch)
        loss = model.criterion(logits, labels)
        acc = compute_accuracy(logits, labels)                
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        statistics = {"loss": loss, "acc": acc}
        
        return statistics
            