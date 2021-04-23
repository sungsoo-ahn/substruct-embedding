import random
import numpy as np
import torch
import torch.nn as nn

from model import NodeEncoder
from scheme.util import compute_accuracy

NUM_ATOM_TYPES = 120

class MaskedRWPredClassifier(nn.Module):
    def __init__(self, num_lstm_layers, emb_dim, lstm_drop_rate):
        super(MaskedRWPredClassifier, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.embedding = nn.Embedding(NUM_ATOM_TYPES, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, emb_dim, batch_first=True, num_layers=num_lstm_layers, dropout=lstm_drop_rate
        )
        self.linear = nn.Linear(emb_dim, NUM_ATOM_TYPES-1)
        
    def forward(self, x, hidden):
        hidden = hidden.unsqueeze(0)
        if self.num_lstm_layers == 2:
            hidden = torch.cat([hidden, hidden], dim=0)
        
        out = self.embedding(x)
        out, _ = self.lstm(out, [hidden, hidden])
        out = self.linear(out)
        return out

class MaskedRWPredModel(nn.Module):
    def __init__(self):
        super(MaskedRWPredModel, self).__init__()
        self.num_layers = 5
        self.emb_dim = 300
        self.drop_rate = 0.0
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_lstm_layers = 1
        self.lstm_drop_rate = 0.0

        self.encoder = NodeEncoder(self.num_layers, self.emb_dim, self.drop_rate)
        self.classifier = MaskedRWPredClassifier(
            self.num_lstm_layers, self.emb_dim, self.lstm_drop_rate
            )
    
    def compute_logits_and_labels(self, batch):
        out = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        
        num_nodes = batch.x.size(0)
        lstm_x = batch.y.clone()
        lstm_x[batch.y_mask] = 0
        pad = torch.zeros(num_nodes, 1, dtype=torch.long).cuda()
        lstm_x = torch.cat([pad, lstm_x[:, :-1]], dim=1)
        logits = self.classifier(lstm_x, out)
        labels = batch.y - 1
        
        logits = logits[batch.y_mask]
        labels = labels[batch.y_mask]
        
        return logits, labels
    
class MaskedRWPredScheme():
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
            