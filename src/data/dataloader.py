import random
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

from data.util import graph_data_obj_to_mol_simple


class PairBatch(Data):
    def __init__(self, batch=None, **kwargs):
        super(PairBatch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        data_list = [data for data in data_list if data is not None]
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        keys = [key for key in keys if key not in ["smiles0", "smiles1"]]
        assert "batch" not in keys

        batch = PairBatch()

        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_masked = []
        batch.batch_num_nodes = []
        batch.batch_num_nodes_masked = []

        cumsum_node = 0
        cumsum_node_masked = 0
        batch_size = 0
        for i, data in enumerate(data_list):
            if data is None:
                continue
            
            batch_size += 1
            
            num_nodes = data.x.size(0)
            num_nodes_masked = data.x_masked.size(0)

            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.batch_masked.append(torch.full((num_nodes_masked,), i, dtype=torch.long))

            batch.batch_num_nodes.append(num_nodes)
            batch.batch_num_nodes_masked.append(num_nodes_masked)

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node
                elif key in ["edge_index_masked"]:
                    item = item + cumsum_node_masked

                batch[key].append(item)


            cumsum_node += num_nodes
            cumsum_node_masked += num_nodes_masked

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch_size = batch_size
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_masked = torch.cat(batch.batch_masked, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        batch.batch_num_nodes_masked = torch.LongTensor(batch.batch_num_nodes_masked)
        
        return batch.contiguous()

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1

class PairDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, **kwargs):
        super(PairDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: PairBatch.from_data_list(data_list),
            **kwargs
        )