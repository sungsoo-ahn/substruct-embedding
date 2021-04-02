import random
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

from data.util import graph_data_obj_to_mol_simple


class SubBatch(Data):
    def __init__(self, batch=None, **kwargs):
        super(SubBatch, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list, compute_true_target):
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = SubBatch()

        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.sub_batch = []
        batch.batch_num_nodes = []
        batch.sub_batch_num_nodes = []

        cumsum_node = 0
        sub_cumsum_node = 0

        for i, data in enumerate(data_list):
            if data.sub_x.size(0) == 0:
                continue
            
            num_nodes = data.num_nodes
            sub_num_nodes = data.sub_x.size(0)

            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
            batch.sub_batch.append(torch.full((sub_num_nodes,), i, dtype=torch.long))

            batch.batch_num_nodes.append(num_nodes)
            batch.sub_batch_num_nodes.append(sub_num_nodes)

            for key in keys:
                item = data[key]
                if key in ["edge_index"]:
                    item = item + cumsum_node
                elif key in ["sub_edge_index"]:
                    item = item + sub_cumsum_node

                batch[key].append(item)

            cumsum_node += num_nodes
            sub_cumsum_node += sub_num_nodes

        for key in keys:
            if key in ["smiles", "sub_smarts"]:
                continue

            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch_size = len(data_list)
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.sub_batch = torch.cat(batch.sub_batch, dim=-1)
        batch.batch_num_nodes = torch.LongTensor(batch.batch_num_nodes)
        batch.sub_batch_num_nodes = torch.LongTensor(batch.sub_batch_num_nodes)

        return batch.contiguous()

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1


class SubDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, compute_true_target, **kwargs):
        super(SubDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: SubBatch.from_data_list(
                data_list, compute_true_target=compute_true_target
            ),
            **kwargs
        )