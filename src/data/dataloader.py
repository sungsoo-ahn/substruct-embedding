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
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = PairBatch()

        for key in keys:
            batch[key] = []

        batch.batch0 = []
        batch.batch1 = []
        batch.batch_num_nodes0 = []
        batch.batch_num_nodes1 = []

        cumsum_node0 = 0
        cumsum_node1 = 0

        for i, data in enumerate(data_list):
            num_nodes0 = data.x0.size(0)
            num_nodes1 = data.x1.size(0)

            batch.batch0.append(torch.full((num_nodes0,), i, dtype=torch.long))
            batch.batch1.append(torch.full((num_nodes1,), i, dtype=torch.long))

            batch.batch_num_nodes0.append(num_nodes0)
            batch.batch_num_nodes1.append(num_nodes1)

            for key in keys:
                item = data[key]
                if key in ["edge_index0"]:
                    item = item + cumsum_node0
                elif key in ["edge_index1"]:
                    item = item + cumsum_node1

                batch[key].append(item)

            cumsum_node0 += num_nodes0
            cumsum_node1 += num_nodes1

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))

        batch.batch_size = len(data_list)
        batch.batch0 = torch.cat(batch.batch0, dim=-1)
        batch.batch1 = torch.cat(batch.batch1, dim=-1)
        batch.batch_num_nodes0 = torch.LongTensor(batch.batch_num_nodes0)
        batch.batch_num_nodes1 = torch.LongTensor(batch.batch_num_nodes1)

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