from scheme.base import GraphContrastiveModel, NodeContrastiveModel
from scheme.base import BaseScheme as Scheme
from data.transform import clone_data, sample_fragment

def transform(data, drop_rate):
    frag_data0 = sample_fragment(clone_data(data), drop_rate)
    frag_data1 = sample_fragment(clone_data(data), drop_rate)
    
    return frag_data0, frag_data1