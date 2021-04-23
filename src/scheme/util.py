import torch
import faiss
import numpy as np

def compute_accuracy(pred, target):
    acc = float(torch.sum(torch.max(pred, dim=1)[1] == target)) / pred.size(0)
    return acc


def get_contrastive_logits_and_labels(features):
    similarity_matrix = torch.matmul(features, features.T)

    batch_size = similarity_matrix.size(0) // 2

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    return logits, labels

def mask_nodes(x, edge_index, edge_attr, mask_rate):
    num_nodes = x.size(0)
    num_mask_nodes = min(int(mask_rate * num_nodes), 1)
    mask_nodes = list(sorted(random.sample(range(num_nodes), num_mask_nodes)))

    x = x.clone()
    x[mask_nodes] = 0

    return x, edge_index, edge_attr


def run_clustering(
    features,
    num_clusters,
    verbose,
    niter,
    nredo,
    seed,
    max_points_per_centroid,
    min_points_per_centroid,
    use_euclidean_clustering,
    device,
):
    d = features.shape[1]
    clus = faiss.Clustering(d, num_clusters)
    clus.verbose = verbose
    clus.niter = niter
    clus.nredo = nredo
    clus.seed = seed
    clus.max_points_per_centroid = max_points_per_centroid
    clus.min_points_per_centroid = min_points_per_centroid
    clus.spherical = (not use_euclidean_clustering)

    res = faiss.StandardGpuResources()
    res.noTempMemory()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    cfg.device = device
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(features, index)

    # for each sample, find cluster distance and assignments
    D, I = index.search(features, 1)
    item2cluster = [int(n[0]) for n in I]

    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(num_clusters, d)

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids)
    item2cluster = torch.LongTensor(item2cluster)

    result = {"centroids": centroids, "item2cluster": item2cluster}

    obj = clus.iteration_stats.at(clus.iteration_stats.size() - 1).obj
    statistics = {"obj": obj}

    index.reset()

    return result, statistics

def sinkhon(out):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    Q /= torch.sum(Q)

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B

    return Q.t()

