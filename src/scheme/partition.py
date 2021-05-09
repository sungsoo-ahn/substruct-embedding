from scheme.base import GraphContrastiveModel


class Model(GraphContrastiveModel):
    def __init__(self):
        super(Model, self).__init__()        
        self.transform_mat = torch.nn.Parameter(torch.empty(self.emb_dim, self.emb_dim))
        torch.nn.init.xavier_normal_(self.transform_mat)
        
    def get_logits_and_labels(self, features):
        transform_mat = 0.5 * (self.transform_mat + self.transform_mat.t())
        transformed_features = torch.matmul(features, transform_mat)
        
        similarity_matrix = torch.matmul(transformed_features, features.t())
        batch_size = similarity_matrix.size(0) // 2

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        mask = (torch.eye(labels.shape[0]) > 0.5).to(features.device)
        labels[mask] = 0.0

        positives = similarity_matrix[(labels > 0.5)].view(labels.shape[0], -1)
        negatives = similarity_matrix[(labels < 0.5)].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        return logits, labels