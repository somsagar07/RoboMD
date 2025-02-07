import torch
import torch.nn.functional as F

def contrastive_loss(embeddings, labels, margin=1.0):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    same_label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

    positive_loss = same_label_mask * pairwise_dist
    negative_loss = (1 - same_label_mask) * F.relu(margin - pairwise_dist)

    loss = positive_loss.sum() + negative_loss.sum()
    return loss / same_label_mask.numel()

def cosine_similarity_manual(vec_a, vec_b):
    import numpy as np
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    sim = dot_product / (norm_a * norm_b + 1e-8)
    return np.clip(sim, -1.0, 1.0)
    
def find_closest_value(new_embedding, embeddings_array, values_array):
    """
    If you want a standalone function for finding the closest embedding.
    This is often used if you do it outside the environment class.
    """
    distances = cdist([new_embedding], embeddings_array, metric="euclidean")[0]
    idx = np.argmin(distances)
    return values_array[idx], distances[idx]
