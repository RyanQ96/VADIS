import torch


def contrastive_loss(relevance_scores):
    """
    Computes the contrastive loss for a batch of relevance scores.
    Args:
        relevance_scores: Tensor of shape (batch_size, batch_size)
    Returns:
        loss: $\mathcal{L} = - \log \left( \frac{e^{r(p, d^+ )}}{e^{r(p, d^+ )} + \sum_{i \neq p} e^{r(p, d^-_i)}} \right)$
    """
    batch_size = relevance_scores.size(0)
    # Get the positive document scores (diagonal elements)
    positive_scores = torch.diag(relevance_scores)  # Shape: (batch_size,)
    # Compute the denominator: sum over all scores for each query
    all_scores = torch.logsumexp(relevance_scores, dim=1)  # Shape: (batch_size,)
    # Compute the loss
    loss = -positive_scores + all_scores
    loss = loss.mean()
    return loss