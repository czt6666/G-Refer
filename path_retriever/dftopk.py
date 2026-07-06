"""Phase 5: DFTopK differentiable top-k selection (#12, arXiv:2510.11472).

f_k(x) = sigmoid((x - (x[k] + x[k+1]) / 2) / temperature)

x[k], x[k+1] are the k-th and (k+1)-th largest values of x (1-indexed in the
paper's notation; 0-indexed here). This gives a soft indicator per element:
scores well above the midpoint between the k-th and (k+1)-th largest get an
indicator near 1 (selected), scores well below get near 0, and gradient only
concentrates near the decision boundary -- O(n) (a sort, which autograd
routes gradients through correctly) instead of the combinatorial cost of
differentiating through an actual arg-topk. Intended (per the paper) as a
drop-in for hard top-k in node-retrieval selection, not for graph search
(path selection is combinatorial, not a plain top-k over independent
scores) -- that's why Phase 1 used graph-powering instead of this for paths.
"""
import torch


def dftopk(scores, k, temperature=1.0):
    """
    Parameters
    ----------
    scores : Tensor, shape [..., N]
    k : int, number of items intended to be selected (1 <= k < N)
    temperature : float, smaller -> sharper approximation to hard top-k

    Returns
    -------
    soft_indicator : Tensor, shape [..., N], values in (0, 1)
    """
    n = scores.shape[-1]
    k = max(1, min(k, n - 1))
    sorted_scores, _ = torch.sort(scores, dim=-1, descending=True)
    threshold = (sorted_scores[..., k - 1] + sorted_scores[..., k]) / 2
    return torch.sigmoid((scores - threshold.unsqueeze(-1)) / temperature)


def dftopk_indices(scores, k, temperature=1.0, hard_threshold=0.5):
    """Convenience: the *set* of indices DFTopK's soft indicator would select
    at `hard_threshold` (for inspection/eval only -- for training, use the
    continuous `dftopk()` output directly so gradients flow).
    """
    soft = dftopk(scores, k, temperature)
    return (soft >= hard_threshold).nonzero(as_tuple=True)[-1]
