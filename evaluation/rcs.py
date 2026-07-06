"""Phase 5: Ranking Consistency Score (#17, arXiv:2205.01289, adapted).

The paper's RCS quantifies how well a cheap pre-ranking stage's ordering
agrees with an expensive downstream ranker's ordering, so pre-ranking can be
optimized for *consistency with* the final ranker rather than just its own
in-isolation accuracy. Adapted here for G-Refer's setting: how well does the
(fast, interpretable) retriever's top-k node/path selection agree with the
(slow, more accurate) GNN link-predictor's own ranking of the same
candidates -- i.e., is the retriever picking what the GNN would have picked?

Two complementary views are provided:
- top-k overlap: fraction of the retriever's top-k that the GNN also ranks
  in its own top-k (order-insensitive, easy to interpret).
- rank correlation (Kendall's tau): full-ranking agreement, sensitive to
  ordering, not just set membership.
"""
import torch


def topk_overlap(retriever_scores, reference_scores, k):
    """Fraction of the two top-k sets that overlap, in [0, 1]."""
    n = retriever_scores.shape[-1]
    k = max(1, min(k, n))
    retriever_topk = set(torch.topk(retriever_scores, k).indices.tolist())
    reference_topk = set(torch.topk(reference_scores, k).indices.tolist())
    return len(retriever_topk & reference_topk) / k


def kendall_tau(retriever_scores, reference_scores):
    """Kendall's tau rank correlation between two full score vectors over the
    same N candidates, in [-1, 1] (1 = identical ranking).
    """
    n = retriever_scores.shape[-1]
    if n < 2:
        return 1.0
    r_rank = torch.argsort(torch.argsort(retriever_scores, descending=True))
    g_rank = torch.argsort(torch.argsort(reference_scores, descending=True))
    concordant = 0
    discordant = 0
    idx = torch.combinations(torch.arange(n), r=2)
    r_diff = r_rank[idx[:, 0]] - r_rank[idx[:, 1]]
    g_diff = g_rank[idx[:, 0]] - g_rank[idx[:, 1]]
    sign_prod = torch.sign(r_diff) * torch.sign(g_diff)
    concordant = (sign_prod > 0).sum().item()
    discordant = (sign_prod < 0).sum().item()
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def ranking_consistency_score(retriever_scores, reference_scores, k=10):
    """Combined RCS: (top-k overlap, Kendall's tau) for one (retriever,
    reference) candidate-score pair. Both retriever_scores and
    reference_scores are 1-D tensors over the same N candidates.
    """
    return {
        'topk_overlap': topk_overlap(retriever_scores, reference_scores, k),
        'kendall_tau': kendall_tau(retriever_scores, reference_scores),
    }
