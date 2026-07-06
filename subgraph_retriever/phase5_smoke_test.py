"""Phase 5 smoke test: DFTopK differentiability + RCS metric, on real R-GCN /
LightGCN checkpoints from Phase 0-2 (same synthetic yelp_trn graph).

Not a training script -- just validates the two new standalone components
(path_retriever/dftopk.py, evaluation/rcs.py) actually work on real model
outputs before writing them up in experiments/phase5/result.md.
"""
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, 'path_retriever')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, 'evaluation')))
from data_processing import load_dataset  # noqa: E402
from model import HeteroRGCN, LightGCN, HeteroLinkPredictionModel  # noqa: E402
from dftopk import dftopk, dftopk_indices  # noqa: E402
from rcs import ranking_consistency_score  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PR = os.path.join(ROOT, os.pardir, 'path_retriever')


def load_model(encoder_cls, ckpt_name, mp_g, **encoder_kwargs):
    encoder = encoder_cls(mp_g, **encoder_kwargs)
    model = HeteroLinkPredictionModel(encoder, 'user', 'item', 'dot')
    state = torch.load(os.path.join(PR, 'saved_models', ckpt_name), map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    processed_g = load_dataset(os.path.join(PR, 'datasets'), 'yelp', 'trn', 0.1, 0.2, 'step1')[1]
    mp_g = processed_g[0]

    rgcn_model = load_model(HeteroRGCN, 'yelp_model_trn.pth', mp_g,
                            emb_dim=128, hidden_size=128, out_size=128)
    lightgcn_model = load_model(LightGCN, 'yelp_model_trn_lightgcn.pth', mp_g,
                                emb_dim=128, num_layers=2)

    with torch.no_grad():
        rgcn_h = rgcn_model.encode(mp_g)
        lightgcn_h = lightgcn_model.encode(mp_g)

    num_items = mp_g.num_nodes('item')
    test_user = 0
    rgcn_scores = rgcn_h['user'][test_user] @ rgcn_h['item'].T
    lightgcn_scores = lightgcn_h['user'][test_user] @ lightgcn_h['item'].T

    # --- 1. DFTopK sanity: does the soft indicator's implied selection match
    # hard topk() on the SAME score vector? (it should, closely) ---
    k = 20
    hard_topk = set(torch.topk(rgcn_scores, k).indices.tolist())
    soft_topk = set(dftopk_indices(rgcn_scores, k, temperature=0.01).tolist())
    overlap = len(hard_topk & soft_topk) / k
    print(f'[DFTopK sanity] hard vs soft top-{k} overlap (same scores, low temp): {overlap:.3f}')

    # --- 2. DFTopK gradient flow: does a loss on the soft indicator actually
    # backprop into the R-GCN's embedding parameters? ---
    rgcn_model.zero_grad()
    h = rgcn_model.encode(mp_g)
    scores = h['user'][test_user] @ h['item'].T
    soft = dftopk(scores, k, temperature=1.0)
    loss = soft.sum()
    loss.backward()
    emb_grad = rgcn_model.encoder.emb.embeds['item'].weight.grad
    has_grad = emb_grad is not None and emb_grad.abs().sum().item() > 0
    print(f'[DFTopK gradient] item embedding received nonzero gradient: {has_grad} '
         f'(grad abs-sum={emb_grad.abs().sum().item() if emb_grad is not None else None})')

    # --- 3. RCS: how consistent are two independently-trained retrievers'
    # rankings of the same candidates for the same user? ---
    rcs = ranking_consistency_score(lightgcn_scores, rgcn_scores, k=20)
    print(f'[RCS] LightGCN vs R-GCN ranking consistency for user {test_user}: {rcs}')

    # average RCS over a small sample of users for a less noisy number
    n_sample = 50
    overlaps, taus = [], []
    for u in range(n_sample):
        rs = rgcn_h['user'][u] @ rgcn_h['item'].T
        ls = lightgcn_h['user'][u] @ lightgcn_h['item'].T
        r = ranking_consistency_score(ls, rs, k=20)
        overlaps.append(r['topk_overlap'])
        taus.append(r['kendall_tau'])
    print(f'[RCS] avg over {n_sample} users: '
         f'topk_overlap={sum(overlaps)/n_sample:.3f}, kendall_tau={sum(taus)/n_sample:.3f}')


if __name__ == '__main__':
    main()
