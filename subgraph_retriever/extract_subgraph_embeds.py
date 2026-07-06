"""Phase 3: precompute l-hop subgraph embeddings for each (uid, iid) pair in
a RAFT split, for soft-prompt injection into the LLM (see subgraph_encoder.py
and ds_training/step1_supervised_finetuning/main.py --subgraph_embed_path).

Popularity-selective strategy (K-RagRec, #9): items with higher in-degree
(more training interactions -> less in need of extra CF signal) get their
subgraph contribution down-weighted; low-degree/cold items get the full
pooled embedding. This is a continuous gate rather than K-RagRec's hard
popularity threshold, to avoid a training-time discontinuity.

Data-availability caveat (see experiments/phase0/result.md, experiments/
phase2/result.md, memory note path-retriever-env-and-data-gaps): the graph
used here is the same Phase 1/2 synthetic one (real 314944 Yelp interaction
pairs from total_trn.csv, but all three edge types are copies of the same
edges, since the raw .pkl needed to distinguish buys/likes/bought_by was
never part of the paper's public data release). Subgraph *topology* quality
is therefore limited by the same degeneracy noted in those reports.

Usage:
    python extract_subgraph_embeds.py --dataset yelp --split trn --num_hops 2
"""
import argparse
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, 'path_retriever')))
from data_processing import load_dataset  # noqa: E402
from model import HeteroRGCN, HeteroLinkPredictionModel  # noqa: E402
from utils import hetero_src_tgt_khop_in_subgraph  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='yelp')
    p.add_argument('--split', type=str, default='trn', choices=['trn', 'tst'])
    p.add_argument('--num_hops', type=int, default=2)
    p.add_argument('--emb_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--out_dim', type=int, default=128)
    p.add_argument('--dataset_dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), os.pardir,
                                       'path_retriever', 'datasets'))
    p.add_argument('--saved_model_dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), os.pardir,
                                       'path_retriever', 'saved_models'))
    p.add_argument('--raft_dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), os.pardir,
                                       'raft_data'))
    p.add_argument('--output_dir', type=str,
                   default=os.path.join(os.path.dirname(__file__), 'embeds'))
    p.add_argument('--popularity_percentile', type=float, default=0.5,
                   help='items with in-degree above this percentile get down-weighted')
    p.add_argument('--popularity_gate_low', type=float, default=1.0)
    p.add_argument('--popularity_gate_high', type=float, default=0.3)
    p.add_argument('--limit', type=int, default=-1, help='only embed the first N pairs (smoke test)')
    return p.parse_args()


def load_pairs(raft_dir, dataset, split_file):
    import json
    path = os.path.join(raft_dir, dataset, split_file)
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pairs.append((d['uid'], d['iid']))
    return pairs


def main():
    args = parse_args()
    device = torch.device('cpu')  # dgl in this env is CPU-only, see memory note

    # Always encode against the 'trn' graph/checkpoint -- only the trn split's
    # synthetic graph.bin exists locally (see experiments/phase0,phase2
    # result.md), and RAFT train/test pairs are largely edges/near-neighbors
    # within that same interaction graph (empirically ~80% of a sample of
    # train.json (uid,iid) pairs are direct edges in it).
    g_split = 'trn'
    processed_g = load_dataset(args.dataset_dir, args.dataset, g_split,
                               0.1, 0.2, 'step1')[1]
    mp_g = processed_g[0].to(device)

    encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
    model = HeteroLinkPredictionModel(encoder, 'user', 'item', 'dot')
    state = torch.load(f'{args.saved_model_dir}/{args.dataset}_model_{g_split}.pth',
                       map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        h_dict = model.encode(mp_g)

    item_in_degree = mp_g.in_degrees(etype=('user', 'likes', 'item'))
    threshold = torch.quantile(item_in_degree.float(), args.popularity_percentile).item()

    num_users = mp_g.num_nodes('user')
    num_items = mp_g.num_nodes('item')

    split_files = {'trn': 'train.json', 'tst': 'test.json'}
    pairs = load_pairs(args.raft_dir, args.dataset, split_files[args.split])
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f'{len(pairs)} (uid, iid) pairs to embed')

    embeds = {}
    skipped = 0
    for i, (uid, iid) in enumerate(pairs):
        if uid >= num_users or iid >= num_items:
            skipped += 1
            continue
        try:
            _, _, sg, feat_nid = hetero_src_tgt_khop_in_subgraph(
                'user', uid, 'item', iid, mp_g, args.num_hops)
        except Exception:
            skipped += 1
            continue

        user_ids = feat_nid['user']
        item_ids = feat_nid['item']
        user_repr = h_dict['user'][user_ids].mean(dim=0) if len(user_ids) > 0 \
            else torch.zeros(args.out_dim)
        item_repr = h_dict['item'][item_ids].mean(dim=0) if len(item_ids) > 0 \
            else torch.zeros(args.out_dim)

        gate = args.popularity_gate_low if item_in_degree[iid].item() <= threshold \
            else args.popularity_gate_high
        vec = torch.cat([user_repr, item_repr]) * gate
        embeds[f'{uid}-{iid}'] = vec

        if (i + 1) % 5000 == 0:
            print(f'  {i + 1}/{len(pairs)}')

    print(f'embedded {len(embeds)} pairs, skipped {skipped}')
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'{args.dataset}_{args.split}_subgraph_embeds.pt')
    torch.save(embeds, out_path)
    print(f'saved to {out_path}')


if __name__ == '__main__':
    main()
