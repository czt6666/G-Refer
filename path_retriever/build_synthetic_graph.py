"""Build the synthetic DGL heterograph used throughout Phase 1-5 for a given
dataset, directly from data/<dataset>/total_trn.csv's real (user, item)
interaction pairs (see experiments/phase0/result.md and memory note
path-retriever-env-and-data-gaps for why this is synthetic: the raw .pkl
needed to distinguish buys/likes/bought_by was never part of the paper's
public data release). All three etypes are filled with the same real edges
-- real node count/scale/topology, degenerate relation semantics.

Usage:
    python build_synthetic_graph.py --dataset amazon
    python build_synthetic_graph.py --dataset google
"""
import argparse
import os
import pandas as pd
import torch
import dgl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['yelp', 'amazon', 'google'])
    p.add_argument('--split', default='trn')
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(root, os.pardir, 'data', args.dataset, f'total_{args.split}.csv')
    df = pd.read_csv(csv_path)
    num_users = int(df.user.max()) + 1
    num_items = int(df.item.max()) + 1
    print(f'{args.dataset}: num_users={num_users} num_items={num_items} edges={len(df)}')

    u = torch.tensor(df.user.values, dtype=torch.long)
    it = torch.tensor(df.item.values, dtype=torch.long)

    g = dgl.heterograph({
        ('user', 'buys', 'item'): (u.tolist(), it.tolist()),
        ('item', 'bought_by', 'user'): (it.tolist(), u.tolist()),
        ('user', 'likes', 'item'): (u.tolist(), it.tolist()),
    }, num_nodes_dict={'user': num_users, 'item': num_items})

    out_dir = os.path.join(root, 'datasets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.dataset}_{args.split}.bin')
    dgl.save_graphs(out_path, [g])
    print(f'saved {g}')
    print(f'-> {out_path}')


if __name__ == '__main__':
    main()
