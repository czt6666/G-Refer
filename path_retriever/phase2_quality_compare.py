"""Phase 2 real generation-quality comparison: R-GCN vs LightGCN vs KGAT-lite
backbone, via path retrieval (same splice-into-real-prompt methodology as
path_retriever/phase1_quality_compare.py -- see that file's docstring for the
full rationale). Uses path_method='power' (Phase 1's validated choice) fixed
across all three backbones, so backbone choice is the only variable.
"""
import argparse
import json
import os
import sys
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'code')))
from dataset import DataMapper  # noqa: E402

from data_processing import load_dataset  # noqa: E402
from model import HeteroRGCN, LightGCN, KGAT, HeteroLinkPredictionModel  # noqa: E402
from explainer import PaGELink  # noqa: E402
from phase1_quality_compare import serialize_paths, splice_prompt  # noqa: E402

ENCODERS = {
    'rgcn': (HeteroRGCN, dict(emb_dim=128, hidden_size=128, out_size=128), ''),
    'lightgcn': (LightGCN, dict(emb_dim=128, num_layers=2), '_lightgcn'),
    'kgat': (KGAT, dict(emb_dim=128, hidden_size=128, out_size=128), '_kgat'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='yelp')
    p.add_argument('--n_samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--num_paths', type=int, default=2)
    p.add_argument('--max_path_length', type=int, default=5)
    p.add_argument('--num_hops', type=int, default=2)
    p.add_argument('--k_core', type=int, default=2)
    p.add_argument('--prune_max_degree', type=int, default=200)
    p.add_argument('--output_dir', default='convert_files/phase2_quality')
    args = p.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    mapper = DataMapper(os.path.join(root, 'data', args.dataset, 'data_trn.pt'))

    processed_g = load_dataset(os.path.join(os.path.dirname(__file__), 'datasets'),
                               args.dataset, 'trn', 0.1, 0.2, 'step1')[1]
    mp_g = processed_g[0]
    num_users = mp_g.num_nodes('user')
    num_items = mp_g.num_nodes('item')

    models = {}
    for name, (cls, kwargs, suffix) in ENCODERS.items():
        encoder = cls(mp_g, **kwargs)
        m = HeteroLinkPredictionModel(encoder, 'user', 'item', 'dot')
        state = torch.load(os.path.join(os.path.dirname(__file__), 'saved_models',
                                        f'{args.dataset}_model_trn{suffix}.pth'),
                           map_location='cpu')
        m.load_state_dict(state)
        m.eval()
        models[name] = m

    raw_pairs = []
    with open(os.path.join(root, 'raft_data', args.dataset, 'test.json')) as f:
        for line in f:
            raw_pairs.append(json.loads(line))

    import random
    random.seed(args.seed)
    sample = random.sample(raw_pairs, min(args.n_samples, len(raw_pairs)))

    os.makedirs(args.output_dir, exist_ok=True)
    out_files = {m: open(os.path.join(args.output_dir, f'test_{m}.json'), 'w')
                for m in ENCODERS}
    timings = {m: [] for m in ENCODERS}
    n_empty = {m: 0 for m in ENCODERS}
    n_written = 0

    for row in sample:
        uid, iid = row['uid'], row['iid']
        if uid >= num_users or iid >= num_items:
            continue

        row_out = {}
        for name, model in models.items():
            pagelink = PaGELink(model, lr=0.01, alpha=1.0, beta=1.0,
                                num_epochs=10, log=False, path_method='power')
            src_nid = torch.tensor([uid])
            tgt_nid = torch.tensor([iid])
            t0 = time.perf_counter()
            try:
                paths = pagelink.explain(src_nid, tgt_nid, mp_g,
                                         num_hops=args.num_hops,
                                         prune_max_degree=args.prune_max_degree,
                                         k_core=args.k_core,
                                         num_paths=args.num_paths,
                                         max_path_length=args.max_path_length)
            except Exception:
                paths = []
            timings[name].append(time.perf_counter() - t0)
            if not paths or not paths[0]:
                n_empty[name] += 1
            path_text = serialize_paths(paths, mapper, k=args.num_paths)
            new_prompt = splice_prompt(row['prompt'], path_text)
            row_out[name] = {**row, 'prompt': new_prompt}

        for name in ENCODERS:
            out_files[name].write(json.dumps(row_out[name]) + '\n')
        n_written += 1
        if n_written % 50 == 0:
            print(f'{n_written}/{len(sample)}')

    for f in out_files.values():
        f.close()

    print(f'wrote {n_written} pairs per backbone to {args.output_dir}/test_{{rgcn,lightgcn,kgat}}.json')
    for name in ENCODERS:
        t = timings[name]
        print(f'[{name}] mean={sum(t)/len(t):.4f}s total={sum(t):.2f}s '
             f'empty_paths={n_empty[name]}/{n_written}')


if __name__ == '__main__':
    main()
