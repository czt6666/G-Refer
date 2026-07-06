"""Phase 1 real generation-quality comparison: Dijkstra vs Power-Link.

Takes a real sample of (uid, iid) test pairs from raft_data/yelp/test.json
(which already have a real prompt + ground-truth chosen explanation from the
original G-Refer pipeline), re-retrieves the "related paths" segment using
BOTH path_method=dijkstra and path_method=power on the same synthetic graph
(see experiments/phase0,phase1/result.md for why it's synthetic -- the real
buys/likes-distinguishing raw data was never part of the public release),
serializes with the exact template code/translation.py uses (so format
matches what the model was trained on), and splices the new path text into
the ORIGINAL prompt in place of the original path segment -- keeping the
business/user profile and node-retrieval segments identical, so the retrieval
method is the only variable that differs between the two output files.

Output: two new test.json-format files (one per method), ready for
ds_inference/infer.py, so both can be run through the SAME already-trained
Llama-3-8B checkpoint and compared with evaluation/eval_lite.py.
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
from model import HeteroRGCN, HeteroLinkPredictionModel  # noqa: E402
from explainer import PaGELink  # noqa: E402

PATH_MARKER = ("\n### For the given user-item pair, here are several related "
              "paths connecting users and items through their interactions:")
PATH_TEMPLATE = ("For the given user-item pair, here are several related paths "
                 "connecting users and items through their interactions:")


def serialize_paths(paths, mapper, k=2):
    """Matches code/translation.py's flatten_pagelink_retrieval_results exactly."""
    top_paths = paths[:k]
    path_prompts = []
    for idx, path in enumerate(top_paths, 1):
        path_prompt = []
        for i, edge in enumerate(path):
            if edge[0][0] == 'user':
                path_prompt.append(f"User (Profile: {mapper.get_user_raw_text(edge[1])})")
            elif edge[0][0] == 'item':
                path_prompt.append(f"Item (Profile: {mapper.get_item_raw_text(edge[1])})")
            if i < len(path) - 1:
                next_edge = path[i + 1]
                if edge[0][0] == 'user' and next_edge[0][0] == 'item':
                    path_prompt.append("buys")
                elif edge[0][0] == 'item' and next_edge[0][0] == 'user':
                    path_prompt.append("bought by")
        if path and path[-1][0][-1] == 'item':
            last_item_text = mapper.get_item_raw_text(path[-1][2])
            path_prompt.append("buys")
            path_prompt.append(f"Item (Profile: {last_item_text})")
            path_text = " -> ".join(path_prompt)
            path_prompts.append(f"{idx}. {path_text}")
    return PATH_TEMPLATE + " " + " ".join(path_prompts)


def splice_prompt(original_prompt, new_path_text):
    # new_path_text already starts with PATH_TEMPLATE (from serialize_paths),
    # so the prefix must stop BEFORE the marker, not include it, or the
    # template text would appear twice.
    prefix = original_prompt.split(PATH_MARKER)[0]
    return prefix + "\n### " + new_path_text + "\n### Explanation:"


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
    p.add_argument('--output_dir', default='convert_files/phase1_quality')
    args = p.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    mapper = DataMapper(os.path.join(root, 'data', args.dataset, 'data_trn.pt'))

    processed_g = load_dataset(os.path.join(os.path.dirname(__file__), 'datasets'),
                               args.dataset, 'trn', 0.1, 0.2, 'step1')[1]
    mp_g = processed_g[0]

    encoder = HeteroRGCN(mp_g, 128, 128, 128)
    model = HeteroLinkPredictionModel(encoder, 'user', 'item', 'dot')
    state = torch.load(os.path.join(os.path.dirname(__file__), 'saved_models',
                                    f'{args.dataset}_model_trn.pth'), map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    num_users = mp_g.num_nodes('user')
    num_items = mp_g.num_nodes('item')

    raw_pairs = []
    with open(os.path.join(root, 'raft_data', args.dataset, 'test.json')) as f:
        for line in f:
            raw_pairs.append(json.loads(line))

    import random
    random.seed(args.seed)
    sample = random.sample(raw_pairs, min(args.n_samples, len(raw_pairs)))

    os.makedirs(args.output_dir, exist_ok=True)
    out_files = {m: open(os.path.join(args.output_dir, f'test_{m}.json'), 'w')
                for m in ('dijkstra', 'power')}
    timings = {m: [] for m in ('dijkstra', 'power')}
    n_empty = {m: 0 for m in ('dijkstra', 'power')}
    n_written = 0

    for row in sample:
        uid, iid = row['uid'], row['iid']
        if uid >= num_users or iid >= num_items:
            continue

        row_out = {}
        for method in ('dijkstra', 'power'):
            pagelink = PaGELink(model, lr=0.01, alpha=1.0, beta=1.0,
                                num_epochs=10, log=False, path_method=method)
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
            except Exception as e:
                paths = []
            timings[method].append(time.perf_counter() - t0)
            if not paths or not paths[0]:
                n_empty[method] += 1
            path_text = serialize_paths(paths, mapper, k=args.num_paths)
            new_prompt = splice_prompt(row['prompt'], path_text)
            row_out[method] = {**row, 'prompt': new_prompt}

        for method in ('dijkstra', 'power'):
            out_files[method].write(json.dumps(row_out[method]) + '\n')
        n_written += 1
        if n_written % 50 == 0:
            print(f'{n_written}/{len(sample)}')

    for f in out_files.values():
        f.close()

    print(f'wrote {n_written} pairs per method to {args.output_dir}/test_{{dijkstra,power}}.json')
    for method in ('dijkstra', 'power'):
        t = timings[method]
        print(f'[{method}] mean={sum(t)/len(t):.4f}s total={sum(t):.2f}s '
             f'empty_paths={n_empty[method]}/{n_written}')


if __name__ == '__main__':
    main()
