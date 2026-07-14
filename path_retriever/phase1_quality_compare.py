"""Retrieve paths with one method (dijkstra or power) and write infer prompts.

Example:
  python phase1_quality_compare.py --method power --n_samples 3000 \\
      --output_dir ../convert_files/phase1_test3000 --resume
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

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
    prefix = original_prompt.split(PATH_MARKER)[0]
    return prefix + "\n### " + new_path_text + "\n### Explanation:"


def _count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    return sorted_vals[min(int(len(sorted_vals) * q), len(sorted_vals) - 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['dijkstra', 'power'])
    p.add_argument('--dataset', default='yelp')
    p.add_argument('--n_samples', type=int, default=3000)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--num_paths', type=int, default=2)
    p.add_argument('--max_path_length', type=int, default=5)
    p.add_argument('--num_hops', type=int, default=2)
    p.add_argument('--k_core', type=int, default=2)
    p.add_argument('--prune_max_degree', type=int, default=200)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--output_dir', default='../convert_files/phase1_test3000')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log_every', type=int, default=50)
    args = p.parse_args()

    method = args.method
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), out_dir))
    os.makedirs(out_dir, exist_ok=True)

    started = datetime.now().isoformat(timespec='seconds')
    print(f'[START] method={method} time={started}', flush=True)

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
            if line.strip():
                raw_pairs.append(json.loads(line))

    random.seed(args.seed)
    sample = random.sample(raw_pairs, min(args.n_samples, len(raw_pairs)))
    sample.sort(key=lambda r: (r['uid'], r['iid']))

    out_path = os.path.join(out_dir, f'test_{method}.json')
    timing_path = os.path.join(out_dir, f'timing_{method}.jsonl')
    start_i = _count_lines(out_path) if args.resume else 0
    if start_i:
        print(f'resume from {start_i}', flush=True)

    mode = 'a' if start_i > 0 else 'w'
    out_f = open(out_path, mode)
    timing_f = open(timing_path, mode)
    times = []
    n_empty = 0

    if args.resume and start_i > 0 and os.path.exists(timing_path):
        with open(timing_path) as f:
            for i, line in enumerate(f):
                if i >= start_i:
                    break
                rec = json.loads(line)
                times.append(rec['time_s'])
                if rec.get('empty'):
                    n_empty += 1

    n_written = start_i
    t_wall0 = time.perf_counter()
    try:
        for idx, row in enumerate(sample):
            if idx < start_i:
                continue
            uid, iid = row['uid'], row['iid']
            if uid >= num_users or iid >= num_items:
                continue

            pagelink = PaGELink(model, lr=0.01, alpha=1.0, beta=1.0,
                                num_epochs=args.num_epochs, log=False,
                                path_method=method)
            src_nid = torch.tensor([uid])
            tgt_nid = torch.tensor([iid])
            t0 = time.perf_counter()
            try:
                paths = pagelink.explain(
                    src_nid, tgt_nid, mp_g,
                    num_hops=args.num_hops,
                    prune_max_degree=args.prune_max_degree,
                    k_core=args.k_core,
                    num_paths=args.num_paths,
                    max_path_length=args.max_path_length)
            except Exception:
                paths = []
            dt = time.perf_counter() - t0
            times.append(dt)
            empty = (not paths) or (not paths[0])
            if empty:
                n_empty += 1

            path_text = serialize_paths(paths, mapper, k=args.num_paths)
            new_row = {**row, 'prompt': splice_prompt(row['prompt'], path_text)}
            out_f.write(json.dumps(new_row) + '\n')
            out_f.flush()
            timing_f.write(json.dumps({
                'uid': uid, 'iid': iid, 'time_s': dt, 'empty': empty
            }) + '\n')
            timing_f.flush()
            n_written += 1

            if n_written % args.log_every == 0:
                elapsed = time.perf_counter() - t_wall0
                recent = times[-args.log_every:]
                print(f'{n_written}/{len(sample)} wall={elapsed:.0f}s '
                      f'recent_mean={sum(recent)/len(recent):.3f}s', flush=True)
    finally:
        out_f.close()
        timing_f.close()

    ended = datetime.now().isoformat(timespec='seconds')
    ts = sorted(times)
    summary = {
        'method': method,
        'n': len(times),
        'n_written': n_written,
        'total_s': sum(times) if times else 0.0,
        'mean_s': (sum(times) / len(times)) if times else None,
        'min_s': ts[0] if ts else None,
        'max_s': ts[-1] if ts else None,
        'p50_s': _pct(ts, 0.50),
        'p95_s': _pct(ts, 0.95),
        'empty_paths': n_empty,
        'empty_rate': (n_empty / n_written) if n_written else 0.0,
        'start_time': started,
        'end_time': ended,
        'seed': args.seed,
        'n_requested': args.n_samples,
    }
    summary_path = os.path.join(out_dir, f'timing_{method}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'[END] method={method} time={ended}', flush=True)
    print(f'wrote {n_written} -> {out_path}', flush=True)
    print(f'timing -> {summary_path}', flush=True)
    if times:
        print(f"[{method}] total={summary['total_s']:.1f}s mean={summary['mean_s']:.4f}s "
              f"p50={summary['p50_s']:.4f}s p95={summary['p95_s']:.4f}s "
              f"max={summary['max_s']:.4f}s empty={n_empty}/{n_written}", flush=True)


if __name__ == '__main__':
    main()
