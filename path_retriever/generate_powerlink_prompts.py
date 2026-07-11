"""Regenerate the "related paths" segment of every prompt in a raft_data
split using Power-Link (path_method='power') on the synthetic graph, for a
given dataset. Unlike phase1_quality_compare.py (which sampled 500 test
pairs for a controlled A/B), this processes an arbitrary input file in full
-- used to build the "Power-Link + GraphLoRA" combined treatment's training
data (not just its eval set), so the model actually trains on the new
retrieval's output distribution.

Reuses phase1_quality_compare.py's serialize_paths/splice_prompt (exact
code/translation.py template, so format matches what any G-Refer checkpoint
expects).

Usage (single file):
    python generate_powerlink_prompts.py --dataset yelp \
        --input_jsonl ../raft_data/yelp_10k/train.json \
        --output_jsonl ../raft_data/yelp_10k_powerlink/train.json

Usage (whole split dir, train+eval+test in one process so the model/graph
are only loaded once):
    python generate_powerlink_prompts.py --dataset yelp \
        --split_dir ../raft_data/yelp_10k \
        --output_dir ../raft_data/yelp_10k_powerlink
"""
import argparse
import json
import os
import sys
import time
import torch

# Without this, each process spins up ~nproc threads for its torch/BLAS
# thread pool (observed 172 threads on an 80-core host). Running 3 datasets
# concurrently then oversubscribes the host ~6x and throughput collapses
# (observed ~17s/row instead of the expected ~0.7s/row). Capping keeps 3
# concurrent processes safely within the core count.
torch.set_num_threads(20)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'code')))
from dataset import DataMapper  # noqa: E402

from data_processing import load_dataset  # noqa: E402
from model import HeteroRGCN, HeteroLinkPredictionModel  # noqa: E402
from explainer import PaGELink  # noqa: E402
from phase1_quality_compare import serialize_paths, splice_prompt  # noqa: E402


def process_file(input_jsonl, output_jsonl, mapper, model, mp_g, args):
    num_users = mp_g.num_nodes('user')
    num_items = mp_g.num_nodes('item')

    rows = []
    with open(input_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    print(f'{len(rows)} rows to process from {input_jsonl}')

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    n_written, n_empty, n_oob = 0, 0, 0
    t_start = time.perf_counter()
    with open(output_jsonl, 'w') as out_f:
        for i, row in enumerate(rows):
            uid, iid = row['uid'], row['iid']
            if uid >= num_users or iid >= num_items:
                n_oob += 1
                out_f.write(json.dumps(row) + '\n')  # keep original prompt, can't retrieve
                continue

            pagelink = PaGELink(model, lr=0.01, alpha=1.0, beta=1.0,
                                num_epochs=10, log=False, path_method='power')
            src_nid = torch.tensor([uid])
            tgt_nid = torch.tensor([iid])
            try:
                paths = pagelink.explain(src_nid, tgt_nid, mp_g,
                                         num_hops=args.num_hops,
                                         prune_max_degree=args.prune_max_degree,
                                         k_core=args.k_core,
                                         num_paths=args.num_paths,
                                         max_path_length=args.max_path_length)
            except Exception:
                paths = []
            if not paths or not paths[0]:
                n_empty += 1

            path_text = serialize_paths(paths, mapper, k=args.num_paths)
            new_prompt = splice_prompt(row['prompt'], path_text)
            out_row = {**row, 'prompt': new_prompt}
            out_f.write(json.dumps(out_row) + '\n')
            n_written += 1

            if (i + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - t_start
                print(f'{i + 1}/{len(rows)}  elapsed={elapsed:.1f}s  '
                     f'mean={elapsed / (i + 1):.3f}s/row  empty={n_empty}  oob={n_oob}')

    total = time.perf_counter() - t_start
    print(f'DONE: wrote {n_written + n_oob} rows to {output_jsonl} in {total:.1f}s '
         f'(mean {total / len(rows):.3f}s/row); empty_paths={n_empty}; out_of_bounds={n_oob}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['yelp', 'amazon', 'google'])
    p.add_argument('--input_jsonl', default=None)
    p.add_argument('--output_jsonl', default=None)
    p.add_argument('--split_dir', default=None, help='process train.json/eval.json/test.json in this dir')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--num_paths', type=int, default=2)
    p.add_argument('--max_path_length', type=int, default=5)
    p.add_argument('--num_hops', type=int, default=2)
    p.add_argument('--k_core', type=int, default=2)
    p.add_argument('--prune_max_degree', type=int, default=200)
    p.add_argument('--log_every', type=int, default=200)
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

    if args.split_dir:
        for split in ('train', 'eval', 'test'):
            in_path = os.path.join(args.split_dir, f'{split}.json')
            if not os.path.exists(in_path):
                continue
            out_path = os.path.join(args.output_dir, f'{split}.json')
            process_file(in_path, out_path, mapper, model, mp_g, args)
    else:
        process_file(args.input_jsonl, args.output_jsonl, mapper, model, mp_g, args)


if __name__ == '__main__':
    main()
