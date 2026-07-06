"""Lite evaluation: BERTScore (P/R/F1), BARTScore, BLEURT, USR.

Replicates the data parsing and metric call conventions of metrics.py
(so numbers align with the paper's Table 1). BLEURT now uses
evaluation/bleurt_scorer.py (real Google bleurt-base-128 checkpoint called
directly via TensorFlow, no external repo code -- see that file's docstring).
GPTScore is still skipped (needs an OpenAI API key).

Run after merging convert_files/yelp_{0,1}/gen_datas.jsonl into
convert_files/<dataset>/gen_datas.jsonl, or pass --gen_path directly.
"""
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="yelp")
parser.add_argument("--gen_path", type=str, default=None,
                    help="override the default convert_files/<dataset>/gen_datas.jsonl path")
parser.add_argument("--skip_bleurt", action="store_true",
                    help="skip BLEURT (slower, loads a second model)")
args = parser.parse_args()


def load(dataset):
    path = args.gen_path or f"convert_files/{dataset}/gen_datas.jsonl"
    data, ref = [], []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            # NOTE: identical to metrics.py — data=ground-truth chosen,
            # ref=model generation. Kept verbatim for paper comparability.
            data.append(s["source_data"]["chosen"].split("### ")[1])
            if "###" in s["output_str"]:
                ref.append(s["output_str"].split("### ")[1])
            else:
                ref.append(s["output_str"])
    return data, ref


def two_seq_same(sa, sb):
    return len(sa) == len(sb) and all(a == b for a, b in zip(sa, sb))


def unique_sentence_percent(seqs):
    uniq = []
    for seq in seqs:
        if not any(two_seq_same(seq, u) for u in uniq):
            uniq.append(seq)
    return len(uniq) / len(seqs)


ROBERTA = "/root/.cache/modelscope/hub/models/AI-ModelScope/roberta-large"
BART = "/root/.cache/modelscope/hub/models/facebook/bart-large-cnn"
BASELINE_TSV = "/opt/miniconda/envs/g-refer/lib/python3.11/site-packages/bert_score/rescale_baseline/en/roberta-large.tsv"


def bert_score_eval(predictions, references):
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(predictions, references, model_type=ROBERTA,
                             num_layers=17, lang="en", rescale_with_baseline=True,
                             baseline_path=BASELINE_TSV, device="cuda:0", verbose=False)
    P, R, F1 = P.numpy(), R.numpy(), F1.numpy()
    return (np.mean(P), np.mean(R), np.mean(F1), np.std(P), np.std(R), np.std(F1))


def bart_score_eval(predictions, references):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from bart_score import BARTScorer
    scorer = BARTScorer(device="cuda:0", checkpoint=BART)
    scores = []
    for i in range(0, len(predictions), 4):
        scores.extend(scorer.score(predictions[i:i+4], references[i:i+4], batch_size=4))
    return np.mean(scores), np.std(scores)


def bleurt_score_eval(predictions, references):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from bleurt_scorer import bleurt_score
    return bleurt_score(predictions, references)


def main():
    data, ref = load(args.dataset)
    print(f"dataset: {args.dataset}  samples: {len(data)}")

    usr = unique_sentence_percent([s.split() for s in data])

    bp, br, bf1, bps, brs, bf1s = bert_score_eval(data, ref)
    bart, bart_std = bart_score_eval(data, ref)

    print("=" * 40)
    print("Explainability metrics (lite):")
    print(f"  BERT-P  : {bp:.4f}  (std {bps:.4f})")
    print(f"  BERT-R  : {br:.4f}  (std {brs:.4f})")
    print(f"  BERT-F1 : {bf1:.4f}  (std {bf1s:.4f})")
    print(f"  BARTscore: {bart:.4f}  (std {bart_std:.4f})")
    if not args.skip_bleurt:
        bleurt, bleurt_std = bleurt_score_eval(data, ref)
        print(f"  BLEURT  : {bleurt:.4f}  (std {bleurt_std:.4f})")
    print(f"  USR     : {usr:.4f}")
    print("  (GPTScore skipped: needs OpenAI API key)")


if __name__ == "__main__":
    main()
