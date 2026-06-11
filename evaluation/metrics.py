import evaluate
import numpy as np
from bart_score import BARTScorer
import argparse
import json
from bleurt import score
import tensorflow as tf
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="amazon", help="amazon, yelp or google")
parser.add_argument("--ratio", type=float, default=0.1, help="ratio of data to use for evaluation")
args = parser.parse_args()

# your api key
api_key = ""

# your api base
api_base = ""

client = OpenAI(base_url=api_base, api_key=api_key)

with open("evaluation/system_prompt.txt", "r") as f:
    system_prompt = f.read()

class MetricScore:
    def __init__(self):
        print(f"evaluating dataset: {args.dataset}")
        self.input_path = f"convert_files/{args.dataset}/gen_datas.jsonl"
        self.data = []
        self.ref_data = []
        with open(self.input_path, 'r') as f:
            for line in f.readlines():
                sample = json.loads(line)
                self.data.append(sample["source_data"]["chosen"].split("### ")[1])
                if "###" in sample["output_str"]:
                    self.ref_data.append(sample["output_str"].split("### ")[1])
                else:
                    self.ref_data.append(sample["output_str"])

    def get_score(self):
        scores = {}
        (
            bert_precison,
            bert_recall,
            bert_f1,
            bert_precison_std,
            bert_recall_std,
            bert_f1_std,
        ) = BERT_score(self.data, self.ref_data)

        gpt_score, gpt_std = get_gpt_score(self.data, self.ref_data)
        bart_score, bart_score_std = BART_score(self.data, self.ref_data)
        bleurt_score, bleurt_score_std = BLEURT_score(self.data, self.ref_data)
        
        tokens_predict = [s.split() for s in self.data]
        usr, _ = unique_sentence_percent(tokens_predict)

        scores["gpt_score"] = gpt_score
        scores["bert_precision"] = bert_precison
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1
        scores["usr"] = usr

        scores["gpt_std"] = gpt_std
        scores["bert_precision_std"] = bert_precison_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std

        scores["bart_score"] = bart_score
        scores["bart_score_std"] = bart_score_std
        scores["bleurt_score"] = bleurt_score
        scores["bleurt_score_std"] = bleurt_score_std
        return scores

    def print_score(self):
        scores = self.get_score()
        print(f"dataset: {args.dataset}")
        print(f"ratio: {args.ratio}")
        print("Explanability Evaluation Metrics:")
        print(f"gpt_score: {scores['gpt_score']:.4f}")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"bart_score: {scores['bart_score']:.4f}")
        print(f"bleurt_score: {scores['bleurt_score']:.4f}")
        print(f"usr: {scores['usr']:.4f}")
        print("-"*30)
        print("Standard Deviation:")
        print(f"gpt_std: {scores['gpt_std']:.4f}")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")
        print(f"bart_score_std: {scores['bart_score_std']:.4f}")
        print(f"bleurt_score_std: {scores['bleurt_score_std']:.4f}")

def get_gpt_response(prompt):
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
    )
    response = completion.choices[0].message.content
    return float(response)


def get_gpt_score(predictions, references):
    prompts = []
    for i in range(len(predictions)):
        prompt = {
            "prediction": predictions[i],
            "reference": references[i],
        }
        prompts.append(json.dumps(prompt))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(get_gpt_response, prompt) for prompt in prompts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Processing GPT responses"):
            results.append(future.result())

    return np.mean(results), np.std(results)

def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True

def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        # seq is a list of words
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)

def BERT_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
    )
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1),
        np.std(precision),
        np.std(recall),
        np.std(f1),
    )

def BART_score(predictions, references):
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    scores = []
    for i in tqdm(range(0, len(predictions), 4), desc="Computing BART scores"):
        batch_pred = predictions[i:i+4]
        batch_ref = references[i:i+4]
        batch_scores = bart_scorer.score(batch_pred, batch_ref, batch_size=4)
        scores.extend(batch_scores)
    return np.mean(scores), np.std(scores)

def BLEURT_score(predictions, references):
    bleurt_ops = score.create_bleurt_ops()
    scores = []
    for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Computing BLEURT scores"):
        ref_tensor = tf.constant([ref])
        pred_tensor = tf.constant([pred])
        bleurt_out = bleurt_ops(references=ref_tensor, candidates=pred_tensor)
        scores.append(bleurt_out["predictions"][0])
    return np.mean(scores), np.std(scores)