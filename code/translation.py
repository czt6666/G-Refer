import argparse
from dataset import DataMapper
import json
import pickle
import os
from utils import load_pickle, load_data
from model import TextModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

# Define mappings
item_map = {"amazon": "Book", "google": "Business", "yelp": "Business"}
split_map = {"trn": "train", "val": "eval", "tst": "test"}

def parse_args():
    parser = argparse.ArgumentParser(description='Flatten retrieval results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., yelp)')
    parser.add_argument('--split', type=str, required=True, help='Data split to use (e.g., trn)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str, choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use (Bert, Roberta, SentenceBert, SimCSE, e5, or t5)')
    parser.add_argument('--k', type=int, default=2, help='Number of paths, users, and items to select for each user-item pair')
    return parser.parse_args()

def flatten_pagelink_retrieval_results(pagelink_retrieval_results, mapper, k):
    flattened_results = {}
    for key, value in pagelink_retrieval_results.items():
        top_paths = value[:k]
        path_prompts = []
        
        prompt_template = "For the given user-item pair, here are several related paths connecting users and items through their interactions:"
        path_prompts = []
        for idx, path in enumerate(top_paths, 1):
            path_prompt = []
            for i, edge in enumerate(path):
                if edge[0][0] == 'user':
                    user_text = mapper.get_user_raw_text(edge[1])
                    path_prompt.append(f"User (Profile: {user_text})")
                elif edge[0][0] == 'item':
                    item_text = mapper.get_item_raw_text(edge[1])
                    path_prompt.append(f"Item (Profile: {item_text})")
                
                if i < len(path) - 1:
                    next_edge = path[i + 1]
                    if edge[0][0] == 'user' and next_edge[0][0] == 'item':
                        path_prompt.append("buys")
                    elif edge[0][0] == 'item' and next_edge[0][0] == 'user':
                        path_prompt.append("bought by")

            if path and path[-1][0][-1] == 'item':
                last_item_id = path[-1][2]
                last_item_text = mapper.get_item_raw_text(last_item_id)
                path_prompt.append("buys")
                path_prompt.append(f"Item (Profile: {last_item_text})")
            
                path_text = " -> ".join(path_prompt)
                path_prompts.append(f"{idx}. {path_text}")
            
        user_id = key[0][1]
        item_id = key[1][1]
        flattened_results[(user_id, item_id)] = prompt_template + " " + " ".join(path_prompts)
    
    return flattened_results

def flatten_dense_retrieval_results(dense_retrieval_results, mapper, k):
    flattened_results = {}
    for key, value in dense_retrieval_results.items():
        user_item_tuple = tuple(map(int, key.split('-')))
        user_item_tuple = (user_item_tuple[0], user_item_tuple[1] - mapper.num_users)
        topk_user_texts = value.get("topk_user_raw_texts", [])[:k]
        topk_item_texts = value.get("topk_item_raw_texts", [])[:k]
        prompt_template = "For the user-item pair, here are some related users and items: Users: {} Items: {}"
        concatenated_texts = prompt_template.format(", ".join(topk_user_texts), ", ".join(topk_item_texts))
        flattened_results[user_item_tuple] = concatenated_texts
    return flattened_results

def merge_flattened_results(flattened_dense_retrieval_results, flattened_pagelink_retrieval_results):
    # merge the two dictionaries, concat the values only if the key is in both dictionaries
    merged_results = {}
    for key, value in flattened_dense_retrieval_results.items():
        if key in flattened_pagelink_retrieval_results:
            merged_results[key] = value + " \n### " + flattened_pagelink_retrieval_results[key]

    # ablation: only has dense retrieval results
    # merged_results = flattened_dense_retrieval_results
    
    # ablation: only has pagelink retrieval results
    # merged_results = flattened_pagelink_retrieval_results
    return merged_results

def write_to_file(file_name, data):
    """
    Write data to a JSON file, creating directories if they don't exist.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str + '\n')

def sample_generation(args, merged_results, dict_data, pyg_data):
    count_no_merge = 0
    all_samples = []
    for i in range(len(dict_data["uid"])):
        # Determine user message based on dataset
        if args.dataset == "amazon":
            user_message = "Given the book title, book profile, and user profile, please explain why the user would buy this book within 50 words."
        elif args.dataset in ["google", "yelp"]:
            user_message = "Given the business title, business profile, and user profile, please explain why the user would enjoy this business within 50 words."
        
        item_type = item_map[args.dataset]
        user_message += f" {item_type} title: {dict_data['title'][i]}. {item_type} profile: {dict_data['item_summary'][i]} User profile: {dict_data['user_summary'][i]}\n### "

        if pyg_data.user_id_to_node.get(dict_data['uid'][i]) and pyg_data.item_id_to_node.get(dict_data['iid'][i]):
            ui_pair = tuple((pyg_data.user_id_to_node.get(dict_data['uid'][i]), pyg_data.item_id_to_node.get(dict_data['iid'][i]) - len(pyg_data.user_id_to_node)))
        else:
            ui_pair = None

        # ablation: no any retrieval results
        if ui_pair in merged_results:
            user_message += f"{merged_results[ui_pair]}"
        else:
            count_no_merge += 1
        user_message += "\n### Explanation:"    

        user_response = f"### {dict_data['explanation'][i]}"
        # Create sample dictionary
        sample = {
            "uid": dict_data['uid'][i],
            "iid": dict_data['iid'][i],
            "prompt": user_message,
            "chosen": user_response,
            "reject": "I DO NOT KNOW"
        }
        all_samples.append(sample)
    print(f"count_no_merge: {count_no_merge}") 
    return all_samples

def rerank_flattened_results(args, all_samples, mapper, write_file):
    if args.split == "tst":
        for sample in all_samples:
            write_to_file(write_file, sample)
        return

    # Initialize the sentence transformer model
    model = TextModel(args.text_encoder)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Process samples in batches
    batch_size = 256
    for i in tqdm(range(0, len(all_samples), batch_size), desc="Reranking samples"):
        batch = all_samples[i:i+batch_size]
        
        uids = [sample['uid'] for sample in batch]
        iids = [sample['iid'] for sample in batch]
        
        user_texts = [mapper.get_user_raw_text(uid) for uid in uids]
        item_texts = [mapper.get_item_raw_text(iid) for iid in iids]
        
        combined_texts = [f"{user_text} {item_text}" for user_text, item_text in zip(user_texts, item_texts)]
        
        ground_truths = [sample['chosen'] for sample in batch]
        
        combined_embeddings = model(combined_texts).cpu().numpy()
        ground_truth_embeddings = model(ground_truths).cpu().numpy()

        similarities = cosine_similarity(combined_embeddings, ground_truth_embeddings)
        
        for j, sample in enumerate(batch):
            sample['similarity_score'] = float(similarities[j][j])

    sorted_samples = sorted(all_samples, key=lambda x: x['similarity_score'])

    for sample in sorted_samples:
        write_to_file(write_file, sample)

    print(f"Reranked and wrote {len(sorted_samples)} samples to {write_file}")


def main():
    args = parse_args()
    
    mapper = DataMapper(f'data/{args.dataset}/data_{args.split}.pt')

    # load dense retrieval results
    with open(f'data/{args.dataset}/dense_retrieval_results_{args.split}.json', 'r') as f:
        dense_retrieval_results = json.load(f)

    # load training/test set
    data = load_pickle(f'data/{args.dataset}/{args.split}.pkl')

    # load pyg data
    pyg_data = load_data(f'data/{args.dataset}/data_{args.split}.pt')

    dict_data = data.to_dict("list")

    # Define write file
    write_file = f"/home/yuhanli/lyh/GRec/convert_files/{args.dataset}/{split_map[args.split]}.json"

    # load pagelink retrieval results
    with open(f'PaGE-Link/saved_explanations/pagelink_{args.dataset}_model_{args.split}_pred_edge_to_paths', 'rb') as f:
        pagelink_retrieval_results = pickle.load(f)

    flattened_dense_retrieval_results = flatten_dense_retrieval_results(dense_retrieval_results, mapper, args.k)
    flattened_pagelink_retrieval_results = flatten_pagelink_retrieval_results(pagelink_retrieval_results, mapper, args.k)

    merged_results = merge_flattened_results(flattened_dense_retrieval_results, flattened_pagelink_retrieval_results)

    # sample several u-i pairs from the merged results
    # sampled_merged_results = {k: v for i, (k, v) in enumerate(merged_results.items()) if i < 5}
    # print(sampled_merged_results)

    all_samples = sample_generation(args, merged_results, dict_data, pyg_data)

    # rerank and store the samples
    rerank_flattened_results(args, all_samples, mapper, write_file)

if __name__ == "__main__":
    main()
