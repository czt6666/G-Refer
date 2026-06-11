import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import extract_k_hop_subgraph, load_data, load_pickle
from tqdm import tqdm
from dataset import DataMapper
import json

class DenseRetriever(nn.Module):
    def __init__(self, pruning_score):
        super(DenseRetriever, self).__init__()
        self.pruning_score = pruning_score

    def forward(self, x):
        return x

    def cosine_similarity(self, x, y):
        return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))

    def retrieve_topk(self, pyg_data, pkl_data, data_mapper, topk):
        self.eval()

        # count the avg users and items retrieved
        avg_users_retrieved = 0
        avg_items_retrieved = 0

        retrieval_results = {}

        progress_bar = tqdm(pkl_data.iterrows(), total=len(pkl_data), desc="Dense Retrieval")

        for _, row in progress_bar:
            user_idx = pyg_data.user_id_to_node.get(row['uid'])
            item_idx = pyg_data.item_id_to_node.get(row['iid'])
            
            if user_idx is None or item_idx is None:
                continue

            _, subgraph_edge_index = extract_k_hop_subgraph(pyg_data, user_idx, item_idx, 2)
            
            # Calculate cosine similarity for users who bought the current item
            user_feat = pyg_data.x[user_idx]
            num_users = len(pyg_data.user_id_to_node)
            subgraph_edge_index = subgraph_edge_index[:, (subgraph_edge_index[0] < num_users) | (subgraph_edge_index[1] >= num_users)]
            users_who_bought_item = set(subgraph_edge_index[0][subgraph_edge_index[1] == item_idx].tolist())
            users_who_bought_item.discard(user_idx)
            user_similarities = [(other_user, self.cosine_similarity(user_feat, pyg_data.x[other_user]).item()) for other_user in users_who_bought_item]
            topk_users = sorted(user_similarities, key=lambda x: x[1], reverse=True)[:topk]

            # Calculate cosine similarity for items bought by the current user
            item_feat = pyg_data.x[item_idx]
            items_bought_by_user = set(subgraph_edge_index[1][subgraph_edge_index[0] == user_idx].tolist())
            items_bought_by_user.discard(item_idx)
            item_similarities = [(other_item, self.cosine_similarity(item_feat, pyg_data.x[other_item]).item()) for other_item in items_bought_by_user]
            topk_items = sorted(item_similarities, key=lambda x: x[1], reverse=True)[:topk]

            # pruning based on the similarity score
            topk_users = [user for user, sim in topk_users if sim >= self.pruning_score]
            topk_items = [item for item, sim in topk_items if sim >= self.pruning_score]
        
            avg_users_retrieved += len(topk_users)
            avg_items_retrieved += len(topk_items)

            # Store retrieval results
            retrieval_results[f"{user_idx}-{item_idx}"] = {
                "topk_user_ids": topk_users,
                "topk_item_ids": topk_items,
                "topk_user_raw_texts": [data_mapper.get_user_raw_text(user) for user in topk_users],
                "topk_item_raw_texts": [data_mapper.get_item_raw_text(item - data_mapper.num_users) for item in topk_items],
                "topk_user_similarities": [sim for sim in topk_users],
                "topk_item_similarities": [sim for sim in topk_items]
            }

        print(f"Average users retrieved: {avg_users_retrieved / len(pkl_data)}")
        print(f"Average items retrieved: {avg_items_retrieved / len(pkl_data)}")    

        return retrieval_results

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve top-k similar users and items')
    parser.add_argument('--dataset', type=str, choices=['yelp', 'google', 'amazon'], required=True,
                        help='Dataset to use (yelp, google, or amazon)')
    parser.add_argument('--split', type=str, choices=['trn', 'val', 'tst'], required=True,
                        help='Data split to use (trn, val, or tst)')
    parser.add_argument('--topk', type=int, default=5, help='Number of top-k similar nodes to retrieve')
    parser.add_argument('--pruning_score', type=float, default=0, help='Pruning score for similarity')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Processing {args.dataset} {args.split} data...")
    
    pyg_data = load_data(f'data/{args.dataset}/data_{args.split}.pt')  # Load PyG data
    pkl_data = load_pickle(f'data/{args.dataset}/{args.split}.pkl')

    data_mapper = DataMapper(f'data/{args.dataset}/data_{args.split}.pt')
    
    retriever = DenseRetriever(args.pruning_score)
    retrieval_results = retriever.retrieve_topk(pyg_data, pkl_data, data_mapper, args.topk)

    # Save retrieval results
    with open(f'data/{args.dataset}/dense_retrieval_results_{args.split}.json', 'w') as f:
        json.dump(retrieval_results, f)

    print(f"Retrieval results saved to data/{args.dataset}/dense_retrieval_results_{args.split}.json")

if __name__ == "__main__":
    main()

# python Retriever/dense_retriever.py --dataset yelp --split trn --topk 5