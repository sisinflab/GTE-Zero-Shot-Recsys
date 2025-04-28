import torch
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--suffix', type=str, required=True, help='suffix of the embeddings')
parser.add_argument('--dataset', type=str, default='McAuley-Lab/Amazon-C4', choices=['McAuley-Lab/Amazon-C4', 'esci'])
parser.add_argument('--plm_size', type=int, required=True, help='size of the embeddings')
args = parser.parse_args()

if args.dataset == 'McAuley-Lab/Amazon-C4':
    item_path = os.path.join('cache', 'Amazon-C4', f"Amazon-C4.{args.suffix}")
    query_path = os.path.join('cache', 'Amazon-C4', f"Amazon-C4.q_{args.suffix}")
elif args.dataset == 'esci':
    item_path = os.path.join('cache', 'esci', f"esci.{args.suffix}")
    query_path = os.path.join('cache', 'esci', f"esci.q_{args.suffix}")
else:
    raise ValueError('Dataset not supported!')

print('Loading embeddings...')
items = torch.tensor(np.fromfile(item_path, dtype=np.float32).reshape(-1, args.plm_size))
queries = torch.tensor(np.fromfile(query_path, dtype=np.float32).reshape(-1, args.plm_size))
idx = torch.randperm(items.shape[0])
idx_sub = idx[:100000]
sub_items = items[idx_sub]

items_c = items - items.mean(dim=0)
queries_c = queries - queries.mean(dim=0)
sub_items_c = sub_items - sub_items.mean(dim=0)
print('computing SVD...')
U, S, Vh = torch.linalg.svd(sub_items_c, full_matrices=False)
print('computing components...')
explained_variance = (S ** 2) / (sub_items.shape[0] - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()
cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)

num_components_80 = torch.searchsorted(cumulative_variance, 0.8).item() + 1
num_components_95 = torch.searchsorted(cumulative_variance, 0.95).item() + 1

print('computing PCA@80...')
items_c_80 = items_c @ Vh[:num_components_80].T
queries_c_80 = queries_c @ Vh[:num_components_80].T

items_c_80.detach().cpu().numpy().astype(np.float32).tofile(item_path+'_PCA80')
queries_c_80.detach().cpu().numpy().astype(np.float32).tofile(query_path+'_PCA80')

print('computing PCA@95...')
items_c_95 = items_c @ Vh[:num_components_95].T
queries_c_95 = queries_c @ Vh[:num_components_95].T

items_c_95.detach().cpu().numpy().astype(np.float32).tofile(item_path+'_PCA95')
queries_c_95.detach().cpu().numpy().astype(np.float32).tofile(query_path+'_PCA95')

print(num_components_80, num_components_95)

