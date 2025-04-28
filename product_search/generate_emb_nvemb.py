import argparse
import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch.nn.functional as F


def sentence2emb(args, order_texts, feat_name, model, prompt=None, typ=None):
    assert typ is not None
    embeddings = []
    start, batch_size = 0, 8
    max_length = 512
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    if typ == 'query':
        prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    elif typ == 'item':
        prefix=""
    else:
        raise ValueError("typ must be 'query' or 'item'")

    print(f'{feat_name}: ', len(order_texts))
    for start in tqdm(range(0, len(order_texts), batch_size)):
        sentences = order_texts[start: start + batch_size]
        with torch.no_grad(): 
            outputs = model.encode(sentences, instruction=prefix, max_length=max_length)
            outputs = F.normalize(outputs, p=2, dim=1)
        embeddings.extend(outputs.cpu())
        # del outputs
        # torch.cuda.empty_cache()

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.cache_path, args.dataset_name,
                        args.dataset_name + f'.{feat_name}')
    embeddings.tofile(file)


def generate_item_emb(args, model):
    item_pool = []
    if args.dataset == 'McAuley-Lab/Amazon-C4':
        filepath = hf_hub_download(
            repo_id=args.dataset,
            filename='sampled_item_metadata_1M.jsonl',
            repo_type='dataset'
        )
    elif args.dataset == 'esci':
        filepath = os.path.join(args.cache_path, 'esci/sampled_item_metadata_esci.jsonl')
    else:
        raise NotImplementedError('Dataset not supported')
    with open(filepath, 'r') as file:
        for line in file:
            item_pool.append(json.loads(line.strip())['metadata'])
    sentence2emb(args, item_pool, args.feat_name, model, typ='item')


def generate_query_emb(args, model):
    if args.dataset == 'McAuley-Lab/Amazon-C4':
        dataset = load_dataset(args.dataset)['test']
    elif args.dataset == 'esci':
        dataset = load_dataset('csv', data_files=os.path.join(args.cache_path, 'esci/test.csv'))['train']
    else:
        raise NotImplementedError('Dataset not supported')
    sentence2emb(args, dataset['query'], f'q_{args.feat_name}', model, typ='query')


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def load_plm(model_name='bert-base-uncased'):
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
        )
    model.eval()
    model.requires_grad_ = False
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='McAuley-Lab/Amazon-C4', choices=['McAuley-Lab/Amazon-C4', 'esci'])
    parser.add_argument('--cache_path', type=str, default='./cache/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='nvidia/NV-Embed-v2')
    parser.add_argument('--feat_name', type=str, default='nvembv2-base', help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.dataset_name = args.dataset.split('/')[-1]

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    
    plm_model = load_plm(args.plm_name).to(args.device)
    print(plm_model)

    # create output dir
    os.makedirs(
        os.path.join(args.cache_path, args.dataset_name),
        exist_ok=True
    )

    generate_item_emb(args, plm_model)
    generate_query_emb(args, plm_model)
