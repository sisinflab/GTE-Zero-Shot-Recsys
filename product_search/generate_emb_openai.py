import argparse
import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import tiktoken
from openai import AzureOpenAI
import numpy as np

ENDPOINT = ""
DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "text-embedding-3-large")
API_VERSION = os.getenv("API_VERSION", "2024-02-01")
API_KEY = ""
EMB_SIZE = 3072

 

def tokenize_truncate_detokenize(sentences, tokenizer, max_length=512):
    """
    Tokenizes a list of sentences with RoBERTa tokenizer, truncates to max_length, and detokenizes back.

    Args:
        sentences (list of str): List of input sentences.
        max_length (int): Maximum number of tokens per sentence (default 512).

    Returns:
        list of str: Detokenized (decoded) strings after truncation.
    """
    results = []
    for sentence in sentences:
        # Tokenize and truncate
        encoded = tokenizer.encode_plus(
            sentence,
            max_length=max_length,
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]

        # Decode back to string
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        results.append(decoded)
    
    return results

def sentence2emb(args, order_texts, feat_name, client, tiktoken_encoder, tokenizer):
    dest_file = os.path.join(args.cache_path, args.dataset_name,
                        args.dataset_name + f'.{feat_name}')

    if os.path.exists(dest_file):
        existing_embeddings = np.fromfile(dest_file, dtype=np.float32).reshape(-1, EMB_SIZE)
        errors = np.sum(np.all(existing_embeddings == 0, axis=1))
    else:
        existing_embeddings = np.array([])
        errors = 0

    if len(existing_embeddings) == len(order_texts):
        print("Full embeddings found")
        return

    embeddings = []
    start, batch_size = 0, 100 

    tot_tokens = 0

    print(f'{feat_name}: ', len(order_texts))
    with tqdm(enumerate(range(len(existing_embeddings), len(order_texts), batch_size)), 
              total=(len(order_texts) + batch_size - 1) // batch_size, initial=len(existing_embeddings) // batch_size) as pbar:
        for bidx, start in pbar:
            sentences = order_texts[start: start + batch_size]
            sentences = tokenize_truncate_detokenize(sentences, tokenizer)  # truncate to 512 to match blair sentences
            tok_length = sum([len(tiktoken_encoder.encode(s)) for s in sentences])

            tot_tokens += tok_length
            try:
                outputs = client.embeddings.create(
                    input=sentences,
                    model=DEPLOYMENT
                )
                outputs = [data.embedding for data in outputs.data]
            except KeyboardInterrupt:
                raise  # Let KeyboardInterrupt propagate
            except:
                outputs = [[0]*EMB_SIZE] * len(sentences)
                print(f"Error in batch id: {bidx}")
                errors += 1

            embeddings.extend(torch.tensor(outputs))

            pbar.set_postfix({"tokens": tot_tokens, "batch_id": bidx, "est_price": tot_tokens / 1000000 * 0.13, "errors": errors})
            if (bidx*batch_size) % 100 == 0: # save every 100 batches
                if len(existing_embeddings) > 0:
                    np.concatenate([existing_embeddings, torch.stack(embeddings, dim=0).numpy()], axis=0).tofile(dest_file)
                else:
                    torch.stack(embeddings, dim=0).numpy().tofile(dest_file)

    if len(existing_embeddings) > 0:
        np.concatenate([existing_embeddings, torch.stack(embeddings, dim=0).numpy()], axis=0).tofile(dest_file)
    else:
        torch.stack(embeddings, dim=0).numpy().tofile(dest_file)

#    print('Embeddings shape: ', embeddings.shape)


def generate_item_emb(args, client, tiktoken_encoder, tokenizer):
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
    sentence2emb(args, item_pool, args.feat_name, client, tiktoken_encoder, tokenizer)


def generate_query_emb(args, client, tiktoken_encoder, tokenizer):
    if args.dataset == 'McAuley-Lab/Amazon-C4':
        dataset = load_dataset(args.dataset)['test']
    elif args.dataset == 'esci':
        dataset = load_dataset('csv', data_files=os.path.join(args.cache_path, 'esci/test.csv'))['train']
    else:
        raise NotImplementedError('Dataset not supported')
    sentence2emb(args, dataset['query'], f'q_{args.feat_name}', client, tiktoken_encoder, tokenizer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='esci', choices=['McAuley-Lab/Amazon-C4', 'esci'])
    parser.add_argument('--cache_path', type=str, default='./cache/')
    parser.add_argument('--plm_name', type=str, default=DEPLOYMENT)
    parser.add_argument('--feat_name', type=str, default=DEPLOYMENT, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.dataset_name = args.dataset.split('/')[-1]
   
    tiktoken_encoder = tiktoken.encoding_for_model(DEPLOYMENT)

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    # create output dir
    os.makedirs(
        os.path.join(args.cache_path, args.dataset_name),
        exist_ok=True
    )
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    generate_item_emb(args, client, tiktoken_encoder, tokenizer)
    generate_query_emb(args, client, tiktoken_encoder, tokenizer)
