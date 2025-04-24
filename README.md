# Text-Based Sequential Recommendation

Reproduction of text-based sequential recommendation results. e.g., UniSRec.

> [!NOTE]  
> The original code has been refactored to be more concise and clean. As a result, statistics of the processed dataset, as well as the recommendation model results could be slightly different from the numbers in our paper.

## Quick Start

### Process the dataset

```bash
cd seq_rec_results/dataset/
python process_amazon_2023.py \
    --domain All_Beauty \
    --device cuda:0 \
    --plm hyp1231/blair-roberta-base
```

For all available domains/categories, please refer to [Hugging Face Hub](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/all_categories.txt).

### Train and evaluate the models

```bash
cd seq_rec_results/
python run.py \
    -m UniSRec \
    -d All_Beauty \
    --gpu_id=0
```

# Product Search

## Reproduction - Dense Retrieval Methods

> [!NOTE]  
> The original code has been refactored to be more concise and clean. As a result, the product search results could be slightly different from the numbers in our paper.

(Optional, only if you'd like to reproduce our results on ESCI)
* Download the processed data from [Google Drive](https://drive.google.com/file/d/1p_x0ec1PgRxLzpcj7dAcasDU-4P8CeN6/view?usp=sharing);
* Unzip and put `sampled_item_metadata_esci.jsonl` and `test.csv` under `AmazonReviews2023/product_search_results/cache/esci/`;

First generate dense query/item representations and cache them

```bash
python generate_emb.py --dataset McAuley-Lab/Amazon-C4 --plm_name hyp1231/blair-roberta-base --feat_name blair-base
```

Then evaluate the product search performance

```bash
python eval_search.py --dataset McAuley-Lab/Amazon-C4 --suffix blair-baseCLS --domain
```

**Arguments**

* `--dataset`
    * `McAuley-Lab/Amazon-C4`
    * `esci`

* `--plm_name`
    * `roberta-base`
    * `roberta-large`
    * `princeton-nlp/sup-simcse-roberta-base`
    * `princeton-nlp/sup-simcse-roberta-large`
    * `hyp1231/blair-roberta-base`
    * `hyp1231/blair-roberta-large`

> [!NOTE]  
> Please update `--feat_name` and `--suffix` accordingly.

## Baseline - BM25

(Optional, only if you'd like to reproduce our results on ESCI)
* Download the processed data from [Google Drive](https://drive.google.com/file/d/1p_x0ec1PgRxLzpcj7dAcasDU-4P8CeN6/view?usp=sharing);
* Unzip and put `sampled_item_metadata_esci.jsonl` and `test.csv` under `AmazonReviews2023/product_search_results/cache/esci/`;

```bash
python bm25.py --dataset McAuley-Lab/Amazon-C4
```

**Arguments**

* `--dataset`
    * `McAuley-Lab/Amazon-C4`
    * `esci`

## Data Preprocessing - ESCI

```bash
python dataset/process_esci.py
```

