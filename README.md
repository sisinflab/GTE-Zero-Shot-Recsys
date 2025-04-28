# Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search

This repository contains the code to reproduce the experiments of the paper entitled "Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search". All the experiments have been conducted on a machine equipped with an AMD EPYC 7452 processor and an Nvidia H100 NVL GPU on Ubuntu 22.04 LTS. The code may be reproduced on other operative system with the arragment needed. 


## Setting Up the Virtual Environment
After cloning the repository, it is recommended to create a virtual environment for installing the required dependencies. The codebase was developed using **Python 3.12** with **CUDA 12.1**.  

To set up the virtual environment using `venv`, follow these steps:  

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


# Sequential Recommendation
Here there are some details to reproduce the results on Sequential Recommendation. In the `sequential_recommendation` folder, there are all the scripts needed to process the dataset and run the experiments.

## Process the dataset
To process the dataset, please navigate to `sequential_recommendation/dataset` folder. Here there are four files that allow you to download and process the datasets with all the PLMs discussed in the paper. 

> [!NOTE]  
> For the `text-embedding-3-large` you are required to put youe ENDPOINT and API-KEY to access to OpenAI models.

For instance, to obtain the results with `blair` please run the following
```bash
cd sequential_recommendation/dataset/
python process_amazon_2023.py \
    --domain <my_domain> \
    --device <my_device> \
    --plm <my_plm>
```
where:
- `--domain`: The domain of the Amazon Reviews 2023 dataset you are considering. Select one of `All_Beauty`, `Video_Games`, `Baby_Products`.
- `--device`: Select `cuda:0` or `cpu`.
- `--plm`: The version of `blair` you are considering. Select one of `hyp1231/blair-roberta-base`, `hyp1231/blair-roberta-large`.

Similarly, you can download and process the dataset with the other PLMs, namely `text-embedding-3-large`, `KALM`, and `NVEmbed-v2`


## Train and evaluate the models
To train the models it is necessary to create config files needed by RecBole. You may do this by running the following commands:
```bash
cd sequential_recommendation/
python create_config.py \
    -m <my_model> \
    --plm <my_plm> \
```
where:
- `-m`: The sequential recommendation model you want to test. Please, select one of the following: `UniSRec`, `SASRecText`, `GRU4RecText`.
- `--plm`: The PLM you want to test. Plese, select one of `blair-base`, `blair-large`, `nvembedv2`, `kalm`, `openai`.

Once you create the necessary config files, you can run the experiments via the following commands:

```bash
python run.py \
    -m <my_recommendation_model> \
    -d <my_dataset> \
    --gpu_id=<my_id>
```
where:
- `-m`: The sequential recommendation model you want to test. Please, select one of the following: `UniSRec`, `SASRecText`, `GRU4RecText` for the text-based baselines.
- `-d`: The domain of the Amazon Reviews 2023 dataset you are considering. Select one of `All_Beauty`, `Video_Games`, `Baby_Products`.
- `-gpu_id`: The id of the available GPU. If it is only one, please select 0.


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

