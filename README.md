# Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search

This repository contains the code to reproduce the experiments from the paper **"Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search"**, accepted at ACM RecSys ‘25.  

All experiments were conducted on a machine equipped with an **AMD EPYC 7452** processor and an **NVIDIA H100 NVL** GPU running **Ubuntu 22.04 LTS**.  
The code should be reproducible on other operating systems with minimal adjustments. Note that the codebase is based on https://github.com/hyp1231/AmazonReviews2023.


## Setting Up the Virtual Environment
After cloning the repository, it is recommended to create a virtual environment for installing the required dependencies. The codebase was developed using **Python 3.12** with **CUDA 12.1**.  

To set up the virtual environment using `venv`, follow these steps:  

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


# Sequential Recommendation
<!-- Here there are some details to reproduce the results on Sequential Recommendation. In the `sequential_recommendation` folder, there are all the scripts needed to process the dataset and run the experiments. -->

The `sequential_recommendation` directory contains scripts for processing datasets and running experiments for sequential recommendation tasks.

## Dataset Processing
To process the dataset, please navigate to the `sequential_recommendation/dataset` folder. Four scripts are provided to download and preprocess datasets using different PLMs.

> [!NOTE]  
> For the `text-embedding-3-large` please supply your API endpoint and API key

<!-- For instance, to obtain the results with `blair` please run the following -->
To process datasets with, for example, the `blair` model, run
```bash
cd sequential_recommendation/dataset/
python process_amazon_2023.py \
    --domain <domain_name> \
    --device <device_type> \
    --plm <plm_name>
```
**Arguments**:
- `--domain`: The domain of the Amazon Reviews 2023 dataset you are considering. Select one of `All_Beauty`, `Video_Games`, `Baby_Products`.
- `--device`: Specify `cuda:0` or `cpu`.
- `--plm`: The version of `blair` you are considering. Choose one of `hyp1231/blair-roberta-base`, `hyp1231/blair-roberta-large`.

Similarly, datasets can be processed for `text-embedding-3-large`, `KALM`, and `NVEmbed-v2`.
<!-- you can download and process the dataset with the other PLMs, namely `text-embedding-3-large`, `KALM`, and `NVEmbed-v2` -->


## Train and evaluate the models
To train the models it is necessary to generate the configuration files required by RecBole. You may do this by running the following commands:
```bash
cd sequential_recommendation/
python create_config.py \
    -m <model_name> \
    --plm <plm_type> \
```
**Arguments**
- `-m`: The sequential recommendation model you want to test. Please, specify one of `UniSRec`, `SASRecText`, `GRU4RecText`.
- `--plm`: The PLM type. Plese, select one of `blair-base`, `blair-large`, `nvembedv2`, `kalm`, `openai`.

Once you created the necessary config files, you can run the experiments via the following command:

```bash
python run.py \
    -m <recommendation_model_name> \
    -d <domain_name> \
    --gpu_id=<gpu_id>
```
**Arguments**
- `-m`: The sequential recommendation model you want to test. Please, specify one of the following: `UniSRec`, `SASRecText`, `GRU4RecText` for the text-based baselines.
- `-d`: The domain of the Amazon Reviews 2023 dataset you are considering. Provide one of `All_Beauty`, `Video_Games`, `Baby_Products`.
- `-gpu_id`: The id of the available GPU. If it is only one, please select 0.

> [!NOTE]  
> If you encounter issues related to data types, please add the following line to the beginning of the `recbole/evaluator/metrics.py`: `np.float = float`.


# Product Search

## Download data
For the ESCI dataset, please download the processed data from [Google Drive](https://drive.google.com/file/d/1p_x0ec1PgRxLzpcj7dAcasDU-4P8CeN6/view?usp=sharing). After downloading, unzip the archive and and put `sampled_item_metadata_esci.jsonl` and `test.csv` under `product_search/cache/esci/`. 

For Amazon-C4 no operations are needed, it will be automatically downloaded.


## Generate Query/Item Representations

To generate query/item representations, navigate to the folder `product_search/`. Here there are several scripts to generate the query/item representations and cache them with all the models considered in the paper. 

> [!NOTE]  
> For `text-embedding-3-large` please ensure that your OpenAI API endpoint and API key are correctly inserted into the corresponding script.

To generate `blair` embeddings (for `base` and `large` variants) you may run the following commands:
```bash
python generate_emb_blair.py --dataset <dataset_name> --plm_name <plm_name> --feat_name blair-base
```
**Arguments**
- `--dataset`: The dataset you want to consider. It must be either `McAuley-Lab/Amazon-C4` or `esci`.
- `--plm_name`: The version of the blair model you want to consider. It may be `hyp1231/blair-roberta-base` or `hyp1231/blair-roberta-large`.
- `--feat_name`: The name of the serialized features (e.g., `blair-base`)

For the other GTEs you do not have to specify `--plm_name`. For instance, for `NVEmbedv2` you may run the following:
```bash
python generate_emb_nvembedv2.py --dataset <dataset_name> --feat_name <feat_name>
```

Upon completion, the script saves the query and item embeddings into two separate files within the `cache` directory, organized by `dataset`: <dataset_name>.q_<feat_name> for query embeddings ("q_" stands for query), and <dataset_name>.<feat_name> for item embeddings.

## Evaluate Product Search Performance

```bash
python eval_search.py --dataset <dataset_name> --suffix <embedding_suffix> --plm_size <embedding_dimension> --domain
```

**Arguments**
- `--dataset`: The dataset you want to consider. It must be either `McAuley-Lab/Amazon-C4` or `esci`.
- `--suffix`: The suffix of the embeddings you extracted. It corresponds to `feat_name`.
- `--domain`: Whether to extract the results for each domain of the dataset.
- `--plm_size`: Specifies the dimensionality of the embeddings being evaluated. For the exact dimensions of each embedding, please refer to Table 2 in our paper. 


For `BM25`, please run the following command:

```bash
python bm25.py --dataset <dataset_name>
```

**Arguments**
- `--dataset`: The dataset you want to consider. It must be either `McAuley-Lab/Amazon-C4` or `esci`.

## Analysis of the Effective Dimensionality of Embeddings

To compute the effective dimension of the embeddings (as in Section 4.3 of our paper) and to save the corresponding reduced versions, please run the following:

```bash
python apply_pca.py --dataset <dataset_name> --suffix <embedding_suffix> --plm_size <embedding_dimension>  
```
**Arguments**
- `--dataset`: The dataset you want to consider. It must be either `McAuley-Lab/Amazon-C4` or `esci`.
- `--suffix`: The suffix of the embeddings you extracted. It corresponds to `feat_name`.
- `--plm_size`: Specifies the dimensionality of the embeddings being evaluated. For the exact dimensions of each embedding, please refer to Table 2 in our paper. 

The script saves the reduced embeddings by appending the suffix `_PCA80` for the version retaining 80% of the variance, and `_PCA95` for the version retaining 95% of the variance. Additionally, it displays the resulting effective dimensionality for each threshold.

You can evaluate the performance of these embeddings by following the procedure outlined in the previous section.

## The team
Currently, this repository is maintained by:
- Matteo Attimonelli (matteo.attimonelli@poliba.it)
- Alessandro De Bellis (alessandro.debellis@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)

The scientific supervision is driven by:
- Dietmar Jannach (dietmar.jannach@aau.at)
- Eugenio Di Sciascio (eugenio.disciascio@poliba.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)
## Acknowledgement
The codebase is built upon [this repository](https://github.com/hyp1231/AmazonReviews2023).