import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, required=True, help='model name', choices=['UniSRec', 'SASRecText', 'GRU4RecText'])
parser.add_argument('--plm', type=str, required=True, help='plm_name', choices=['blair-base', 'blair-large', 'nvembedv2', 'kalm', 'openai'])
args, unparsed = parser.parse_known_args()

if args.plm == 'blair-base':
    feat_name = 'blair-roberta-base.feature'
    plm_size = 768
elif args.plm == 'blair-large':
    feat_name = 'blair-roberta-large.feature'
    plm_size = 1024
elif args.plm == 'kalm':
    feat_name = 'KaLM-embedding-multilingual-mini-v1.feature'
    plm_size = 896
elif args.plm == 'openai':
    feat_name = 'text-embedding-3-large.feature'
    plm_size = 3072
elif args.plm == 'nvembedv2':
    feat_name = 'NV-Embed-v2.feature'
    plm_size = 4096
else:
    raise ValueError

if args.m == 'UniSRec':
    config = f'''
n_layers: 2
n_heads: 2
hidden_size: 300
inner_size: 256
hidden_dropout_prob: 0.5
attn_dropout_prob: 0.5
hidden_act: 'gelu'
layer_norm_eps: 1e-12
initializer_range: 0.02
loss_type: 'CE'

item_drop_ratio: 0.2
item_drop_coefficient: 0.9
lambda: 1e-3

plm_suffix: {feat_name}
train_stage: inductive_ft  # pretrain / inductive_ft / transductive_ft
plm_size: {plm_size}
adaptor_dropout_prob: 0.2
adaptor_layers: [{plm_size},300]
temperature: 0.07
n_exps: 8

'''
    
elif args.m == "SASRecText":
    config = f'''
n_layers: 2
n_heads: 2
hidden_size: 64
inner_size: 256
hidden_dropout_prob: 0.5
attn_dropout_prob: 0.5
hidden_act: 'gelu'
layer_norm_eps: 1e-12
initializer_range: 0.02
loss_type: 'CE'

plm_suffix: {feat_name}
plm_size: {plm_size}
adaptor_dropout_prob: 0.2
adaptor_layers: [{plm_size},300,64]
'''

elif args.m == 'GRU4RecText':
    config = f'''
num_layers: 2
hidden_size: 64
inner_size: 256
dropout_prob: 0.3
hidden_act: 'gelu'
layer_norm_eps: 1e-12
initializer_range: 0.02
loss_type: 'CE'

plm_suffix: {feat_name}
plm_size: {plm_size}
adaptor_dropout_prob: 0.2
adaptor_layers: [{plm_size},300,64]
embedding_size: 64

'''
else:
    raise ValueError

with open(os.path.join('config', f'{args.m}.yaml'), 'w') as conf_file:
        conf_file.write(config)