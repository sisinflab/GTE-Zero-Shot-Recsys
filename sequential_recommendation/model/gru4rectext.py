import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
import copy
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.sequential_recommender.gru4rec import GRU4Rec


class AdaptorLayer(nn.Module):
    def __init__(self, layers, dropout=0.0):
        super(AdaptorLayer, self).__init__()

        self.layers = layers
        self.dropout = dropout

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])):
            if idx != 0:
                mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
        self.mlp_layers = nn.Sequential(*mlp_modules)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_tensor):
        return self.mlp_layers(input_tensor)
    

class GRU4RecText(GRU4Rec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.item_embedding = None
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        self.index_assignment_flag = False

        self.adaptor = AdaptorLayer(
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.adaptor(self.plm_embedding(item_seq))
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.adaptor(self.plm_embedding(pos_items))
            neg_items_emb = self.adaptor(self.plm_embedding(neg_items))
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.adaptor(self.plm_embedding.weight)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.adaptor(self.plm_embedding(test_item))
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.adaptor(self.plm_embedding.weight)
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores



# class GRU4RecText(GRU4Rec):
#     def __init__(self, config, dataset):
#         super().__init__(config, dataset)

#         self.item_embedding = None
#         self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
#         self.index_assignment_flag = False

#         self.adaptor = AdaptorLayer(
#             config['adaptor_layers'],
#             config['adaptor_dropout_prob']
#         )

#     def _init_weights(self, module):
#         if isinstance(module, nn.Embedding):
#             xavier_normal_(module.weight)
#         elif isinstance(module, nn.GRU):
#             xavier_uniform_(module.weight_hh_l0)
#             xavier_uniform_(module.weight_ih_l0)

#     def forward(self, item_seq, item_seq_len):
#         item_seq_emb = self.plm_embedding(item_seq)
#         item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
#         gru_output, _ = self.gru_layers(item_seq_emb_dropout)
#         gru_output = self.dense(gru_output)
#         # the embedding of the predicted item, shape of (batch_size, embedding_size)
#         seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
#         return seq_output

#     def calculate_loss(self, interaction):
#         item_seq = interaction[self.ITEM_SEQ]
#         item_seq_len = interaction[self.ITEM_SEQ_LEN]
#         seq_output = self.forward(item_seq, item_seq_len)
#         pos_items = interaction[self.POS_ITEM_ID]
#         if self.loss_type == "BPR":
#             neg_items = interaction[self.NEG_ITEM_ID]
#             pos_items_emb = self.plm_embedding(pos_items)
#             neg_items_emb = self.plm_embedding(neg_items)
#             pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
#             neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
#             loss = self.loss_fct(pos_score, neg_score)
#             return loss
#         else:  # self.loss_type = 'CE'
#             test_item_emb = self.plm_embedding.weight
#             logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#             loss = self.loss_fct(logits, pos_items)
#             return loss

#     def predict(self, interaction):
#         item_seq = interaction[self.ITEM_SEQ]
#         item_seq_len = interaction[self.ITEM_SEQ_LEN]
#         test_item = interaction[self.ITEM_ID]
#         seq_output = self.forward(item_seq, item_seq_len)
#         test_item_emb = self.plm_embedding(test_item)
#         scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
#         return scores

#     def full_sort_predict(self, interaction):
#         item_seq = interaction[self.ITEM_SEQ]
#         item_seq_len = interaction[self.ITEM_SEQ_LEN]
#         seq_output = self.forward(item_seq, item_seq_len)
#         test_items_emb = self.plm_embedding.weight
#         scores = torch.matmul(
#             seq_output, test_items_emb.transpose(0, 1)
#         )  # [B, n_items]
#         return scores