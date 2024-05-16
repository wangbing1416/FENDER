import os
import torch
import tqdm
import datetime
import copy
# from .layers import *
from sklearn.metrics import *
from transformers import BertModel
# from opendelta import PrefixModel, LoraModel, SoftPromptModel
from utils.utils import data2gpu, Averager, metrics, Recorder
# from utils.dataloader import get_dataloader
from utils.dataloader import DataProcess
from models.layers import *
import torchdiffeq as ode
# from .huggingface.modeling_bert import BertModel


class PrefixModel(torch.nn.Module):
    def __init__(self, para_len=6, emb_dim=768, value=None):
        super(PrefixModel, self).__init__()
        if value is None:
            self.prefix = torch.nn.Parameter(torch.randn(para_len, emb_dim))
        else:
            self.prefix = torch.nn.Parameter(value)

    def forward(self):
        return self.prefix


class BERTFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, prefix_mlp_dim, dropout, para_len):
        super(BERTFENDModel, self).__init__()
        # self.para_len = para_len
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"): \
                    # or name.startswith('encoder.layer.10') \
                # or name.startswith('encoder.layer.9') \
                # or name.startswith('encoder.layer.8') \
                # or name.startswith('encoder.layer.7') \
                # or name.startswith('encoder.layer.6') \
                # or name.startswith('encoder.layer.5') \
                # or name.startswith('encoder.layer.4') \
                # or name.startswith('encoder.layer.3'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.bertembedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder

        # self.prefix_mlp = nn.Sequential(
        #     nn.Linear(emb_dim, prefix_mlp_dim),
        #     nn.Linear(prefix_mlp_dim, emb_dim),
        #     nn.ReLU()
        # )
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.deltamodel = PrefixModel(self.bert, prefix_token_num=6)

        self.mlp = MLP(emb_dim, mlp_dims, dropout)
        self.attention = MaskAttention(emb_dim)

    def forward(self, prefixmodel, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']

        # bert_feature = self.bert(inputs, attention_mask = masks)[0]  # opendelta is a in-place tool, so using forward function of self.bert
        token_type_ids = torch.zeros(inputs.size(), dtype=torch.long, device=inputs.device)

        bert_embedding = self.bertembedding(input_ids=inputs, token_type_ids=token_type_ids, past_key_values_length=0)
        bert_prefix = torch.cat([prefixmodel().repeat(inputs.size()[0], 1, 1), bert_embedding], dim=1)
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(masks, bert_prefix.size())
        bert_feature = self.bertencoder(bert_prefix, attention_mask=extended_attention_mask)[0]

        bert_feature, _ = self.attention(bert_feature, masks)
        output = self.mlp(bert_feature)
        return torch.sigmoid(output.squeeze(1))


class BERT_ENDEFModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, prefix_mlp_dim, dropout, para_len):
        super(BERT_ENDEFModel, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.bertembedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder

        self.mlp = MLP(emb_dim, mlp_dims, dropout)
        self.attention = MaskAttention(emb_dim)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp)

    def forward(self, prefixmodel, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']

        # bert_feature = self.bert(inputs, attention_mask=masks)[0]
        token_type_ids = torch.zeros(inputs.size(), dtype=torch.long, device=inputs.device)
        bert_embedding = self.bertembedding(input_ids=inputs, token_type_ids=token_type_ids, past_key_values_length=0)
        bert_prefix = torch.cat([prefixmodel().repeat(inputs.size()[0], 1, 1), bert_embedding], dim=1)
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(masks, bert_prefix.size())
        bert_feature = self.bertencoder(bert_prefix, attention_mask=extended_attention_mask)[0]
        feature, _ = self.attention(bert_feature, masks)
        bias_pred = self.mlp(feature).squeeze(1)

        entity = kwargs['entity']
        entity_masks = kwargs['entity_masks']
        # entity_feature = self.bert(entity, attention_mask=masks)[0]
        entity_token_type_ids = torch.zeros(entity.size(), dtype=torch.long, device=entity.device)
        entity_bert_embedding = self.bertembedding(input_ids=entity, token_type_ids=entity_token_type_ids, past_key_values_length=0)
        bert_prefix = torch.cat([prefixmodel().repeat(entity.size()[0], 1, 1), entity_bert_embedding], dim=1)
        entity_extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(entity_masks, bert_prefix.size())
        entity_feature = self.bertencoder(bert_prefix, attention_mask=entity_extended_attention_mask)[0]

        entity_prob = self.entity_net(entity_feature).squeeze(1)
        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), torch.sigmoid(bias_pred)


class BERTEmoModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, prefix_mlp_dim, dropout, para_len):
        super(BERTEmoModel, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.bertembedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder

        self.fea_size = emb_dim
        self.mlp = MLP(emb_dim * 2 + 47, mlp_dims, dropout)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, prefixmodel, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']

        # bert_feature = self.bert(inputs, attention_mask=masks)[0]
        token_type_ids = torch.zeros(inputs.size(), dtype=torch.long, device=inputs.device)
        bert_embedding = self.bertembedding(input_ids=inputs, token_type_ids=token_type_ids, past_key_values_length=0)
        bert_prefix = torch.cat([prefixmodel().repeat(inputs.size()[0], 1, 1), bert_embedding], dim=1)
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(masks, bert_prefix.size())
        bert_feature = self.bertencoder(bert_prefix, attention_mask=extended_attention_mask)[0]

        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, emotion], dim=1))
        return torch.sigmoid(output.squeeze(1))


class BERTEmo_ENDEFModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, prefix_mlp_dim, dropout, para_len):
        super(BERTEmo_ENDEFModel, self).__init__()
        self.bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext').requires_grad_(False)
        self.embedding = self.bert.embeddings

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.bertembedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder

        self.fea_size = emb_dim
        self.mlp = MLP(emb_dim * 2 + 47, mlp_dims, dropout)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=self.fea_size, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.attention = MaskAttention(emb_dim * 2)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.entity_convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_mlp = MLP(mlp_input_shape, mlp_dims, dropout)
        self.entity_net = torch.nn.Sequential(self.entity_convs, self.entity_mlp)

    def forward(self, prefixmodel, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']

        token_type_ids = torch.zeros(inputs.size(), dtype=torch.long, device=inputs.device)
        bert_embedding = self.bertembedding(input_ids=inputs, token_type_ids=token_type_ids, past_key_values_length=0)
        bert_prefix = torch.cat([prefixmodel().repeat(inputs.size()[0], 1, 1), bert_embedding], dim=1)
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(masks, bert_prefix.size())
        bert_feature = self.bertencoder(bert_prefix, attention_mask=extended_attention_mask)[0]

        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        bias_pred = self.mlp(torch.cat([feature, emotion], dim=1)).squeeze(1)

        entity = kwargs['entity']
        entity_masks = kwargs['entity_masks']
        # entity_feature = self.bert(entity, attention_mask=masks)[0]
        entity_token_type_ids = torch.zeros(entity.size(), dtype=torch.long, device=entity.device)
        entity_bert_embedding = self.bertembedding(input_ids=entity, token_type_ids=entity_token_type_ids, past_key_values_length=0)
        bert_prefix = torch.cat([prefixmodel().repeat(entity.size()[0], 1, 1), entity_bert_embedding], dim=1)
        entity_extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(entity_masks, bert_prefix.size())
        entity_feature = self.bertencoder(bert_prefix, attention_mask=entity_extended_attention_mask)[0]
        entity_prob = self.entity_net(entity_feature).squeeze(1)
        return torch.sigmoid(0.9 * bias_pred + 0.1 * entity_prob), torch.sigmoid(entity_prob), torch.sigmoid(bias_pred)


class ODEFunc(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.wt = nn.Linear(hidden_size, hidden_size)

    def forward(self, t, x):
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        x = self.wt(x)
        x = self.dropout_layer(x)
        x = F.relu(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=0.01, atol=0.001, method='dopri5', adjoint=False, terminal=False):
        """
        :param odefunc: X' = f(X, t, G, W)
        :param rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        :param atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        :param method:
            'explicit_adams': AdamsBashforth,
            'fixed_adams': AdamsBashforthMoulton,
            'adams': VariableCoefficientAdamsBashforth,
            'tsit5': Tsit5Solver,
            'dopri5': Dopri5Solver,
            'euler': Euler,
            'midpoint': Midpoint,
            'rk4': RK4,
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint
        self.terminal = terminal

    def forward(self, vt, x):
        integration_time_vector = vt.type_as(x)
        if self.adjoint:
            out = ode.odeint_adjoint(self.odefunc, x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            out = ode.odeint(self.odefunc, x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # return out[-1]
        return out[-1] if self.terminal else out


class NeuralDynamics(nn.Module):
    def __init__(self, input_size, hidden_size, rtol=0.01, atol=0.001, method='dopri5'):
        super(NeuralDynamics, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.Tanh(),
                                         nn.Linear(hidden_size, hidden_size, bias=True))
        self.neural_dynamic_layer = ODEBlock(ODEFunc(hidden_size),
                                             rtol= rtol, atol= atol, method= method)  # t is like  continuous depth
        self.output_layer = nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, vt, x):
        """
        :param t:  end time tick, if t is not used, it is an autonomous system
        :param x:  initial value   N_node * N_dim   400 * hidden_size
        :return:
        """
        x = self.input_layer(x)
        hvx = self.neural_dynamic_layer(vt, x)
        output = self.output_layer(hvx)
        return output


class LSTMDynamics(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMDynamics, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=False)

    def forward(self, x):
        """
        :param x:  initial value   para_len * emb_dim(768)
        :return:
        """
        x = self.LSTM(x)
        return x