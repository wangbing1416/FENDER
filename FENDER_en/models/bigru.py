import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
# from utils.dataloader import get_dataloader
from utils.dataloader import DataProcess


class BiGRUModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, dropout, num_layers):
        super(BiGRUModel, self).__init__()
        self.fea_size = emb_dim
        self.bert = BertModel.from_pretrained('bert-base-uncased').requires_grad_(False)
        self.embedding = self.bert.embeddings

        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)

        input_shape = self.fea_size * 2
        self.attention = MaskAttention(input_shape)
        self.mlp = MLP(input_shape, mlp_dims, dropout)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        feature = self.embedding(inputs)
        feature, _ = self.rnn(feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(feature)
        return torch.sigmoid(output.squeeze(1))


class Trainer():
    def __init__(self,
                 config
                 ):
        self.config = config

        self.save_path = os.path.join(self.config['save_param_dir'], self.config['model_name'])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)

    def train(self, logger=None):
        if (logger):
            logger.info('start training......')
        self.model = BiGRUModel(self.config['emb_dim'], self.config['model']['mlp']['dims'],
                                self.config['model']['mlp']['dropout'], num_layers=1)
        if self.config['use_cuda']:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])
        train_dataprocess = DataProcess(path=self.config['root_path'] + 'train.json', max_len=self.config['max_len'],
                                        aug_prob=self.config['aug_prob'], para_len=self.config['para_len'],
                                        flag='train')
        train_loader = train_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=True,
                                                        segment_type=self.config['segment_type'])

        val_dataprocess = DataProcess(path=self.config['root_path'] + 'val.json', max_len=self.config['max_len'],
                                      aug_prob=self.config['aug_prob'], para_len=self.config['para_len'], flag='val')
        val_loader = val_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=True,
                                                    segment_type=self.config['segment_type'])

        for epoch in range(self.config['epoch']):
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                pred = self.model(**batch_data)
                loss = loss_fn(pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_path, 'parameter_bigru.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bigru.pkl')))

        test_dataprocess = DataProcess(path=self.config['root_path'] + 'test.json', max_len=self.config['max_len'],
                                       aug_prob=self.config['aug_prob'], para_len=self.config['para_len'], flag='val')
        test_future_loader = test_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=False,
                                                             segment_type=self.config['segment_type'])
        future_results = self.test(test_future_loader)
        if (logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info(
                "lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'],
                                                                       future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bigru.pkl')

    def test(self, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']
                batch_pred = self.model(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)