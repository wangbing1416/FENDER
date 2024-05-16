import os
import torch
import tqdm
import datetime
import copy
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
# from opendelta import PrefixModel, LoraModel, SoftPromptModel
from utils.utils import data2gpu, Averager, metrics, Recorder
# from utils.dataloader import get_dataloader
from utils.dataloader import DataProcess
from model import BERTFENDModel, BERT_ENDEFModel, BERTEmoModel, BERTEmo_ENDEFModel, PrefixModel, ODEBlock, ODEFunc, LSTMDynamics

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
        global lstm_input
        nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
        if(logger):
            logger.info('start training......')
        if self.config['model_name'] == 'bertlstm':
            self.model = BERTFENDModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['prefix_mlp_dim'], self.config['model']['mlp']['dropout'], para_len=self.config['para_len'])
        elif self.config['model_name'] == 'bertemolstm':
            self.model = BERTEmoModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['prefix_mlp_dim'], self.config['model']['mlp']['dropout'], para_len=self.config['para_len'])
        # elif self.config['model_name'] == 'bertemoendef':
        #     self.model = BERTEmo_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['prefix_mlp_dim'], self.config['model']['mlp']['dropout'], para_len=self.config['para_len'])
        # elif self.config['model_name'] == 'bertendef':
        #     self.model = BERT_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['prefix_mlp_dim'], self.config['model']['mlp']['dropout'], para_len=self.config['para_len'])
        else:
            self.model = BERT_ENDEFModel(self.config['emb_dim'], self.config['model']['mlp']['dims'], self.config['prefix_mlp_dim'], self.config['model']['mlp']['dropout'], para_len=self.config['para_len'])

        if self.config['use_cuda']:
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()

        recorder = Recorder(self.config['early_stop'])
        train_dataprocess = DataProcess(path=self.config['root_path'] + 'train.json', max_len=self.config['max_len'],
                                        aug_prob=self.config['aug_prob'], para_len=self.config['para_len'], flag='train')
        train_loader = train_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=True, segment_type=self.config['segment_type'])

        val_dataprocess = DataProcess(path=self.config['root_path'] + 'val.json', max_len=self.config['max_len'],
                                      aug_prob=self.config['aug_prob'], para_len=self.config['para_len'], flag='val')
        val_loader = val_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=True, segment_type=self.config['segment_type'])

        # val_loader = get_dataloader(self.config['root_path'] + 'val.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'], para_len=self.config['para_len'])
        # train_loader = get_dataloader(self.config['root_path'] + 'train.json', self.config['max_len'],
        #                               self.config['batchsize'], shuffle=True, use_endef=False,
        #                               aug_prob=self.config['aug_prob'], para_len=self.config['para_len'])

        # todo: pre-training
        logger.info('Stage 1: overall model pre-training...')
        self.prefixmodel = PrefixModel(para_len=self.config['para_len'], emb_dim=self.config['emb_dim'])
        if self.config['use_cuda']:
            self.prefixmodel = self.prefixmodel.cuda()
        diff_part = ["bertModel.embeddings", "bertModel.encoder"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.config['lr']
            },
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.config['mlp_lr']
            },
            {
                "params": [p for n, p in self.prefixmodel.named_parameters()],
                "weight_decay": 0.0,
                "lr": self.config['mlp_lr']
            }
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=self.config['adam_epsilon'])
        for epoch in range(self.config['epoch']):
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):

                batch_data = data2gpu(batch, self.config['use_cuda'])
                label = batch_data['label']

                if 'endef' in self.config['model_name']:
                    pred, entity_pred, _ = self.model(self.prefixmodel, **batch_data)
                    loss = loss_fn(pred, label.float()) + self.config['loss_cof'] * loss_fn(entity_pred, label.float())
                else:
                    pred = self.model(self.prefixmodel, **batch_data)
                    loss = loss_fn(pred, label.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            logger.info('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(self.prefixmodel, val_loader)
            mark = recorder.add(results)  # early stop with validation metrics
            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_bert' + nowtime + '.pkl'))
            elif mark == 'esc':
                break
            else:
                continue


        # todo: time adaptive fine-tuning
        logger.info('Stage 2: time adaptive fine-tuning...')
        dynamics_label = []
        dynamics_time = []
        if self.config['segment_type'] == 'year':
            future_index = 11
        else:
            future_index = 38
        for period in range(1, future_index):  # 10 periods by year
            logger.info('training the {}-th period'.format(period))
            self.adaptivemodel = copy.deepcopy(self.prefixmodel)
            train_loader_period = train_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=True, segment_type=self.config['segment_type'], period=period)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.adaptivemodel.named_parameters()],
                    "weight_decay": 0.0,
                    "lr": self.config['finetuning_lr']
                }
            ]
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=self.config['adam_epsilon'])
            min_loss = 100
            flag = 0
            for epoch in range(self.config['finetuning_epoch']):
                self.model.train()
                train_data_iter = tqdm.tqdm(train_loader_period)
                avg_loss = Averager()

                for step_n, batch in enumerate(train_data_iter):
                    batch_data = data2gpu(batch, self.config['use_cuda'])
                    label = batch_data['label']

                    if 'endef' in self.config['model_name']:
                        pred, entity_pred, _ = self.model(self.prefixmodel, **batch_data)
                        loss = loss_fn(pred, label.float()) + self.config['loss_cof'] * loss_fn(entity_pred, label.float())
                    else:
                        pred = self.model(self.prefixmodel, **batch_data)
                        loss = loss_fn(pred, label.float())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss.add(loss.item())

                if avg_loss.item() < min_loss:  # early stop with the training loss
                    min_loss = avg_loss.item()
                    flag = 0
                else:
                    flag += 1
                if flag >= self.config['early_stop']:
                    break
                logger.info('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            dynamics_time.append(period)
            dynamics_label.append(self.adaptivemodel.prefix.detach().unsqueeze(dim=0))
        dynamics_time = torch.tensor(dynamics_time)  # 10
        dynamics_label = torch.cat(dynamics_label, dim=0)  # 10 * para_len * emb_dim


        # todo: dynamics learning
        logger.info('Stage 3: dynamics learning')
        loss_fn_dyn = torch.nn.L1Loss()
        x0 = self.prefixmodel.prefix.detach()  # para_len * emb_dim
        self.dynamicsmodel = LSTMDynamics(input_size=self.config['emb_dim'], hidden_size=self.config['emb_dim'])
        if self.config['use_cuda']:
            self.dynamicsmodel = self.dynamicsmodel.cuda()

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.dynamicsmodel.named_parameters()],
                "weight_decay": 0.0,
                "lr": self.config['dynamics_lr']
            }
        ]
        dynamics_optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=self.config['adam_epsilon'])
        min_loss = 100
        flag = 0
        for epoch in range(self.config['dynamics_epoch']):
            self.dynamicsmodel.train()
            avg_loss = Averager()
            lstm_input = torch.cat([x0.unsqueeze(dim=0), dynamics_label], dim=0)
            lstm_pred = self.dynamicsmodel(lstm_input)
            dynamics_loss = loss_fn_dyn(lstm_pred[0][:-1, :, :], dynamics_label)

            dynamics_optimizer.zero_grad()
            dynamics_loss.backward()
            dynamics_optimizer.step()
            avg_loss.add(dynamics_loss.item())
            logger.info('Dynamics learning epoch: {} / {}; Loss: {}'.format(epoch + 1, self.config['dynamics_epoch'], avg_loss.item()))
            if avg_loss.item() < min_loss:  # early stop with the training loss
                min_loss = avg_loss.item()
                flag = 0
            else:
                flag += 1
            if flag >= self.config['dynamics_early_stop']:
                break
        torch.save(self.dynamicsmodel.state_dict(), os.path.join(self.save_path, 'dynamics' + nowtime + '.pkl'))


        # todo: test stage
        logger.info("Stage: testing...")
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bert' + nowtime + '.pkl')))
        test_dataprocess = DataProcess(path=self.config['root_path'] + 'test.json', max_len=self.config['max_len'],
                                       aug_prob=self.config['aug_prob'], para_len=self.config['para_len'], flag='val')
        test_loader =test_dataprocess.get_dataloader(batch_size=self.config['batchsize'], shuffle=True, segment_type=self.config['segment_type'])
        # if self.config['para_len'] == 0:
        #     self.future_prefixmodel = self.prefixmodel
        # else:
        future_prefix = self.dynamicsmodel(lstm_input)[0][-1, :, :].squeeze()
        self.future_prefixmodel = PrefixModel(value=future_prefix)
        # test_future_loader = get_dataloader(self.config['root_path'] + 'test.json', self.config['max_len'], self.config['batchsize'], shuffle=False, use_endef=False, aug_prob=self.config['aug_prob'], para_len=self.config['para_len'])
        future_results = self.test(self.future_prefixmodel, test_loader)

        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.config['lr'], self.config['aug_prob'], future_results['metric']))
        print('test results:', future_results)

        return future_results, os.path.join(self.save_path, 'parameter_bert.pkl')

    def test(self, prefixmodel, dataloader):
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.config['use_cuda'])
                batch_label = batch_data['label']

                if 'endef' in self.config['model_name']:
                    _, _, batch_pred = self.model(prefixmodel, **batch_data)
                else:
                    batch_pred = self.model(prefixmodel, **batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
        
        return metrics(label, pred)