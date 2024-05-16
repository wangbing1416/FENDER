import logging
import sys
import os
import json
import datetime
import torch
import random
import numpy as np

from models.bert import Trainer as BertTrainer
from models.bertlstm import Trainer as BertLSTMTrainer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))  # logger output as print()

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump
class Run():
    def __init__(self,
                 config
                 ):
        self.config = config
    

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + 'param' + nowtime + '.txt')
        logger.addHandler(logging.FileHandler(param_log_file))
        logger.info('> training arguments:')
        for arg in self.config:
            logger.info('>>> {0}: {1}'.format(arg, self.config[arg]))

        train_param = {
            'seed': [self.config['seed']],
        }
        print(train_param)
        param = train_param
        best_param = []
        json_path = './logs/json/' + self.config['model_name'] + str(self.config['aug_prob']) + '-' + nowtime + '.json'
        json_result = []
        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                torch.cuda.manual_seed(v)
                if 'lstm' in self.config['model_name']:
                    trainer = BertLSTMTrainer(self.config)
                else:
                    # if self.config['model_name'] == 'bert':
                    trainer = BertTrainer(self.config)
                # elif self.config['model_name'] == 'bertemo':
                #     trainer = BertEmoTrainer(self.config)
                # elif self.config['model_name'] == 'bigru':
                #     trainer = BiGRUTrainer(self.config)
                # elif self.config['model_name'] == 'mdfend':
                #     trainer = MDFENDTrainer(self.config)
                # elif self.config['model_name'] == 'eann':
                #     trainer = EANNTrainer(self.config)
                # elif self.config['model_name'] == 'bigru_endef':
                #     trainer = BiGRU_ENDEFTrainer(self.config)
                # elif self.config['model_name'] == 'bert_endef':
                #     trainer = BERT_ENDEFTrainer(self.config)
                # elif self.config['model_name'] == 'bertemo_endef':
                #     trainer = BERTEmo_ENDEFTrainer(self.config)
                # elif self.config['model_name'] == 'eann_endef':
                #     trainer = EANN_ENDEFTrainer(self.config)
                # elif self.config['model_name'] == 'mdfend_endef':
                #     trainer = MDFEND_ENDEFTrainer(self.config)
                metrics, model_path = trainer.train(logger)
                json_result.append(metrics)
                if metrics['metric'] > best_metric['metric']:
                    best_metric['metric'] = metrics['metric']
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('--------------------------------------\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)
