import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--segment_type', default='year')
parser.add_argument('--finetuning_epoch', type=int, default=50)
parser.add_argument('--dynamics_epoch', type=int, default=1000)
parser.add_argument('--aug_prob', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--para_len', type=int, default=32)

parser.add_argument('--loss_cof', type=float, default=0.2)

parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--dynamics_early_stop', type=int, default=5)
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--inner_dim', type=int, default=384)
parser.add_argument('--prefix_mlp_dim', type=int, default=384)
parser.add_argument('--dynamics_mlp_dim', type=int, default=768)

parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=7e-5)
parser.add_argument('--mlp_lr', type=float, default=7e-5)
parser.add_argument('--finetuning_lr', type=float, default=1e-5)
parser.add_argument('--dynamics_lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--param_log_dir', default = './logs/param')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from grid_search import Run
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {}'.format(args.lr, args.model_name, args.batchsize, args.epoch, args.gpu))


config = {
        'use_cuda': True,
        'segment_type': args.segment_type,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'dynamics_early_stop': args.dynamics_early_stop,
        'root_path': args.root_path,
        'aug_prob': args.aug_prob,
        'weight_decay': args.weight_decay,
        'loss_cof': args.loss_cof,
        'model':
            {
            'mlp': {'dims': [args.inner_dim], 'dropout': args.dropout}
            },
        'emb_dim': args.emb_dim,
        'prefix_mlp_dim': args.prefix_mlp_dim,
        'dynamics_mlp_dim': args.dynamics_mlp_dim,
        'para_len': args.para_len,
        'lr': args.lr,
        'mlp_lr': args.mlp_lr,
        'finetuning_lr': args.finetuning_lr,
        'dynamics_lr': args.dynamics_lr,
        'adam_epsilon': args.adam_epsilon,
        'epoch': args.epoch,
        'finetuning_epoch': args.finetuning_epoch,
        'dynamics_epoch': args.dynamics_epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_log_dir': args.save_log_dir,
        'save_param_dir': args.save_param_dir,
        'param_log_dir': args.param_log_dir
        }

if __name__ == '__main__':
    Run(config = config).main()
