
import argparse
import os
import random
import time
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from data_set import DataSet
from model import MBGCN

from trainer import Trainer 


seed = 2026
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # True can improve train speed
    torch.backends.cudnn.deterministic = True  # Guarantee that the convolution algorithm returned each time will be deterministic
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=64, help='')
    parser.add_argument('--layers', type=int, default=3)


    parser.add_argument('--data_name', type=str, default='taobao', help='')
    parser.add_argument('--behaviors', help='', action='append')
    parser.add_argument('--loss_type', type=str, default='bpr', help='')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--topk', type=list, default=[10, 20], help='')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='')

    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0.001, help='')

    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=200, help='')
    parser.add_argument('--model_path', type=str, default='./check_point', help='')
    parser.add_argument('--check_point', type=str, default='a_jdata_base.pth', help='')
    parser.add_argument('--model_name', type=str, default='', help='')
    parser.add_argument('--device', type=str, default='cuda:6', help='')
    
    parser.add_argument('--irm_mode', type=str, default='rex',help='rex  v1  v2') 
    parser.add_argument('--penalty_irm_coeff', type=float, default=1) 
    
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    
    parser.add_argument('--lambda_cl', type=float, default=1, help='Weight for the cl loss.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Weight for the cl temperature.')



    args = parser.parse_args()
    if args.data_name == 'tmall':
        args.data_path = './data/Tmall'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'jdata':
        args.data_path = './data/jdata'
        args.behaviors = ['view', 'collect', 'cart', 'buy']
    elif args.data_name == 'taobao':
        args.data_path = './data/taobao'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_delete10':
        args.data_path = './data/taobao_delete10'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_delete30':
        args.data_path = './data/taobao_delete30'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_delete50':
        args.data_path = './data/taobao_delete50'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add10':
        args.data_path = './data/taobao_add10'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add30':
        args.data_path = './data/taobao_add30'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_add50':
        args.data_path = './data/taobao_add50'
        args.behaviors = ['view','cart', 'buy']
    elif args.data_name == 'taobao_disview':
        args.data_path = './data/taobao_disview'
        args.behaviors = ['cart', 'buy']
    elif args.data_name == 'taobao_discart':
        args.data_path = './data/taobao_discart'
        args.behaviors = ['view', 'buy']
    elif args.data_name == 'beibei':
        args.data_path = './data/beibei'
        args.behaviors = ['ipv','cart', 'buy']
    elif args.data_name == 'OODtmall0021':
        args.data_path = './data/OODTmall0021'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    elif args.data_name == 'Tmall_cart_perturbed':
        args.data_path = './data/Tmall_cart_perturbed'
        args.behaviors = ['click', 'collect', 'cart', 'buy']
    else:
        raise Exception('data_name cannot be None')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
    args.TIME = TIME

    logfile = '{}_enb_{}_{}'.format(args.data_name, args.embedding_size, TIME)
    args.train_writer = SummaryWriter('./log/train/' + logfile)
    args.test_writer = SummaryWriter('./log/test/' + logfile)
    logger.add('./log/{}/{}.log'.format(args.model_name, logfile), encoding='utf-8')

    start = time.time()
    dataset = DataSet(args)
    model = MBGCN(args, dataset)

    logger.info(args.__str__())
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    # trainer.evaluate(0, 1, dataset.test_dataset(), dataset.test_interacts, dataset.test_gt_length, args.test_writer)
    logger.info('train end total cost time: {}'.format(time.time() - start))



