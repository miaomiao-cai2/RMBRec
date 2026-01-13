
import copy
import time
import os

import torch
import numpy as np
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data_set import DataSet
from metrics import metrics_dict
from tool import shuffle

class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.train_writer = args.train_writer
        self.test_writer = args.test_writer
        self.device = args.device
        self.TIME = args.TIME

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer

    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)
        # train_dataset_loader = self.dataset.train_data
        # best_result = np.zeros(len(self.topk) * len(self.metrics))
        best_result = 0
        best_dict = {}
        best_epoch = 0
        best_model = None
        final_test = None
        for epoch in range(self.epochs):

            self.model.train()
            test_metric_dict, validate_metric_dict = self._train_one_epoch(train_dataset_loader, epoch)

            if validate_metric_dict is not None:
                result = validate_metric_dict['hit@10']
                # early stop
                if result - best_result > 0:
                    final_test = test_metric_dict
                    best_result = result
                    best_dict = validate_metric_dict
                    best_model = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                if epoch - best_epoch > 20:
                    break
        # save the best model
        self.save_model(best_model)
        logger.info(f"training end, best iteration %d, results: %s" %
                    (best_epoch + 1, best_dict.__str__()))

        logger.info(f"final test result is:  %s" % final_test.__str__())

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        start_time = time.time()

        # users = torch.Tensor(behavior_dataset_loader[:, 0]).long()
        # posItems = torch.Tensor(behavior_dataset_loader[:, 1]).long()
        # negItems = torch.Tensor(behavior_dataset_loader[:, 2]).long()
        # users = users.to(self.device)
        # posItems = posItems.to(self.device)
        # negItems = negItems.to(self.device)
        # users, posItems, negItems = shuffle(users, posItems, negItems)
        # total_batch = len(users) // self.batch_size + 1
        
        # # 使用 tqdm 构造进度条
        # behavior_dataset_iter = tqdm(
        #     range(total_batch),  # 迭代的总次数是总批次数量
        #     desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"  # 进度条描述
        # )

        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )

        
        total_loss = 0.0
        batch_no = 0
        # for batch_index in behavior_dataset_iter:
        for batch_index, batch_data in behavior_dataset_iter:

            #  # 计算当前批次的起始和结束索引
            # start_idx = batch_index * self.batch_size
            # end_idx = min((batch_index + 1) * self.batch_size, len(users))  # 防止超出范围

            # # 从 users, posItems, negItems 中提取当前批次的数据
            # batch_users = users[start_idx:end_idx].to(self.device)  # 当前批次的用户 ID
            # batch_posItems = posItems[start_idx:end_idx].to(self.device)  # 当前批次的正样本物品 ID
            # batch_negItems = negItems[start_idx:end_idx].to(self.device)  # 当前批次的负样本物品 ID

            start = time.time()
            batch_data = batch_data.to(self.device)
          
            self.optimizer.zero_grad()
            # loss = self.model(batch_users,batch_posItems,batch_negItems)
            # main_loss,total_behavior_loss,cl_loss,penalty_loss,reg_loss = self.model(batch_data,epoch)
            # logger.info(f"Loss: main={main_loss.item():.4f}, behavior={total_behavior_loss.item():.4f}, cl={cl_loss.item():.4f}, penalty={penalty_loss.item():.4f}, reg={reg_loss.item():.4f}")
            # loss = main_loss + total_behavior_loss + cl_loss + penalty_loss + reg_loss
            loss = self.model(batch_data,epoch)
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()
        total_loss = total_loss / batch_no

        self.train_writer.add_scalar('total Train loss', total_loss, epoch + 1)
        epoch_time = time.time() - start_time
        logger.info('epoch %d %.2fs Train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

        # validate
        start_time = time.time()
        validate_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.validate_dataset(),
                                             self.dataset.validation_interacts, self.dataset.validation_gt_length,
                                             self.train_writer)
        epoch_time = time.time() - start_time
        logger.info(
            f"validate %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, validate_metric_dict.__str__()))

        # test
        start_time = time.time()
        test_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                         self.dataset.test_interacts, self.dataset.test_gt_length,
                                         self.test_writer)
        epoch_time = time.time() - start_time
        logger.info(
            f"test %d cost time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))


        return test_metric_dict, validate_metric_dict


    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, writer):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)
        self.model.eval()
        start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_behavior_dict['buy']
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            start = time.time()
            scores = self.model.full_predict(batch_data)

            for index, user in enumerate(batch_data):
                user_score = scores[index]
                items = train_items.get(str(user.item()), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[str(user.item())]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        for key, value in metric_dict.items():
            writer.add_scalar('evaluate ' + key, value, epoch + 1)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save(model, os.path.join(self.model_path, self.model_name + self.TIME + '.pth'))


 