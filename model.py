
import os.path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from lightGCN import LightGCN
from utils import BPRLoss, EmbLoss



class MBGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(MBGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.all_edge_index = dataset.all_edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size

        self.args = args
        
        # IRM相关参数
        self.irm_mode = args.irm_mode
        
        # 为IRMv1创建虚拟分类器参数
        if self.irm_mode == 'v1':
            self.dummy_w = nn.Parameter(torch.ones(1, device=self.device))

        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.1)
        
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.behavior_gcns = nn.ModuleDict({
            behavior: LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, inter, self.embedding_size)
            for behavior, inter in dataset.inter_matrix.items()
        })

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self._load_model()



    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    

    def gcn_propagate(self, user_id_emb,item_id_emb):
        """
        gcn propagate in each behavior
        """
        user_embeddings, item_embeddings = [], []
        for behavior in self.behaviors:
            behavior_embeddings = self.behavior_gcns[behavior](user_id_emb,item_id_emb)
            user_embedding, item_embedding = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1])
            
            user_embeddings.append(user_embedding)
            item_embeddings.append(item_embedding)



        all_user_embeddings = torch.stack(user_embeddings, dim=1)
        all_item_embeddings = torch.stack(item_embeddings, dim=1)


        stacked_item_embs = torch.stack(item_embeddings, dim=0)  # dim=0 表示沿新的第一个维度堆叠
        final_item_embs = stacked_item_embs.mean(dim=0)      # 沿行为维度（dim=0）求平均，形状变为 (num_item, embedding_dim)

        stacked_user_embs = torch.stack(user_embeddings, dim=0)
        final_user_embs = stacked_user_embs.mean(dim=0)


        return all_user_embeddings, all_item_embeddings,final_user_embs,final_item_embs

    
    def get_all_contrastive_loss(self,all_user_embeddings,users):
         # --- START: “一对多”对比学习损失计算模块 ---
    
        total_contrastive_loss = 0.0
        temperat = self.args.temperature # 温度系数超参数

        # 1. 确定锚点视角。
        #    假设您的 self.behaviors 列表是 ['click', 'collect', 'cart', 'buy']
        #    我们选择 索引3 (buy) 作为锚点。
        anchor_view_index = len(self.behaviors) - 1
        # anchor_view_index = 1
        anchor_embs = all_user_embeddings[:, anchor_view_index][users]
       
        anchor_embs = F.normalize(anchor_embs, dim=1)
        
        # 2. 循环遍历所有其他非锚点视角，并计算对比损失。
        for i in range(len(self.behaviors)):
            if i == anchor_view_index:
                continue # 跳过锚点自身

            # 获取当前对比的另一个视角
            other_embs = all_user_embeddings[:, i][users]
            other_embs = F.normalize(other_embs, dim=1)

            # 计算批内相似度矩阵
            sim_matrix_anchor_vs_other = torch.matmul(anchor_embs, other_embs.T) / temperat
            
            # 创建InfoNCE损失的目标标签 (对角线)
            labels = torch.arange(sim_matrix_anchor_vs_other.shape[0]).long().to(self.device)

            # 计算对称的对比损失
            loss_anchor_vs_other = F.cross_entropy(sim_matrix_anchor_vs_other, labels)
            loss_other_vs_anchor = F.cross_entropy(sim_matrix_anchor_vs_other.T, labels)
            
            # 累加当前这一对视角的对比损失
            total_contrastive_loss += (loss_anchor_vs_other + loss_other_vs_anchor) / 2.0
            # total_contrastive_loss += loss_anchor_vs_other

        # --- END: 对比学习损失计算模块 ---
        return total_contrastive_loss
        
    def compute_irm_v1_penalty(self, behavior_losses):
        """
        IRMv1惩罚项计算
        对于每个行为环境，计算在dummy_w=1处的梯度范数
        """
        irm_penalty = 0.0
        
        # 为每个行为环境计算IRMv1惩罚
        for i, behavior_loss in enumerate(behavior_losses):
            # 计算在dummy_w=1处的梯度
            grad_w = torch.autograd.grad(
                behavior_loss, 
                self.dummy_w, 
                create_graph=True,  # 保留计算图以便二阶导数
                retain_graph=True   # 保留计算图供后续使用
            )[0]
            
            # 添加梯度范数的平方作为惩罚
            irm_penalty += torch.norm(grad_w, p=2) ** 2
        
        return irm_penalty

    def compute_irm_v2_penalty(self, behavior_losses):
        """
        IRMv2惩罚项计算
        计算不同行为环境间梯度的方差
        """
        # 收集每个行为环境的梯度
        gradients = []
        
        for i, behavior_loss in enumerate(behavior_losses):
            # 计算梯度
            grad_w = torch.autograd.grad(
                behavior_loss,
                self.dummy_w if self.irm_mode == 'v1' else list(self.parameters())[0],  # 使用第一个参数作为参考
                create_graph=True,
                retain_graph=True
            )[0]
            
            gradients.append(grad_w.flatten())
        
        # 堆叠所有梯度
        grad_matrix = torch.stack(gradients)  # [n_behaviors, grad_dim]
        
        # 计算环境间梯度的方差
        grad_variance = torch.var(grad_matrix, dim=0).mean()
        
        return grad_variance

    def compute_irm_loss(self, behavior_losses):
        """
        根据IRM模式计算不变性惩罚
        """
        if self.irm_mode == 'rex':
            return torch.var(torch.stack(behavior_losses, dim=0))
        elif self.irm_mode == 'v1':
            return self.compute_irm_v1_penalty(behavior_losses)
        elif self.irm_mode == 'v2':
            return self.compute_irm_v2_penalty(behavior_losses)
        else:
            raise ValueError(f"Unknown IRM mode: {self.irm_mode}")
        
    # def forward(self, batch_users,batch_posItems,batch_negItems):
    def forward(self, batch_data, epoch=None):
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        user_id_preference = self.user_embedding.weight
        itwm_id_preference = self.item_embedding.weight
        all_user_embeddings, all_item_embeddings,final_user_embs,final_item_embs = self.gcn_propagate(user_id_preference,itwm_id_preference)


        total_behavior_loss = 0
        main_loss = 0
        grad_list = []
        behavior_reg = []
        for i in range(len(self.behaviors)):
            data = batch_data[:, i]
            users = data[:, 0].long()
            positems = data[:, 1].long()
            negItems = data[:, 2].long()
            
            user_feature = all_user_embeddings[:, i][users]
            positem_feature = all_item_embeddings[:, i][positems]
            negItem_feature = all_item_embeddings[:, i][negItems]
            posscores = torch.sum(user_feature * positem_feature, dim=1)
            negscores = torch.sum(user_feature * negItem_feature, dim=1)

            behavior_i_bpr = self.bpr_loss(posscores, negscores)
            
             # 如果是IRMv1模式，使用dummy_w缩放损失
            if self.irm_mode == 'v1':
                behavior_i_bpr = behavior_i_bpr * self.dummy_w
                
            reg_embedding_loss = self.calculate_reg_loss(user_feature,positem_feature,negItem_feature,len(users))
            grad_list.append(behavior_i_bpr)
            behavior_reg.append(reg_embedding_loss)
            # total_behavior_loss += behavior_i_bpr
            
            
        final_user_feature = final_user_embs[users]
        final_positem_feature = final_item_embs[positems]
        final_negitem_feature = final_item_embs[negItems]

        final_posscores = torch.sum(final_user_feature * final_positem_feature, dim=1)
        final_negscores = torch.sum(final_user_feature * final_negitem_feature, dim=1)
        main_loss += self.bpr_loss(final_posscores, final_negscores)
        
        unique_users = torch.unique(users)
        cl_user_loss = self.get_all_contrastive_loss(all_user_embeddings , unique_users)
        unique_items = torch.unique(positems)
        cl_item_loss = self.get_all_contrastive_loss(all_item_embeddings , unique_items)
        # cl_loss = cl_user_loss * 0.5 + cl_item_loss * 0.01
        cl_loss = cl_user_loss 
        penalty_irm_coeff =  self.args.penalty_irm_coeff


        # penalty_grad_var = torch.var(torch.stack(grad_list, dim=0))
        
        # 计算IRM惩罚
        irm_penalty = self.compute_irm_loss(grad_list)
        penalty_irm_loss = penalty_irm_coeff *irm_penalty
        
        # 对 behavior_reg 列表中的值求和
        behavior_reg_avg = torch.sum(torch.stack(behavior_reg, dim=0))
        reg_loss = self.args.reg_coeff *behavior_reg_avg

        # total_behavior_loss = behaviors_coeff * total_behavior_loss
        # main_loss = main_coeff * main_loss
        cl_loss = cl_loss * self.args.lambda_cl
        # 打印每个损失值
        print(f"loss: {main_loss.item(),cl_loss.item(),penalty_irm_loss.item(),reg_loss.item()}", flush=True)
               
        
        total_loss = main_loss  + cl_loss + penalty_irm_loss + reg_loss

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
           
            user_embedding = self.user_embedding.weight
            item_embedding = self.item_embedding.weight

            all_user_embeddings, all_item_embeddings,final_user_embs,final_item_embs = self.gcn_propagate(user_embedding,item_embedding)
            # target_embeddings = all_user_embeddings[:, -1].unsqueeze(1)
            # self.storage_user_embeddings = target_embeddings.squeeze() + user_embedding
            # self.storage_item_embeddings = torch.sum(all_item_embeddings, dim=1) 
            self.storage_user_embeddings = final_user_embs
            self.storage_item_embeddings = final_item_embs

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores


     #计算正则化损失（Regularization Loss）
    def calculate_reg_loss(self, user_feature,positem_feature,negItem_feature,batchlength):

        # all_user_emb = self.user_emb.weight
        # all_item_emb = self.item_emb.weight

        reg_embedding_loss = (1 / 2) * (user_feature.norm(2).pow(2) + positem_feature.norm(2).pow(2) + negItem_feature.norm(2).pow(2)) / float(batchlength)

        return reg_embedding_loss

