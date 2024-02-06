# @Time   : 2022/3/23
# @Author : Zeyu Zhang
# @Email  : wfzhangzeyu@163.com

"""
recbole.MetaModule.model.MeLU
##########################
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from recbole.model.layers import MLPLayers
from recbole.utils import InputType, FeatureSource, FeatureType
from recbole_ccml.MetaRecommender import MetaRecommender
from recbole_ccml.MetaUtils import GradCollector,EmbeddingTable
from recbole_ccml.model.CCML.VAE import VAE
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class CCML(MetaRecommender):

    input_type = InputType.POINTWISE

    def __init__(self,config,dataset):
        super(CCML, self).__init__(config, dataset)

        self.MLPHiddenSize = config['mlp_hidden_size']
        self.localLr = config['melu_args']['local_lr']
        self.localUpdateCount = config['local_update_count']
        self.similar_task_weight = config['similar_task_weight']

        self.profile_list = config['profile_list']
        self.profile_num = config['profile_num']

        self.user_fields = ['age', 'gender', 'occupation', 'zip_code']
        self.item_fields = ['release_year']
        self.profile_num = [0, 8, 11, 33]
        self.user_profile_num = [8, 3, 22, 2167]

        self.rating=6

        self.embeddingTable = EmbeddingTable(self.embedding_size,self.dataset)

        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

        self.model = VAE(config, self.embeddingTable.getUserEmdDim(), self.embeddingTable.getItemEmdDim())

        self.metaGradCollector = GradCollector(list(self.state_dict().keys()))

        self.keepWeightParams = deepcopy(self.model.state_dict())

        self.sample = False


    def taskDesolveEmb(self,task, paramWeightDict=None):
        spt_x=self.embeddingTable.embeddingAllFields(task.spt, paramWeightDict=paramWeightDict)
        spt_y = task.spt[self.RATING].view(-1, 1)
        qrt_x = self.embeddingTable.embeddingAllFields(task.qrt, paramWeightDict=paramWeightDict)
        qrt_y = task.qrt[self.RATING].view(-1, 1)
        return spt_x,spt_y,qrt_x,qrt_y

    def fieldsEmb(self,interaction):
        return self.embeddingTable.embeddingAllFields(interaction)

    def forward(self,spt_x,spt_y,qrt_x):

        # Copy meta parameters into model.
        self.model.load_state_dict(self.keepWeightParams)

        for i in range(self.localUpdateCount):
            originWeightParams = list(self.model.state_dict().values())
            paramNames = self.model.state_dict().keys()
            fastWeightParams = OrderedDict()

            # Calculate task-specific parameters.
            vae_loss = self.model.calculate_loss(spt_x, spt_y)
            self.model.zero_grad()
            grad = torch.autograd.grad(vae_loss, self.model.parameters())

            for index, name in enumerate(paramNames):
                fastWeightParams[name] = originWeightParams[index] - self.localLr * grad[index]

            # Load task-specific parameters and make prediction.
            self.model.load_state_dict(fastWeightParams)

        qrt_y_predict, _, _ = self.model(qrt_x)

        self.model.load_state_dict(self.keepWeightParams)

        return qrt_y_predict


    def sample_similar_task(self, similar_task_set, similar_task_emb, current_task_grad, target_task, target_spt_emb):
        similarity = []
        for spt_x, spt_y, _, _ in similar_task_emb:
            vae_loss = self.model.calculate_loss(spt_x, spt_y)
            localLoss = vae_loss
            self.model.zero_grad()
            temp_grad = torch.autograd.grad(localLoss, self.model.parameters(), create_graph=True, retain_graph=True)
            similarity.append(self.cosine(torch.cat([g.contiguous().view(-1) for g in temp_grad]).unsqueeze(0),
                                          torch.cat([g.contiguous().view(-1) for g in current_task_grad]).unsqueeze(0)).item())

        combined_list = list(zip(similar_task_set, similarity))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        # 获取相似度最高的前5个任务
        top_20_tasks = [task for task, similarity in sorted_combined_list[:20]]
        top_10_tasks = self.rating_filter(top_20_tasks, similar_task_emb, target_task, target_spt_emb)
        top_100_neg_tasks = [task for task, similarity in sorted_combined_list[-100:]]

        return top_10_tasks, top_100_neg_tasks


    def rating_filter(self, top_20_tasks, similar_task_emb, target_task, target_spt_emb):
        target_rating = []
        item_emb = target_spt_emb[:, (len(self.user_fields) * self.embedding_size):]
        for rating in [1, 2, 3, 4, 5]:
            indices = (target_task.spt['rating'] == rating).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                target_rating.append(torch.zeros_like(item_emb.mean(dim=0)))
            else:
                target_rating.append(item_emb[indices].mean(dim=0))
        target_rating = torch.stack(target_rating)

        rating_embeddings = []
        for task, task_emb in zip(top_20_tasks, similar_task_emb):
            temp = []
            item_emb = task_emb[0][:, (len(self.user_fields) * self.embedding_size):]
            for rating in [1, 2, 3, 4, 5]:
                indices = (task.spt['rating'] == rating).nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    temp.append(torch.zeros_like(item_emb.mean(dim=0)))
                else:
                    temp.append(item_emb[indices].mean(dim=0))
            rating_embeddings.append(torch.stack(temp))

        omegas = []
        for sim_task_rating in rating_embeddings:
            omega = 0.0
            for i in range(5):
                for j in range(i + 1, 5):
                    # 计算点积
                    dot_product = torch.dot(target_rating[i], sim_task_rating[j])
                    # 计算两个向量的范数乘积
                    norm_product = torch.norm(target_rating[i]) * torch.norm(sim_task_rating[j])
                    # 索引的差
                    if norm_product.item() == 0:
                        continue
                    index_diff = j - i
                    # 累加ω的值
                    omega += (dot_product / norm_product) * index_diff
            omegas.append(omega)

        combined_list = list(zip(top_20_tasks, omegas))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        # 获取相似度最高的前5个任务
        top_10_tasks = [task for task, similarity in sorted_combined_list[:10]]
        return top_10_tasks

    def co_relation_GNN(self, task, similar_tasks):
        user_field_num = 0
        item_field_num = 0
        user_profile = []
        item_profile = []
        all_similar_task = [task] + similar_tasks
        for field in self.user_fields:
            user_field_num += self.dataset.num(field) if self.dataset.field_num_dict is None else self.dataset.field_num_dict[field]

        for i_field in self.item_fields:
            item_field_num += self.dataset.num(i_field) if self.dataset.field_num_dict is None else self.dataset.field_num_dict[i_field]
            for index, u_field in enumerate(self.user_fields):
                user_profile.append(torch.cat([(t.spt[u_field] + self.profile_num[index]) for t in all_similar_task]))
                item_profile.append(torch.cat([t.spt[i_field] for t in all_similar_task]))

        # 初始化邻接矩阵
        user_inter = torch.cat(user_profile)
        item_inter = torch.cat(item_profile)
        UserItemNet = csr_matrix((torch.ones(len(user_inter)).cpu(), (user_inter.cpu(), item_inter.cpu())),
                                 shape=(user_field_num, item_field_num))
        adjacency_matrix = torch.zeros((user_field_num, item_field_num), device=self.device)

        adj_mat = sp.dok_matrix((user_field_num + item_field_num, user_field_num + item_field_num), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = UserItemNet.tolil()
        adj_mat[:user_field_num, user_field_num:] = R
        adj_mat[user_field_num:, :user_field_num] = R.T
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        graph = graph.coalesce().to(self.device)

        users_emb = self.embeddingTable.getEmdWeight(self.user_fields)
        items_emb = self.embeddingTable.getEmdWeight(self.item_fields)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(3):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [user_field_num, item_field_num])

        paramdict = {}
        for weight, field in zip(torch.split(users, self.user_profile_num), self.user_fields):
            paramdict[field] = weight

        # only one embedding matrix for items
        for field in self.item_fields:
            paramdict[field] = items
        # return the embedding
        return paramdict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


    def calculate_loss(self, taskBatch, train_data):
        totalLoss=torch.tensor(0.0).to(self.config.final_config_dict['device'])
        for task in taskBatch:
            similar_user = task.spt['similar_user']
            similar_task_set = [train_data.taskDict[k.item()] for k in similar_user]
            similar_task_emb = [self.taskDesolveEmb(k) for k in similar_task_set]
            spt_x, spt_y, qrt_x, qrt_y=self.taskDesolveEmb(task)

            # find the similar task of the current task
            vae_loss = self.model.calculate_loss(spt_x, spt_y)
            localLoss = vae_loss
            self.model.zero_grad()
            temp_grad = torch.autograd.grad(localLoss, self.model.parameters(), create_graph=True, retain_graph=True)

            similar_task, neg_tasks = self.sample_similar_task(similar_task_set, similar_task_emb, temp_grad, task, spt_x)

            # Calculate FOMAML Loss
            self.model.load_state_dict(self.keepWeightParams)

            emb_dict = self.co_relation_GNN(task, similar_task)

            # get correlative embedding
            spt_x, spt_y, qrt_x, qrt_y = self.taskDesolveEmb(task, paramWeightDict=emb_dict)
            similar_task_emb = [self.taskDesolveEmb(k, paramWeightDict=emb_dict) for k in similar_task]
            neg_task_emb = [self.taskDesolveEmb(k, paramWeightDict=emb_dict) for k in neg_tasks]

            for i in range(self.localUpdateCount):
                originWeightParams = list(self.model.state_dict().values())
                modelParamNames = self.model.state_dict().keys()
                fastWeightParams = OrderedDict()

                # the training loss of target task
                vae_loss = self.model.calculate_loss(spt_x, spt_y)

                localLoss = vae_loss

                # the training loss of similar tasks
                sim_user_emb = []
                for s_task_spt_x, s_task_spt_y, _, _ in similar_task_emb:
                    sim_user_emb.append(s_task_spt_x[0, :(len(self.user_fields) * self.embedding_size)])
                    s_task_spt_loss = self.model.calculate_loss(s_task_spt_x, s_task_spt_y)
                    localLoss += self.similar_task_weight * s_task_spt_loss

                # the contrastive loss of target task and neg tasks
                target_user_emb = spt_x[0, :(len(self.user_fields) * self.embedding_size)].unsqueeze(0)
                neg_user_emb = torch.stack([neg_task[0][0, :(len(self.user_fields) * self.embedding_size)] for neg_task in neg_task_emb])
                sim_user_emb = torch.stack(sim_user_emb)

                tau = 10

                all_embs = torch.cat([sim_user_emb, neg_user_emb], dim=0)
                # 计算锚点与所有样本的余弦相似度
                similarity = torch.matmul(target_user_emb, all_embs.T) / tau  # 形状: (1, 110), 包括温度调节

                exp_scores = torch.exp(similarity)

                sum_exp_scores = torch.sum(exp_scores, dim=1, keepdim=True)

                pos_exp_score = exp_scores[:, :sim_user_emb.shape[0]]  # 取出与正样本相关的得分

                # 计算每个正样本对应的InfoNCE损失
                info_nce_loss = -torch.log(pos_exp_score / sum_exp_scores)

                # 由于存在多个正样本，我们可以取平均损失
                loss_cl = torch.mean(info_nce_loss)
                localLoss += self.similar_task_weight * loss_cl

                self.model.zero_grad()
                grad = torch.autograd.grad(localLoss, self.model.parameters(),create_graph=True,retain_graph=True)
                for index, name in enumerate(modelParamNames):
                    fastWeightParams[name] = originWeightParams[index] - self.localLr * grad[index]

                self.model.load_state_dict(fastWeightParams)

            qrt_y_predict, _, _ = self.model(qrt_x)

            loss=F.mse_loss(qrt_y_predict,qrt_y.float())

            self.model.zero_grad()
            gradModel=torch.autograd.grad(loss, self.model.parameters(),create_graph=True,retain_graph=True)
            self.embeddingTable.zero_grad()
            gradEmb = torch.autograd.grad(loss, self.embeddingTable.parameters(), create_graph=True, retain_graph=True)

            self.metaGradCollector.addGrad(gradEmb+gradModel)
            totalLoss+=loss.detach()

            self.model.load_state_dict(self.keepWeightParams)           # Params back
        self.metaGradCollector.averageGrad(self.config['train_batch_size'])
        totalLoss/=self.config['train_batch_size']
        return totalLoss,self.metaGradCollector.dumpGrad()

    def predict(self, spt_x, spt_y, qrt_x):

        spt_x = self.embeddingTable.embeddingAllFields(spt_x)
        spt_y = spt_y.view(-1, 1)
        qrt_x = self.embeddingTable.embeddingAllFields(qrt_x)

        predict_qrt_y = self.forward(spt_x,spt_y,qrt_x)

        return predict_qrt_y

