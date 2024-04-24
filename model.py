from Data_loader import BasicDataset
from torch import nn, optim
import torch
import torch.nn.functional as F
import numpy as np


class GC_FIT:
    def __init__(self,
                 model,
                 config : dict):
        self.model = model
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def Fit_One(self, users, pos, neg):
        loss, reg_loss ,contrastive_loss= self.model.Get_loss(users, pos, neg)
        #contrastive
        # contrastive_loss = self.model.con_loss(users, pos, neg)

        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss + 0.1* contrastive_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

def  get_contrastive_loss(list1 ,list2):
    temperature = 0.07
    contrastive_loss = None
    for i in range(len(list1)):
        x1 = list1[i]
        x2=  list2[i]

        x1_user, x2_user = F.normalize(x1), F.normalize(x2)
        pos_score_user = torch.mul(x1_user, x2_user).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / temperature)

        x2_user_neg = torch.flipud(x2_user)
        ttl_score_user = torch.mul(x1_user, x2_user_neg).sum(dim=1)
        ttl_score_user = pos_score_user + torch.exp(ttl_score_user / temperature)
        loss = - torch.log(pos_score_user/ttl_score_user).mean()
        contrastive_loss = loss if contrastive_loss is None else contrastive_loss + loss

    return contrastive_loss

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class GC(BasicModel):
    def __init__(self,
                 config:dict,
                 dataset: BasicDataset):
        super(GC, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.num_ingre =  self.dataset.n_ingre
        self.latent_dim = self.config['latent_dim']
        self.content_dim = self.config['content_dim']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        # img
        self.recipe_img = torch.load(self.config['data_path']+'img_tensor.pt')
        self.img_net = nn.Sequential(
            nn.Linear(2048, self.content_dim),
            # nn.Sigmoid(),
            nn.ReLU(),
        )

        self.embedding_visu_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.content_dim)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_ingre = torch.nn.Embedding(
            num_embeddings=self.num_ingre, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_visu_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.normal_(self.embedding_ingre.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))

        self.f = nn.Sigmoid()
        self.Graph, self.u_mul_s, self.vt, self.v_mul_s, self.ut, self.u_mul_s2, self.vt2 = self.dataset.getFoodGraph()

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def node_drop(self, feats, drop_rate, training):
        n = feats.shape[0]
        drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

        if training:
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
            feats = masks.to(feats.device) * feats / (1. - drop_rate)
        else:
            feats = feats
        return feats

    def computer_con(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ingre_emb = self.embedding_ingre.weight
        all_emb = torch.cat([users_emb, items_emb, ingre_emb])
        # all_emb = self.node_drop(all_emb, 0.1, True)

        embs = [all_emb]
        if False:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items, ingres = torch.split(light_out, [self.num_users, self.num_items, self.num_ingre])
        return users, items, ingres

    def computer_con1(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ingre_emb = self.embedding_ingre.weight
        all_emb = torch.cat([users_emb, items_emb, ingre_emb])
        # all_emb = self.node_drop(all_emb, 0.1, True)

        embs = [all_emb]
        # embs1 = [all_emb]
        usr_embs = [users_emb]
        item_embs = [items_emb]

        if False:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

            usr_emb = self.vt @ embs[layer][self.num_users:self.num_users + self.num_items]
            usr_emb = (self.u_mul_s @ usr_emb)

            usr_embs.append(usr_emb)

            item_emb = self.ut @ embs[layer][0:self.num_users]
            item_emb = (self.v_mul_s @ item_emb)

            item_emb2 = self.vt2 @ embs[layer][
                                   self.num_users + self.num_items:self.num_users + self.num_items + self.num_ingre]
            item_emb += (self.u_mul_s2 @ item_emb2)

            item_embs.append(item_emb)

        usr_embs = torch.stack(usr_embs, dim=1)
        usr_embs = torch.mean(usr_embs, dim=1)

        item_embs = torch.stack(item_embs, dim=1)
        item_embs = torch.mean(item_embs, dim=1)

        # print(embs.size())
        # light_out1 = torch.mean(embs1, dim=1)
        # users1, items1, ingres1 = torch.split(light_out1, [self.num_users, self.num_items, self.num_ingre])

        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items, ingres = torch.split(light_out, [self.num_users, self.num_items, self.num_ingre])

        return users, items, ingres, usr_embs, item_embs

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ingre_emb = self.embedding_ingre.weight
        all_emb = torch.cat([users_emb, items_emb, ingre_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items, ingres = torch.split(light_out, [self.num_users, self.num_items, self.num_ingre])
        return users, items, ingres

    def getUsersRating(self, users):
        all_users, all_items, all_ingres = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def get_conEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, all_ingres = self.computer_con()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb

    def get_conEmbedding1(self, users, pos_items, neg_items):
        all_users, all_items, all_ingres = self.computer_con1()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, all_ingres, all_users1, all_items1 = self.computer_con1()

        users_emb1 = all_users1[users]
        pos_emb1 = all_items1[pos_items]
        neg_emb1 = all_items1[neg_items]

        list1 = [users_emb1, pos_emb1, neg_emb1]

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        list = [users_emb, pos_emb, neg_emb]
        # img
        pos_img = self.recipe_img[pos_items].to('cuda:0')
        pos_img = self.img_net(pos_img)
        pos_emb = torch.cat([pos_emb, pos_img], dim=1)

        # img
        neg_img = self.recipe_img[neg_items].to('cuda:0')
        neg_img = self.img_net(neg_img)
        neg_emb = torch.cat([neg_emb, neg_img], dim=1)

        # USR
        users_vis = self.embedding_visu_user(users)
        users_emb = torch.cat([users_emb, users_vis], dim=1)

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        ingre_emb_ego = self.embedding_ingre.weight
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, ingre_emb_ego, list, list1

    def Get_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0, ingre_emb_ego, list, list1) = self.getEmbedding(users.long(), pos.long(),
                                                                                     neg.long())
        reg_loss = (1 / 3) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        reg_loss += (1 / 3) * (ingre_emb_ego.norm(2).pow(2)) / float(self.num_ingre)
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        contrastive_loss = get_contrastive_loss(list, list1)
        return loss, reg_loss, contrastive_loss

    def con_loss(self, users, pos, neg):
        emb_list1 = self.get_conEmbedding(users.long(), pos.long(), neg.long())
        emb_list2 = self.get_conEmbedding1(users.long(), pos.long(), neg.long())

        contrastive_loss = get_contrastive_loss(emb_list1, emb_list2)

        return contrastive_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items, all_ingre = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma