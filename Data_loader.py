from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from time import time
import numpy as np
import scipy.sparse as sp

import torch

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class data_loader(BasicDataset):


    def __init__(self,config):
        self.device = config['device']
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.path = config['data_path']
        train_file = self.path + '/train.txt'
        test_file = self.path + '/test.txt'
        self.ingre_net = torch.load(self.path + 'ingre_matrix.pt')


        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1

        self.n_ingre = self.ingre_net.shape[1]
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} for Training")
        print(f"{self.testDataSize} for Testing")

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getFoodGraph(self):
        print("Loading Food Graph")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/Food_pre_adj_mat.npz')
                print("Successfully Loading Food Graph...")
                norm_adj = pre_adj_mat
            except :
                print("Generating Food Graph...")
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items + self.n_ingre, self.n_users + self.m_items + self.n_ingre),
                    dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:self.m_items + self.n_users] = R
                adj_mat[self.n_users:self.m_items + self.n_users, :self.n_users] = R.T

                R1 = self.ingre_net.tolil()
                adj_mat[self.n_users:self.n_users + self.m_items, self.n_users + self.m_items:] = R1
                adj_mat[self.n_users + self.m_items:, self.n_users:self.n_users + self.m_items] = R1.T

                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"Costing {end - s}s, Saving Food Data...")
                sp.save_npz(self.path + '/Food_pre_adj_mat.npz', norm_adj)

            svd_q = 5
            adj1 = norm_adj[:self.n_users, self.n_users:self.m_items + self.n_users]
            adj1 = scipy_sparse_mat_to_torch_sparse_tensor(adj1).coalesce()
            # adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
            print('Performing GC...')
            svd_u, s, svd_v = torch.pca_lowrank(adj1, q=svd_q)
            u_mul_s = svd_u @ (torch.diag(s))
            v_mul_s = svd_v @ (torch.diag(s))

            adj2 = norm_adj[self.n_users:self.n_users + self.m_items, self.n_users + self.m_items:]
            adj2 = scipy_sparse_mat_to_torch_sparse_tensor(adj2).coalesce()
            # adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
            print('Performing GC...')
            svd_u2, s2, svd_v2 = torch.pca_lowrank(adj2, q=svd_q)
            u_mul_s2 = svd_u2 @ (torch.diag(s2))

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("Done split")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.device)

                print("Don't split")
        return self.Graph, u_mul_s.to(self.device), svd_v.T.to(self.device), v_mul_s.to(self.device), svd_u.T.to(
            self.device), u_mul_s2.to(self.device), svd_v2.T.to(self.device)
