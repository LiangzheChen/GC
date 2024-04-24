import torch
import numpy as np
from sklearn.metrics import roc_auc_score
def dcg_at_k(r, k, method=0):
    # method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
    #         If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
def precision_and_recall_k(user_emb, item_emb, train_user_list, test_user_list, usr_auc, pos_auc, neg_auc, klist, batch=512):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k

    """
    # Calculate max k value
    y_score = None
    #AUC
    for i in range(0, len(usr_auc), batch):
        usr_ids = usr_auc[i:i + min(batch, len(usr_auc) - i)]
        item_ids = pos_auc[i:i + min(batch, len(usr_auc) - i)]
        neg_ids = neg_auc[i:i + min(batch, len(usr_auc) - i)]

        usr_t = user_emb[usr_ids]
        item_t = item_emb[item_ids]
        item_t = item_emb[item_ids]

        cur_result = torch.mul(usr_t, item_t)
        cur_score = torch.sum(cur_result, dim=1)
        # cur_score = torch.sigmoid(cur_score)

        # cur_result = torch.mm(usr_t, item_t.t())
        # cur_score = torch.sigmoid(cur_result)
        y_score = cur_score if y_score is None else torch.cat((y_score, cur_score), dim=0)
    real_score = torch.ones(y_score.shape)
    neg_score =  torch.zeros(y_score.shape)
    real_score = torch.cat((real_score, neg_score), dim=0)

    for i in range(0,len(usr_auc), batch):
        usr_ids = usr_auc[i:i + min(batch, len(usr_auc) - i)]
        item_ids = neg_auc[i:i + min(batch, len(usr_auc) - i)]

        usr_t = user_emb[usr_ids]
        item_t = item_emb[item_ids]

        cur_result = torch.mul(usr_t, item_t)
        cur_score = torch.sum(cur_result, dim=1)
        # cur_score = torch.sigmoid(cur_score)

        y_score = cur_score if y_score is None else torch.cat((y_score, cur_score), dim=0)
    AUC = roc_auc_score(real_score.reshape(-1), y_score.reshape(-1))
    max_k = max(klist)

    # Compute all pair of training and test record
    result = None
    # y_score = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i+j])), value=torch.tensor(0.0))
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        cur_score, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
        # y_score = cur_score if y_score is None else torch.cat((y_score, cur_score), dim=0)

    result = result.cpu()
    # y_score = y_score.cpu()
    # real_score = torch.zeros(y_score.shape)
    # Sort indice and get test_pred_topk
    precisions, recalls = [], []
    F1s =[]
    ndcgs =[]

    for k in klist:
        ndcg = 0
        precision, recall = 0, 0
        count  = 0

        for i in range(user_emb.shape[0]):
            r = [0] * k
            test = set(test_user_list[i])
            if len(test)<1:
                continue
            count+=1
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / max([min([k, len(test)]), 1])
            recall += val / max([len(test), 1])

            if val>0:
                # indexs = []
                pred_list = result[i, :k].numpy().tolist()
                val_set = test & pred
                for ele in val_set:
                    index = pred_list.index(ele)
                    # real_score[i, index] = 1
                    r[index] =1

            ndcg += ndcg_at_k(r, k)


            # if k==10:
            #     pos = [0]*k
            #     fla = [1]*k
            #     if val>0:
            #         for i in range(val):
            #             pos[i] =1

                # ndcg =metrics.ndcg_at_k(pos, k)
                # FPR, TPR, thresholds = metrics.roc_curve(pos, fla)
                # AUC = roc_auc_score(pos, fla)


        precisions.append(precision /count)
        recalls.append(recall / count)
        ndcgs.append(ndcg/count)
        pre = precision / count
        rec = recall / count
        f1 = (1 / pre + 1 / rec) / 2
        f1 = 1 / f1
        F1s.append(f1)
    print('F1: ' +str(F1s))
    print('ndcg: ' + str(ndcgs))
    print('precisions: ' + str(precisions))
    print('recalls: ' + str(recalls))
    print('AUC: ' +str(AUC))



    return precisions, recalls
def Rec_test(Recmodel):
    path = 'data/'
    Recmodel = Recmodel.eval()
    batch = 512
    usr_auc, pos_auc, neg_auc  = torch.load(path+'U_R_fortest.pt')
    all_users, all_items, _ = Recmodel.computer()
    all_items1 = []
    for i in range(0, all_items.shape[0], batch):
        tamp_item =  all_items[i:i + min(batch, all_items.shape[0] - i), :]
        pos_img = Recmodel.recipe_img[i:i + min(batch, all_items.shape[0] - i)].to('cuda:0')
        pos_img = Recmodel.img_net(pos_img)
        tamp_item = torch.cat([tamp_item,pos_img],dim=1)

        all_items1.append( tamp_item)
    all_items1 = torch.cat(all_items1).reshape(-1,all_users.shape[1]+pos_img.shape[1])

    all_users1 = torch.cat([all_users,Recmodel.embedding_visu_user.weight],dim=1)
    train_user_list, test_user_list, train_pair = torch.load(path+ 'torch_dataset.pt')
    precision_and_recall_k(all_users1.detach().to('cpu'),
                           all_items1.detach().to('cpu'),
                           train_user_list,
                           test_user_list,usr_auc, pos_auc, neg_auc,
                           klist=[1,2,3,4,5,6,7,8,9,10])