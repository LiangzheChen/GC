import torch
import utils
from utils import timer

def Train_process(dataset, recommend_model, GC , config):
    Recmodel = recommend_model
    Recmodel.train()


    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(config['device'])
    posItems = posItems.to(config['device'])
    negItems = negItems.to(config['device'])
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(config,users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config['bpr_batch_size'])):
        cri = GC.Fit_One(batch_users, batch_pos, batch_neg)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"Loss{aver_loss:.3f}-{time_info}"



