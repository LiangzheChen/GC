import utils
import time
import torch
from Data_loader import data_loader
from test import Rec_test
from train import Train_process
from model import GC, GC_FIT

if __name__ == "__main__":
    seed = 2020
    Neg_k = 1
    num_epochs = 1000
    weight_file = 'best_state.pt'


    config_dict ={ 'device': 'cuda:0','data_path': 'data/',
             'keep_prob': 0.6, 'A_n_fold': 100, 'test_u_batch_size': 100, 'multicore': 0, 'lr': 0.001,
             'bpr_batch_size': 2048, 'latent_dim': 512, 'content_dim': 16, 'n_layers': 3, 'dropout': 0,
             'decay': 0.0001, 'pretrain': 0, 'A_split': False, 'bigdata': False}

    device = config_dict['device']
    utils.set_seed(seed)
    dataset = data_loader(config_dict)
    model = GC(config_dict, dataset)
    model.to(device)

    GC_fit = GC_FIT(model, config_dict)

    #Traning

    for epoch in range(num_epochs):
        start = time.time()
        if epoch %10 == 0:
            Rec_test(model)

        output_information = Train_process(dataset, model, GC_fit, config_dict)
        print(f'Epoch[{epoch+1}/{num_epochs}] {output_information}')
        torch.save(model.state_dict(), weight_file)




