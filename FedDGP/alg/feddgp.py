from alg.fedavg import fedavg

class fedbn(fedavg):
    def __init__(self,args):
        super(fedbn, self).__init__(args)
import torch
import copy
import torch.nn as nn
import torch.optim as optim

from util.modelsel import modelsel
from util.traineval import train, test, train_balance, my_train, my_test
from alg.core.comm import communication, communication_cycle, communication_cycle_nbn

class feddgp(torch.nn.Module):
    def __init__(self, args):
        super(feddgp, self).__init__()
        self.server_model, self.p_model, self.g_model, self.client_weight = modelsel(
            args, args.device)
        self.p_server_model = copy.deepcopy(self.server_model)
        self.p_optimizers = [optim.SGD(params=self.p_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.g_optimizers = [optim.SGD(params=self.g_model[idx].parameters(
        ), lr=args.lr) for idx in range(args.n_clients)]
        self.loss_fun = nn.CrossEntropyLoss()
        self.args = args

    def client_train(self, c_idx, dataloader, round):
        train_g_loss, train_g_acc = train_balance(self.g_model[c_idx], dataloader, self.g_optimizers[c_idx], self.args.device, self.args)
        train_p_loss, train_p_acc = my_train(
            self.p_model[c_idx], self.g_model[c_idx], dataloader, self.p_optimizers[c_idx], self.loss_fun,
            self.args.device, self.args, round)

        return train_p_loss, train_p_acc

    def server_aggre(self, round):
        _, self.p_model = communication_cycle(
            self.args, self.server_model, self.p_model, self.client_weight, round)
        self.server_model, self.g_model = communication_cycle_nbn(
            self.args, self.server_model, self.g_model, self.client_weight, round)

    def client_eval(self, c_idx, dataloader):
        train_loss, train_acc = my_test(
             self.p_model[c_idx], self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc

    def server_eval(self, dataloader):
        train_loss, train_acc = test(
            self.server_model, dataloader, self.loss_fun, self.args.device)
        return train_loss, train_acc




