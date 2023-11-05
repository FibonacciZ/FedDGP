import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict

from datautil.datasplit import define_pretrain_dataset
from datautil.prepare_data import get_whole_dataset

def my_train(pmodel, gmodel, data_loader, optimizer, loss_fun, device, args, round):
    pmodel.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        if round == 0:
            g_f=0
        else:
            g_f = gmodel.get_sel_fea(data).detach()
        output = pmodel(data, g_f)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def train_balance_sup(pmodel, gmodel, data_loader, optimizer, device, args, round):
    pmodel.train()
    loss_all = 0
    total = 0
    correct = 0
    sample_per_class = torch.zeros(args.num_classes)
    for x, y in data_loader:
        for yy in y:
            sample_per_class[yy.item()] += 1
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        if round == 0:
            g_f=0
        else:
            g_f = gmodel.get_sel_fea(data).detach()
        output = pmodel(data, g_f)
        loss = balanced_softmax_loss(target, output, sample_per_class)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total

def train_balance(model, data_loader, optimizer, device, args):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    sample_per_class = torch.zeros(args.num_classes)
    for x, y in data_loader:
        for yy in y:
            sample_per_class[yy.item()] += 1
    for data, target in data_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        output = model(data)
        loss = balanced_softmax_loss(target, output, sample_per_class)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
        spc = sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
        return loss

def test(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total

def my_test(model, gmodel, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device).float()
            target = target.to(device).long()
            g_f = gmodel.get_sel_fea(data).detach()
            output = model(data, g_f)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total






def pretrain_model(args, model, filename, device='cuda'):
    print('===training pretrained model===')
    data = get_whole_dataset(args.dataset)(args)
    predata = define_pretrain_dataset(args, data)
    traindata = torch.utils.data.DataLoader(
        predata, batch_size=args.batch, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    opt = optim.SGD(params=model.parameters(), lr=args.lr)
    for _ in range(args.pretrained_iters):
        _, acc = train(model, traindata, opt, loss_fun, device)
    torch.save({
        'state': model.state_dict(),
        'acc': acc
    }, filename)
    print('===done!===')
