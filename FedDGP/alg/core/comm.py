# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

def communication(args, server_model, models, client_weights):
    client_num=len(models)
    with torch.no_grad():
        if args.alg.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower()=='fedap':
            tmpmodels=[]
            for i in range(client_num):
                tmpmodels.append(copy.deepcopy(models[i]).to(args.device))
            with torch.no_grad():
                for cl in range(client_num):
                    for key in server_model.state_dict().keys():
                        temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                        for client_idx in range(client_num):
                            temp += client_weights[cl,client_idx] * tmpmodels[client_idx].state_dict()[key]
                        server_model.state_dict()[key].data.copy_(temp)
                        if 'bn' not in key:
                            models[cl].state_dict()[key].data.copy_(server_model.state_dict()[key])
        elif args.alg.lower() == 'fedsim':
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp_avg = torch.zeros_like(server_model.state_dict()[key])
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp_avg += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    cosim_list = [F.cosine_similarity(torch.flatten(models[client_idx].state_dict()[key]), torch.flatten(temp_avg), dim=0)
                                                      for client_idx in range(len(client_weights))]
                    cosim_list = torch.stack(cosim_list)
                    softmax = nn.Softmax(0)
                    client_new_weights = softmax(cosim_list)
                    for client_idx in range(len(client_new_weights)):
                        temp += client_new_weights[client_idx] * models[client_idx].state_dict()[key]

                    server_model.state_dict()[key].data.copy_(temp)

                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def communication_cycle(args, server_model, models, client_weights, round):
    for key in server_model.state_dict().keys():
        if 'num_batches_tracked' in key:
            server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
        else:
            if round<args.cround:
                temp = models[round%args.n_clients].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                if 'bn' not in key:
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            else:
                temp_avg = torch.zeros_like(server_model.state_dict()[key])
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp_avg += client_weights[client_idx] * models[client_idx].state_dict()[key]
                cosim_list = [F.cosine_similarity(torch.flatten(models[client_idx].state_dict()[key]), torch.flatten(temp_avg),dim=0)
                                for client_idx in range(len(client_weights))]
                cosim_list = torch.stack(cosim_list)
                softmax = nn.Softmax(0)
                client_new_weights = softmax(cosim_list)
                for client_idx in range(len(client_new_weights)):
                    temp += client_new_weights[client_idx] * models[client_idx].state_dict()[key]

                server_model.state_dict()[key].data.copy_(temp)
                if 'bn' not in key:
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

def communication_cycle_nbn(args, server_model, models, client_weights, round):
    for key in server_model.state_dict().keys():
        if 'num_batches_tracked' in key:
            server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
        else:
            if round<args.cround:
                temp = models[round%args.n_clients].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            else:
                temp_avg = torch.zeros_like(server_model.state_dict()[key])
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp_avg += client_weights[client_idx] * models[client_idx].state_dict()[key]
                cosim_list = [F.cosine_similarity(torch.flatten(models[client_idx].state_dict()[key]), torch.flatten(temp_avg),dim=0)
                                for client_idx in range(len(client_weights))]
                cosim_list = torch.stack(cosim_list)
                softmax = nn.Softmax(0)
                client_new_weights = softmax(cosim_list)
                for client_idx in range(len(client_new_weights)):
                    temp += client_new_weights[client_idx] * models[client_idx].state_dict()[key]

                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

