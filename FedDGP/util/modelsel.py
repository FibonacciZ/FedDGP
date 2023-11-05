from network.models import AlexNet, PamapModel, lenet5v, lenet, resnet10, FedAvgCNN, LocalModel
import torch.nn as nn
import copy


def modelsel(args, device):
    if args.dataset in ['vlcs', 'pacs', 'officehome', 'off-cal', 'covid']:
        server_model = AlexNet(num_classes=args.num_classes).to(device)
    elif 'medmnist' in args.dataset:
        server_model = lenet().to(device)
        # server_model = lenet5v().to(device)
    elif 'pamap' in args.dataset:
        server_model = PamapModel().to(device)
    elif 'Cifar10' in args.dataset:
        server_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)

    client_weights = [1 / args.n_clients for _ in range(args.n_clients)]

    if args.alg.lower()=='fedper' or args.alg.lower()=='moon':
        model_head = copy.deepcopy(server_model.fc)
        server_model.fc = nn.Identity()
        server_model = LocalModel(server_model, model_head)
        model = [copy.deepcopy(server_model).to(device) for _ in range(args.n_clients)]
        return server_model, model, client_weights

    model = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    p_models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    g_models = [copy.deepcopy(server_model).to(device)
                for _ in range(args.n_clients)]
    model_head = [copy.deepcopy(server_model.fc).to(device)
              for _ in range(args.n_clients)]
    if args.alg.lower()=='feddgp':
        return server_model, p_models, g_models, client_weights
    else:
        return server_model, model, client_weights

