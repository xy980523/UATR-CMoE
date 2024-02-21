import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F


class ResNet_MoE(nn.Module):

    def __init__(self, base_model="resnet18", channel=1, expert_num=4,output_dim=9, activate='linear'):
        super(ResNet_MoE, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        model_pretrained = self.get_basemodel(base_model, channel)

        self.conv1 = model_pretrained.conv1
        self.bn1 = model_pretrained.bn1
        self.relu = model_pretrained.relu
        self.maxpool = model_pretrained.maxpool

        self.layer1 = model_pretrained.layer1
        self.layer2 = model_pretrained.layer2
        self.layer3 = model_pretrained.layer3
        self.layer4 = model_pretrained.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.routing = nn.Linear(model_pretrained.fc.in_features, expert_num)
        self.fc = nn.ModuleList()
        for i in range(expert_num):
            self.fc.append(nn.Linear(model_pretrained.fc.in_features,output_dim))
        self.output_dim = output_dim
        self.activate = activate

    def get_basemodel(self, model_name, channel):
        try:
            model = self.resnet_dict[model_name]
            model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            print('load successfully')
        except KeyError:
            print("Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512

        routing_score = self.routing(x)  # batch, expert_num

        if self.activate == 'softmax':
            routing_score = torch.softmax(routing_score, dim=-1)
        if self.activate == 'sigmoid':
            routing_score = torch.sigmoid(routing_score)

        expert_id = torch.argmax(routing_score, dim=1) # batch   value=expert_id

        logits = torch.zeros([x.shape[0], self.output_dim], dtype=torch.float, device='cuda')

        for i in range(x.shape[0]):
            logits[i,:] = self.fc[expert_id[i].int()](x[i,:])

        return logits

class ResNet_MoE_loadbalance(nn.Module):

    def __init__(self, base_model="resnet18", channel=1, expert_num=4,output_dim=9,activate='linear',balance_wt=0.01):
        super(ResNet_MoE_loadbalance, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        model_pretrained = self.get_basemodel(base_model, channel)

        self.conv1 = model_pretrained.conv1
        self.bn1 = model_pretrained.bn1
        self.relu = model_pretrained.relu
        self.maxpool = model_pretrained.maxpool

        self.layer1 = model_pretrained.layer1
        self.layer2 = model_pretrained.layer2
        self.layer3 = model_pretrained.layer3
        self.layer4 = model_pretrained.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.routing = nn.Linear(model_pretrained.fc.in_features, expert_num)
        self.fc = nn.ModuleList()
        for i in range(expert_num):
            self.fc.append(nn.Linear(model_pretrained.fc.in_features,output_dim))
        self.expert_num = expert_num
        self.output_dim = output_dim
        self.activate = activate
        self.balance_wt = balance_wt

    def get_basemodel(self, model_name, channel):
        try:
            model = self.resnet_dict[model_name]
            model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            print('load successfully')
        except KeyError:
            print("Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512

        routing_score = self.routing(x)  # batch, expert_num

        if self.activate == 'softmax':
            routing_score = torch.softmax(routing_score, dim=-1)
        if self.activate == 'sigmoid':
            routing_score = torch.sigmoid(routing_score)   # # batch, expert_num

        expert_id = torch.argmax(routing_score, dim=1)  # batch   value=expert_id
        expert_id_onehot = F.one_hot(expert_id, self.expert_num).float()  # batch, expert_num

        logits = torch.zeros([x.shape[0], self.output_dim], dtype=torch.float, device='cuda')


        for i in range(x.shape[0]):
            logits[i,:] = self.fc[expert_id[i].int()](x[i,:])

        fi = torch.mean(expert_id_onehot, dim=0) # 1, expert_num
        pi = torch.mean(routing_score, dim=0)  # 1, expert_num
        loss_balance = self.balance_wt * self.expert_num * torch.matmul(fi, pi)

        return logits, loss_balance







class ResNet_MoE_residual(nn.Module):

    def __init__(self, base_model="resnet18", channel=1, expert_num=4,output_dim=9,activate='linear',balance_wt=0.01):
        super(ResNet_MoE_residual, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        model_pretrained = self.get_basemodel(base_model, channel)

        self.conv1 = model_pretrained.conv1
        self.bn1 = model_pretrained.bn1
        self.relu = model_pretrained.relu
        self.maxpool = model_pretrained.maxpool

        self.layer1 = model_pretrained.layer1
        self.layer2 = model_pretrained.layer2
        self.layer3 = model_pretrained.layer3
        self.layer4 = model_pretrained.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.routing = nn.Linear(model_pretrained.fc.in_features, expert_num)
        self.fc = nn.Linear(model_pretrained.fc.in_features,output_dim)
        self.expert = nn.ModuleList()
        for i in range(expert_num):
            self.expert.append(nn.Linear(model_pretrained.fc.in_features,output_dim))
        self.expert_num = expert_num
        self.output_dim = output_dim
        self.activate = activate
        self.balance_wt = balance_wt

    def get_basemodel(self, model_name, channel):
        try:
            model = self.resnet_dict[model_name]
            model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            print('load successfully')
        except KeyError:
            print("Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512

        routing_score = self.routing(x)  # batch, expert_num

        if self.activate == 'softmax':
            routing_score = torch.softmax(routing_score, dim=-1)
        if self.activate == 'sigmoid':
            routing_score = torch.sigmoid(routing_score)   # # batch, expert_num

        expert_id = torch.argmax(routing_score, dim=1)  # batch   value=expert_id
        expert_id_onehot = F.one_hot(expert_id, self.expert_num).float()  # batch, expert_num
        fi = torch.mean(expert_id_onehot, dim=0)  # 1, expert_num
        pi = torch.mean(routing_score, dim=0)  # 1, expert_num
        loss_balance = self.balance_wt * self.expert_num * torch.matmul(fi, pi)

        logits = torch.zeros([x.shape[0], self.output_dim], dtype=torch.float, device='cuda')
        for i in range(x.shape[0]):
            logits[i,:] = self.expert[expert_id[i].int()](x[i,:]) + self.fc(x[i,:])

        return logits, loss_balance


if __name__ == '__main__':
    input = torch.randn(8, 1, 300, 300)

    '''
    # ResNet_MoE
    model = ResNet_MoE(base_model="resnet18", channel=1, expert_num=4, output_dim=9, activate='sigmoid')
    model, input = model.cuda(), input.cuda()
    output = model(input)
    print(output.shape)
    '''

    # ResNet_MoE with load balance
    model = ResNet_MoE_loadbalance(base_model="resnet18", channel=1, expert_num=4, output_dim=9, activate='sigmoid', balance_wt=0.01)
    model, input = model.cuda(), input.cuda()
    output, loss_balance = model(input)
    print(output.shape)
    print(loss_balance)

