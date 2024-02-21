import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os, sys, glob
import cmoe_model as moe
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_file = 'best_model.ckpt'
test_dir = sys.argv[1]
name_output = 'confusion_matrix.npy'
kindnum = 9


class PrepareData(Dataset):
    def __init__(self, data_dir):
        self.data = glob.glob(os.path.join(data_dir, '*.npy'))
        self.map = {'Dredger':'0',
                    'Fishboat':'1',
                    'Motorboat':'2',
                    'Musselboat':'3',
                    'Naturalnoise':'4',
                    'Oceanliner':'5',
                    'Passengers':'6',
                    'RORO':'7',
                    'Sailboat':'8'
                    }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        npy_name = self.data[index]
        feature = np.load(npy_name)
        feature_torch = torch.from_numpy(feature).unsqueeze(0).float()
        type_name = npy_name.split('/')[-1].split('_')[0]
        label = int(self.map[type_name])
        return feature_torch, label





def main():
    model = moe.ResNet_MoE_loadbalance(channel=1, expert_num=8, output_dim=9, activate='sigmoid', balance_wt=0.01)
    print(model)
    model = model.cuda()
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'model.', ''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    test_loader = DataLoader(PrepareData(test_dir), batch_size=1, num_workers=8, shuffle=False)
    test(test_loader, model)
    



def test(test_loader, model):
    print('Testing:')
    top1 = AverageMeter()
    top2 = AverageMeter()
    model.eval()
    pred_all = []
    label_all = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            label_all.append(target)
            input, target = input.cuda(), target.cuda()
            output, loss_balance = model(input)
            pred_all.append(output.cpu())
            prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
            top1.update(prec1.item(), input.size(0))
            top2.update(prec2.item(), input.size(0))

    print('9-class Accuracy Top1: ', top1.avg)
    print('9-class Accuracy Top2: ', top2.avg)
    cmtx = get_confusion_matrix(pred_all, label_all, kindnum)
    print('Confusion Matrix: \n', cmtx)
    np.save(name_output, cmtx)
    tb_writer = SummaryWriter('Tensorboard')
    add_confusion_matrix(tb_writer,cmtx,num_classes=kindnum,class_names=['Dredger','Fishboat','Motorboat','Musselboat','Naturalnoise','Oceanliner','Passengers','RORO','Sailboat'], tag="Train Confusion Matrix",figsize=[kindnum,kindnum],)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True,
                          True)  # Returns two parameters : one returns the maximum value in output, and the other returns the index of the value
    pred = pred.t()
    correct = pred.eq(
        target.contiguous().view(1, -1).expand_as(pred))  # correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)  # correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_confusion_matrix(preds, labels, num_classes, normalize="true"):

    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))
    return cmtx

def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):

    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)


    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


 
def add_confusion_matrix(
    writer,
    cmtx,
    num_classes,
    global_step=None,
    subset_ids=None,
    class_names=None,
    tag="Confusion Matrix",
    figsize=None,
    ):

    if subset_ids is None or len(subset_ids) != 0:

        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]

        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
        )
        
        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)
        print('Done!')


if __name__ == '__main__':
    main()
