from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from model import *
import argparse
import pickle


def _calc_mean_std(train_dataset, 
                   train_batch_gen):
    num_of_pixels = len(train_dataset) * 32 * 32
    channels_sum = [0., 0., 0.]
    for (X_batch, y_batch) in tqdm(train_batch_gen): 
        for i in range(3):
            channels_sum[i] += X_batch[:, i, :, :].sum()
    means = channels_sum / num_of_pixels
    sums_of_squared_error = [0., 0., 0.]
    for (X_batch, y_batch) in tqdm(train_batch_gen): 
        for i in range(3):
            sums_of_squared_error[i] += ((X_batch[:, i, :, :] - means[i]).pow(2)).sum()
    stds = torch.sqrt(sums_of_squared_error / num_of_pixels)
    return means, sts

def _create_dataset(use_cifar_10 : bool,
                    transform, 
                    dataset_path : str = None,
                    is_train : bool = True):
        
    if use_cifar_10:
        train_dataset = datasets.CIFAR10("data",
                                         transform=transform, 
                                         train=is_train, 
                                         download=True)
    else:
        train_dataset = datasets.ImageFolder(dataset_path, 
                                             transform=transform)
    return train_dataset
        
def _create_dataset_and_gen(train_dataset_path : str = None, 
                            val_dataset_path : str = None,
                            use_cifar_10 : bool = True, 
                            batch_size : int = 2):
    
    transforms_comb = [transforms.ToTensor()]
    
    if not use_cifar_10:
        transform_train = transforms.Compose(transforms)
        train_dataset = _create_dataset(use_cifar_10, 
                                        transform_train, 
                                        train_dataset_path)
        train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                                      batch_size=batch_size,
                                                      shuffle=True, 
                                                      num_workers=100)
        
        means, stds = _calc_mean_std(train_dataset, 
                                    train_batch_gen)
    else:
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.247, 0.243, 0.261]
    transforms_comb.append(transforms.Normalize((means[0], means[1], means[2]), 
                                                (stds[0], stds[1], stds[2])))    
    transform_train = transforms.Compose(transforms_comb)
    
    train_dataset = _create_dataset(use_cifar_10, 
                                    transform_train, 
                                    train_dataset_path)
    val_dataset = None
    val_batch_gen = None
    if use_cifar_10:
        total_ex_number = len(train_dataset)
        train_ex_number = int(total_ex_number * 4. / 5.)
        val_ex_number = total_ex_number - train_ex_number
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_ex_number, 
                                                                                   val_ex_number])
        val_batch_gen = torch.utils.data.DataLoader(val_dataset, 
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=10)
    
    train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True, 
                                                  num_workers=100)
    return train_dataset, train_batch_gen, means, stds, val_dataset, val_batch_gen

def _create_val_dataset_and_gen(means : list, 
                                stds : list, 
                                val_dataset_path : str = None,
                                use_cifar_10 : bool = True):
    
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((means[0], means[1], means[2]), 
                                                             (stds[0], stds[1], stds[2]))])
    val_dataset = _create_dataset(use_cifar_10, 
                                  transform_val, 
                                  val_dataset_path)
    val_batch_gen = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=32,
                                                shuffle=False, 
                                                num_workers=100)
    
    return val_dataset, val_batch_gen

def _get_accuracy(model : nn.Module,
                  batch_gen, 
                  use_cuda : bool, 
                  results : list):
    batch_i = 0
    for X_batch, y_batch in tqdm(batch_gen):
        torch.cuda.empty_cache()
        var = Variable(torch.FloatTensor(X_batch))
        if use_cuda:
            logits = model(var.cuda())
        else:
            logits = model(var)
        y_pred = logits.max(1)[1].data
        results += list((y_batch.cpu() == y_pred.cpu()).numpy())
        batch_i += 1
        
    acc = np.mean(results)
    return acc, results
        
def compute_loss(X_batch,
                 y_batch,
                 model,
                 use_cuda: bool = False):
    X_batch = Variable(torch.FloatTensor(X_batch))
    y_batch = Variable(torch.LongTensor(y_batch))
    if use_cuda:
        model.cuda()
        logits = model.cuda()(X_batch.cuda())
        return F.cross_entropy(logits, y_batch.cuda()).mean()
    else:
        logits = model(X_batch)
        return F.cross_entropy(logits, y_batch).mean()
        
def train(model : nn.Module,
          train_dataset_path : str = None, 
          val_dataset_path : str = None,
          use_cifar_10 : bool = True, 
          batch_size : int = 2,
          num_epoches : int = 100,
          use_tensorboard: bool = False, 
          use_cuda: bool = False):
    
    train_dataset, train_batch_gen, \
    means, stds, \
    val_dataset, val_batch_gen = _create_dataset_and_gen(train_dataset_path, 
                                                          val_dataset_path,
                                                          use_cifar_10, 
                                                          batch_size)
    
    do_validation = (use_cifar_10) or (val_dataset_path is not None)
    
    if val_dataset_path is not None:
        val_dataset, val_batch_gen = _create_val_dataset_and_gen(means, 
                                                                 stds, 
                                                                 val_dataset_path, 
                                                                 use_cifar_10)
    
    print("Datasets ready")

    if use_tensorboard:
        os.system('tensorboard --logdir=logs_mobileNetV3')
        
        writer = SummaryWriter('logs_mobileNetV3/training')
        
    loss_history_train = []
    accuracy_history_train = []
    accuracy_history_val = []
    
    if use_cuda:
        model.cuda()
    
    for epoch in range(num_epoches):
        print("Epoch number", epoch + 1)
        
        model.train(True)
        
        batch_i = 0
        for (X_batch, y_batch) in tqdm(train_batch_gen):
            loss = compute_loss(X_batch, y_batch, model, use_cuda)
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_history_train.append(loss.data.cpu().numpy())
            batch_i += 1
            
        loss_train = np.mean(loss_history_train[-len(train_dataset) // batch_size :])
        print("Train loss \t{:.6f}".format(loss_train))
        
        results_val = []
        results_train = []
        acc_val = 0.
        acc_train = 0.
        
        model.train(False)
        
        with torch.no_grad():
            if do_validation:
                acc_val, results_val = _get_accuracy(model, val_batch_gen, use_cuda, results_val)
                print("Val acc = ", acc_val)
                accuracy_history_val.append(acc_val)

            acc_train, results_train = _get_accuracy(model, train_batch_gen, use_cuda, results_train)
            print("Train acc = ", acc_train)
            accuracy_history_train.append(acc_train)
            
        if use_tensorboard:
            global_step = epoch
            values_dict = {'train_loss': loss_train, 
                           'train_acc': acc_train}
            if do_validation:
                values_dict['val_acc'] = acc_val
            writer.add_scalars('model', values_dict, global_step=global_step)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help="model type: Small or Large, default=Small")
    parser.add_argument("--num_classes", help="number of classes, default=10")
    parser.add_argument("--batch_size", help="dataloader batch_size, default=2")
    parser.add_argument("--num_epoches", help="number of epoches, default=50")
    parser.add_argument("--use_cuda", help="1 if cuda is used, 0 otherwise, default=0")
    parser.add_argument("--use_tensorboard", help="1 if log metrics with tensorboard, 0 otherwise, default=0")
    parser.add_argument("--train_dataset_path", help="path to the training set. If not specified, cifar-10 used. default=None")
    parser.add_argument("--val_dataset_path", help="path to the validation set. If not specified, cifar-10 used. default=None")
    args = parser.parse_args()

    num_classes = 10
    batch_size = 2
    use_cuda = 0
    use_tensorboard = 0
    train_dataset_path = None
    val_dataset_path = None
    use_cifar_10 = 1
    num_epoches=1

    if args.num_classes:
        num_classes = int(num_classes)

    if args.model_type and args.model_type not in ['Small', 'Large']:
        print("ERROR! Incorrect model_type")
    else:
        model = MobileNetV3_Small(num_classes=num_classes)
        if args.model_type:
            if args.model_type == 'Large':
                model = MobileNetV3_Large(num_classes=num_classes)
        if args.batch_size:
            batch_size = int(args.batch_size)
        if args.num_epoches:
            num_epoches = int(args.num_epoches)
        if args.use_cuda:
            use_cuda = int(args.use_cuda)
        if args.use_tensorboard:
            use_tensorboard = int(args.use_tensorboard)
        if args.train_dataset_path:
            train_dataset_path = args.train_dataset_path
            use_cifar_10 = 0
        if args.val_dataset_path:
            val_dataset_path = args.val_dataset_path
        print('Start')
        opt = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
        model = train(model,
                      batch_size=batch_size,
                      num_epoches=num_epoches,
                      use_cuda=use_cuda, 
                      use_tensorboard=use_tensorboard, 
                      train_dataset_path=train_dataset_path, 
                      val_dataset_path=val_dataset_path, 
                      use_cifar_10=use_cifar_10)
        torch.save(model, "mobilenetv3.pth")
        print('Finish')
