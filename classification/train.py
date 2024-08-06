
from colorama import Fore
import argparse 
import os
import random
import torch
import torch.nn.parallel
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from scripts.point_dataset import  ShapeObj_dataset
from scripts.h5_dataset import Dataset
from model import Model
from scripts.functions import save_checkpoint
from tqdm import tqdm
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument(
        '--run_name', type=str, default='adapt_1', help='training run name')
    parser.add_argument(
        '--desc', type=str, default='Trail 1.', help='description')
    parser.add_argument('--data_root', type=str, default='dataset/h5',
                        help='data root containing the data set folder (point clouds)')
    parser.add_argument('--logdir', type=str,
                        default='runs', help='training log folder')
    parser.add_argument('--refine', type=bool,
                        default=False, help='train again a pretrained model or not')
    parser.add_argument('--loaddir', type=str,
                        default='', help='path to model file (.pth.tar) to finetune')
    parser.add_argument('--saveinterval', type=int,
                        default=10, help='save model each n epochs')
    parser.add_argument('--cuda_idx', type=int,
                        default=0,help='cuda opt.cuda_idxindex to train on' )

    # dataset
    parser.add_argument('--dataset', type=str,
                        default='', help='modelnet40, modelnet10, shapeobj')
    parser.add_argument('--augment', type=bool,
                        default=True, help='Apply jitter and rotation to data for training')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle Dataset or not')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points of point clouds')
    parser.add_argument('--device', type=str, default='cpu',
                        help='"gpu" or "cpu"')
    parser.add_argument('--final_dim', type=int, default=1024,
                        help='final feature embedding dimension you want')
    parser.add_argument('--pool_factor', type=int, default=2,
                        help='pool factor for pooling graph (either 2 or 4)')
    parser.add_argument('--k_n', type=int, default=10,
                        help='number of neighbours to consider for a point')

    # training parameters
    parser.add_argument('--task', type=str, default='class',
                        help='"class" for Classification and "seg" for segmentation')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=600,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int,
                        default=-1, help='manual seed')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')

    # model hyperparameters
    parser.add_argument('--trans_loss', type=bool,
                        default=True, help='use transform loss')
    parser.add_argument('--alpha', type=int,
                        default=0.001, help='coeficient for transform loss')

    return parser.parse_args()


def check_path_existance(log_dirname):
    if os.path.exists(log_dirname):
        print(Fore.GREEN + "log directory : ", log_dirname)
        return
    else:
        os.mkdir(log_dirname)
        os.mkdir(log_dirname+'/val')
        os.mkdir(log_dirname+'/test')
        print(Fore.GREEN + "Created log directory :  ", log_dirname)

def compute_accuracy(conf_matrix):
    # Total number of samples
    total_samples = np.sum(conf_matrix)
    
    # Compute overall accuracy
    overall_accuracy = np.trace(conf_matrix) / total_samples
    
    # Compute mean accuracy
    class_acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    mean_accuracy = np.mean(class_acc)
    
    return round(overall_accuracy*100,2), round(mean_accuracy*100,2)

def get_data(opt, train=True):
    # create train and test dataset loaders

    if train:
        if (opt.dataset == 'modelnet40' or opt.dataset == 'modelnet10'):
            dataset = Dataset(root=opt.data_root,
                                dataset_name= opt.dataset,
                                random_jitter= opt.augment,
                                random_rotate=opt.augment,
                                random_translate=False
                                )
            print(Fore.GREEN+"ModelNet dataset Augmented")
            return dataset
        elif opt.dataset == 'shapeobj':
            dataset = ShapeObj_dataset(data_root=opt.data_root,
                                        transform=opt.augment,
                                        train=True,
                                        conv=False)
            print(Fore.GREEN+"ShapeObj dataset Augmented")
            return dataset
        else:
            print(Fore.RED+"Wrong dataset choosen! : ",opt.data_root)
            raise AssertionError
            
        
    else:
        if (opt.dataset == 'modelnet40' or opt.dataset == 'modelnet10'):
            dataset = Dataset(root=opt.data_root,
                                dataset_name= opt.dataset,
                                random_jitter= opt.augment,
                                random_rotate=opt.augment,
                                random_translate=False,
                                split = 'test'
                                )
            print(Fore.GREEN+"ModelNet dataset Augmented")
            return dataset
        elif opt.dataset == 'shapeobj':
            dataset = ShapeObj_dataset(data_root=opt.data_root,
                                        transform=opt.augment,
                                        train=False,
                                        conv=False)
            print(Fore.GREEN+"ShapeObj Test dataset Augmented")
            return dataset
        else:
            print(Fore.RED+"Wrong dataset choosen! : ",opt.data_root)
            raise AssertionError



def train_adpnet(opt):

    log_dirname = os.path.join(opt.logdir, opt.run_name)
    params_filename = os.path.join(log_dirname, "config.txt")

    check_path_existance(log_dirname)

    if (opt.dataset == 'modelnet40'):
        num_classes = 40
    elif (opt.dataset == 'modelnet10'):
        num_classes = 10
    else:
        num_classes = 15

    adpnet = Model(num_classes,
                            k_neighbours =opt.k_n,
                            index=opt.cuda_idx)
    adpnet = adpnet.cuda(opt.cuda_idx)
    
    
    print("Number of Model parameters: ", sum(p.numel() for p in adpnet.parameters()))
    optimizer = torch.optim.SGD(adpnet.parameters(), lr=opt.lr, momentum=opt.momentum)
    if opt.refine:
        if os.path.exists(opt.loaddir):
            last_model = torch.load(opt.loaddir)
            adpnet.load_state_dict(last_model['model_state_dict'])
            optimizer.load_state_dict(last_model['optimizer_state_dict'])
            max_acc = last_model['accuracy']
            max_test = last_model['test_accuracy']
            print(Fore.GREEN+"Model Loaded sucessfully : ", opt.loaddir)
        else:
            print(Fore.RED+"invalid load path for model params")
            return

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    print(Fore.GREEN+"Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # create train and test dataset loaders

    train_dataset = get_data(opt, train=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.workers),
        shuffle=True)

    test_dataset = get_data(opt, train=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.workers),
        shuffle=False, drop_last=False)

    print(Fore.GREEN+'training set: %d objects (in %d batches) /n' %
          (len(train_dataset), len(train_dataloader)))
    print(Fore.GREEN+'Testing set: %d objects (in %d batches) /n' %
          (len(test_dataset), len(test_dataloader)))


    # save parameters
    if (os.path.exists(params_filename)):
        torch.save(opt, params_filename)
    else:
        with open(params_filename, 'w') as fp:
            fp.close()
        torch.save(opt, params_filename)
    
    # milestones in number of optimizer iterations
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.nepochs, eta_min= opt.lr / 100, verbose=True)
      
    criterion = torch.nn.CrossEntropyLoss()

    train_loss = []
    train_acc = []
    test_acc = []
    embd =[]
    color = []

    if opt.refine:
        print(max_acc)
    else:
        max_acc = 0

    max_test = 0

    print(Fore.GREEN)
    for epoch in range(1, opt.nepochs + 1):

        print(Fore.GREEN)
        num_train_examples = train_dataset.__len__()
        num_train_batch = len(train_dataloader)
        num_test_examples = test_dataset.__len__()
        num_test_batch = len(test_dataloader)

        epoch_loss, correct, eval_correct, test_correct = 0, 0, 0, 0
        test_confusion = np.zeros((num_classes, num_classes))

        progress_bar = tqdm(
            range(num_train_batch),
            desc=f"Training Epoch {epoch}/{opt.nepochs}"
        )
        adpnet.train()
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data = data.transpose(1,2).cuda(opt.cuda_idx)
            label =  label.squeeze(1).cuda(opt.cuda_idx)   # for h5 -> label.squeeze(1).cuda(opt.cuda_idx)
            
            optimizer.zero_grad()

            prediction,_ = adpnet(data) 
            
            loss = criterion(prediction, label)
            # if opt.trans_loss:
            #     loss += pointnet_regularization(trans_matrix)*opt.alpha
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = prediction.data.max(1)[1]
            # OVerall acc
            correct += preds.eq(label.data).sum().item()

        scheduler.step()

        epoch_loss = round(epoch_loss / num_train_batch, 5)
        epoch_accuracy = round((correct / num_train_examples)*100, 2)

        print(Fore.CYAN+"Loss : ", epoch_loss)
        print(Fore.CYAN+"accuracy : ", epoch_accuracy)

        train_loss.append(epoch_loss)
        train_acc.append(epoch_accuracy)
    
        print(Fore.WHITE)
        progress_bar = tqdm(
            range(num_test_batch),
            desc=f"Testing Epoch {epoch}/{opt.nepochs}"
        )
        adpnet.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_dataloader):
                # data, labels = next(iter(test_dataloader))
                data = data.transpose(1,2).cuda(opt.cuda_idx)
                labels = labels.squeeze(1).cuda(opt.cuda_idx) # for h5 - > labels.squeeze(1).cuda(opt.cuda_idx)
                prediction, embb = adpnet(data)
                preds = prediction.data.max(1)[1]
                # OVerall iacc
                test_correct += preds.eq(labels.data).sum().item()
                # Class acc
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    test_confusion[t.long(), p.long()] += 1

                if opt.dataset == 'shapeobj':
                    embd.append(embb.detach().cpu())
                    color.append(preds.detach().cpu())
        
        test_acc_s, ma = compute_accuracy(test_confusion)
        print("==> Test Overall accuracy : ", test_acc_s)
        print("==> Test Mean accuracy : ", ma)
        test_acc.append(test_acc_s)

        if test_acc_s > max_test:
            test_state = {'epoch': epoch,
                'model_state_dict': adpnet.state_dict(),
                'test_accuracy': test_acc_s,
                'accuracy': epoch_accuracy,
                'loss': epoch_loss,
                'confusion' : test_confusion,
                'embed': embd,
                'labels': color
                }
            with open(log_dirname+'/test/'+'model_pt_'+str(test_acc_s)+'.pth.tar', 'w') as fp:
                fp.close()
            torch.save(test_state, log_dirname+'/test/'+'model_pt_'+str(test_acc_s)+'.pth.tar' )
            max_test = test_acc_s
        
        embd =[]
        color = []

        print(Fore.CYAN)
        print("Max training Accuracy : ", max_acc)
        print("Max Testing Overall Accuracy : ", max_test)

        if epoch_accuracy > max_acc:
            max_acc = epoch_accuracy
            state = {'epoch': epoch,
                    'model_state_dict': adpnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': epoch_accuracy,
                    'loss': epoch_loss,
                    'test_accuracy': max_test
                    }
            save_checkpoint(state, log_dirname, opt.run_name, '.pth.tar')
        

    if (os.path.exists == log_dirname+'/'+'train_acc.npy'):
            a = np.load(log_dirname+'/'+'train_acc.npy')
            a = a.append(train_acc)
            np.save(log_dirname+'/'+'train_acc.npy', a)
            print(Fore.GREEN+">>> Saved train_acc.npy")
    else:
        np.save(log_dirname+'/'+'train_acc.npy', train_acc)
        print(Fore.GREEN+">>> Saved train_acc.npy")

    if (os.path.exists == log_dirname+'/'+'train_loss.npy'):
        a = np.load(log_dirname+'/'+'train_loss.npy')
        a = a.append(train_loss)
        np.save(log_dirname+'/'+'train_loss.npy', a)
        print(Fore.GREEN+">>> Saved train_loss.npy")
    else:
        np.save(log_dirname+'/'+'train_loss.npy', train_loss)
        print(Fore.GREEN+">>> Saved train_loss.npy")

    if (os.path.exists == log_dirname+'/'+'test_acc.npy'):
        a = np.load(log_dirname+'/'+'test_acc.npy')
        a = a.append(test_acc)
        np.save(log_dirname+'/'+'test_acc.npy', a)
        print(Fore.GREEN+">>> Saved test_acc.npy")
    else:
        np.save(log_dirname+'/'+'test_acc.npy', test_acc)
        print(Fore.GREEN+">>> Saved test_acc.npy")


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_adpnet(train_opt)
