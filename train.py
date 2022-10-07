import argparse
import sys
import random
import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from data.Feeders import TST_Feeder_5F, STS_Feeder_5F, Asian_Feeder_10F
from utils.tools import Logger, seed_everything, adjust_learning_rate
from model import Network_bilstm, Network_MSTCN, Network_EDTCN, Network_MSGCN, Network_EDGCN
from utils.penalty import dist_penalty_tst, dist_penalty_sts,dist_penalty_asian,\
    adjcent_penalty_tst, adjcent_penalty_sts, adjcent_penalty_asian, phaseweight_penalty_tst, phaseweight_penalty_sts

'''
Hyper-parameters settings:
batch size = 1 for all experiments

TST_TUG Dataset:
    Bi-lstm: Epoch-150;; lr-1e-4; wd-0
    SS-TCN: Epoch-50; lr-5e-4; wd-0;
    ED-TCN: Epoch-100; lr-1e-5; wd-0;
    MS-TCN: Epoch-50; lr-5e-4; wd-0;
    MS-GCN: Epoch-50; lr-1e-4; wd-0;
    ED-GCN: Epoch-50; lr-5e-4; wd-0;

Sit-to-stand(STS) Dataset:
    Bi-lstm: Epoch-150; lr-1e-4; wd-0
    SS-TCN: Epoch-50; lr-5e-4; wd-0;
    ED-TCN: Epoch-100; lr-1e-5; wd-0;
    MS-TCN: Epoch-50; lr-5e-4; wd-0;
    MS-GCN: Epoch-50; lr-1e-4; wd-0;
    ED-GCN: Epoch-50; lr-5e-4; wd-0;

Asian_TUG Dataset:
    ED-TCN: Epoch-100; lr-1e-5; wd-0;
    ED-GCN: Epoch-50; lr-5e-4; wd-0;
    
Other hyper-parameters follow the default ones in the parser
'''

parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='TST', choices=['STS','TST','Asian'])
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--STSpath', type=str, default=r'...\TUG-EDGCN\data\STS\Five_fold')
parser.add_argument('--TSTpath', type=str, default=r'...\TUG-EDGCN\data\TST_TUG\Five_fold')
parser.add_argument('--Asianpath', type=str, default=r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold')
parser.add_argument('--labels', type=str, default='fine', choices=['coarse','fine'],
                    help='whether segmenting the STS into finer sub-phases (only applicable to STS dataset)')
parser.add_argument('--class_num', type=int, default=6)
parser.add_argument('--jointnum', type=int, default=25)

# training
parser.add_argument('--lr',type=float,default=5e-4,help='learning rate')
parser.add_argument('--wd',type=float,default=0,help='weight decay')
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--printfreq', type=int, default=5)
parser.add_argument('--schedule',type=int,default=[], help='epochs to decrease learning rate')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--fold', type=int, default=5, help='Number of folds, do not change this hyparam')
parser.add_argument('--seed', type=str, default='random', choices=['random','fixed'], help='seeding option')
parser.add_argument('--seedid', type=int, default=0, help='seed number for fixed seeding')

# model
parser.add_argument('--network',type=str,default='edgcn', choices=['bilstm','sstcn','unet6','edtcn','mstcn','msgcn'])
parser.add_argument('--inputfeq', type=int, default=75, help='number of features inputted to the network')
    #Bi-LSTM
parser.add_argument('--lstmhidden', type=int, default=256, help='number of features in the hidden state in lstm. In unet decoder, this number is changed')
parser.add_argument('--lstmlayers', type=int, default=2, help='number of layers in the lstm encoder')
    #ED-TCN
parser.add_argument('--edkernel', type=int, default=5, help='Kernel size for 1D convolution in ED-TCN network')
    #MS-TCN
parser.add_argument('--tcnstages', type=int, default=4, help='number of stages in the MS-TCN network')
parser.add_argument('--tcnlayers', type=int, default=10, help='number of layers in each TCN stage')
parser.add_argument('--tcnhidden', type=int, default=64, help='number of features in each TCN layer')
    #MS-GCN
parser.add_argument('--msgcn_dil', type=int, default=[1,2,4,8,16,32,64,128,256,512], help='dilations in the MS-TCN network')
parser.add_argument('--msgcn_layers', type=int, default=10, help='dilations in the MS-TCN network')
parser.add_argument('--msgcn_hidden', type=int, default=64, help='number of features in each TCN layer')
    #ED-GCN
parser.add_argument('--dtcn_layers', type=int, default=5, help='number of layers in dilated TCN unit')

# saving and resume
parser.add_argument('--resume', type=str, default=None,
                    metavar='PATH', help='path to designated checkpoint (default: None)')
parser.add_argument('--checkpoint_path', default=r'...\TUG-EDGCN\checkpoints', type=str)

# device
parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use.')
parser.add_argument('--workers', type=int, default=0, help='number of workers')



def train(args,train_loader,model,optimizer,criterion,device,epoch,fold):
    adjust_learning_rate(optimizer, epoch, args)
    lossmeter = []
    accmeter = []
    model.train().to(device)

    for i, (skeleton, labels, frames_count) in enumerate(train_loader):

        # data preprocessing
        skeleton = skeleton.to(device)
        fc_record = frames_count.clone().long()
        frames_count = torch.max(frames_count)
        frames_count = frames_count - np.mod(frames_count, 8)
        frames_count = frames_count.to(device).long()
        fc_record[torch.argmax(fc_record)] = frames_count

        if args.dataset == 'STS':
            if args.labels == 'fine':
                labels = labels[:, 0, 0:frames_count].to(device).long()
            elif args.labels == 'coarse':
                labels = labels[:, 1, 0:frames_count].to(device).long()
        else:
            labels = labels.to(device).long()
            labels = labels[:, 0:frames_count]

        skeleton = skeleton[:, 0:frames_count, :, :]


        N, T, V, C = skeleton.size()
        tcn_mask = torch.zeros([N, args.class_num, T])
        for ex in range(0, N):
            tcn_mask[ex, :, 0:fc_record[ex]] = torch.ones([args.class_num, fc_record[ex]])
        tcn_mask = tcn_mask.to(device)


        # model
        if args.network == 'mstcn' or args.network == 'sstcn' or args.network == 'msgcn':
            output = model(skeleton, tcn_mask)
            _, pred = torch.max(output[-1].data, 1)

        else:
            output, outputsoftmax = model(skeleton)
            pred = torch.argmax(outputsoftmax, dim=2).long()

        # accuracy
        correct = pred.eq_(labels).view(-1)
        accuracy = float(torch.sum(correct) / len(correct))

        accmeter.append(accuracy)
        acc_mean = np.mean(accmeter)

        # penalization terms: DPP and more
        if args.dataset == 'TST':
            frame_dist_penalty = dist_penalty_tst(pred, labels).to(device)
            #front_adj_penalty = adjcent_penalty_tst(pred,labels,'pre').to(device)
            #back_adj_penalty = adjcent_penalty_tst(pred,labels,'sub').to(device)
            #phase_penalty = phaseweight_penalty_tst(pred,args).to(device)
        elif args.dataset == 'STS':
            frame_dist_penalty = dist_penalty_sts(pred, labels, args).to(device)
            #front_adj_penalty = adjcent_penalty_sts(pred,labels,'pre',args).to(device)
            #back_adj_penalty = adjcent_penalty_sts(pred,labels,'sub',args).to(device)
            #phase_penalty = phaseweight_penalty_sts(pred).to(device)
        elif args.dataset == 'Asian':
            frame_dist_penalty = dist_penalty_asian(pred, labels).to(device)
            #front_adj_penalty = adjcent_penalty_asian(pred,labels,'pre').to(device)
            #back_adj_penalty = adjcent_penalty_asian(pred,labels,'sub').to(device)

        # loss
        if args.network == 'mstcn' or args.network == 'sstcn' or args.network == 'msgcn':
            loss = 0
            MseLoss = torch.nn.MSELoss(reduction='none')
            for p in output:
                loss += criterion(p.transpose(2, 1).contiguous().view(-1, args.class_num), labels.view(-1))
                loss += 0.15*torch.mean(torch.clamp(MseLoss(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*tcn_mask[:, :, 1:])

        elif args.dataset == 'Asian':
            MseLoss = torch.nn.MSELoss(reduction='none')
            output = output.permute(0,2,1) #N,T,C -> N,C,T
            celoss = criterion(output.transpose(2, 1).contiguous().view(-1, args.class_num), labels.view(-1))
            mseloss = 0.15*torch.mean(torch.clamp(MseLoss(F.log_softmax(output[:, :, 1:], dim=1), F.log_softmax(output.detach()[:, :, :-1], dim=1)), min=0, max=16))
            loss = celoss + mseloss

        else:
            output = output.permute(0, 2, 1)  # N,T,C -> N,C,T
            CEloss = criterion(output, labels)
            framewise_loss = torch.mul(CEloss, frame_dist_penalty)
            loss = torch.mean(framewise_loss)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # loss recorder
        lossmeter.append(loss.item())
        loss_mean = np.mean(lossmeter)

        if (i+1) % args.printfreq == 0:
            print('Fold[%d/%d]-Epoch[%d]:iteration(%d/%d)  Loss:%.4f(%.4f)  Acc:%.4f(%.4f)' %
                  (fold+1,args.fold, epoch,i+1,len(train_loader),loss,loss_mean,accuracy,acc_mean))
    return loss_mean, acc_mean



def validate(args,val_loader,model,criterion,device,epoch,best_acc,fold):
    accmeter = []
    lossmeter = []
    model.eval().to(device)

    for i, (skeleton, labels, frames_count) in enumerate(val_loader):

        # data preprocessing
        skeleton = skeleton.to(device)
        fc_record = frames_count.clone().long()
        frames_count = torch.max(frames_count)
        frames_count = frames_count - np.mod(frames_count, 8)
        frames_count = frames_count.to(device).long()
        fc_record[torch.argmax(fc_record)] = frames_count

        if args.dataset == 'STS':
            if args.labels == 'fine':
                labels = labels[:, 0, 0:frames_count].to(device).long()
            elif args.labels == 'coarse':
                labels = labels[:, 1, 0:frames_count].to(device).long()
        else:
            labels = labels.to(device).long()
            labels = labels[:, 0:frames_count]

        skeleton = skeleton[:, 0:frames_count, :, :]

        N, T, V, C = skeleton.size()
        tcn_mask = torch.zeros([N, args.class_num, T])
        for ex in range(0, N):
            tcn_mask[ex, :, 0:fc_record[ex]] = torch.ones([args.class_num, fc_record[ex]])
        tcn_mask = tcn_mask.to(device)

        # model
        if args.network == 'mstcn' or args.network == 'sstcn' or args.network == 'msgcn':
            output = model(skeleton, tcn_mask)
            _, pred = torch.max(output[-1].data, 1)

        else:
            output, outputsoftmax = model(skeleton)
            pred = torch.argmax(outputsoftmax, dim=2).long()

        # accuracy
        correct = pred.eq_(labels).view(-1)
        accuracy = float(torch.sum(correct) / len(correct))

        accmeter.append(accuracy)
        acc_mean = np.mean(accmeter)

        # penalization terms: DPP and more
        if args.dataset == 'TST':
            frame_dist_penalty = dist_penalty_tst(pred, labels).to(device)
            # front_adj_penalty = adjcent_penalty_tst(pred,labels,'pre').to(device)
            # back_adj_penalty = adjcent_penalty_tst(pred,labels,'sub').to(device)
            # phase_penalty = phaseweight_penalty_tst(pred,args).to(device)
        elif args.dataset == 'STS':
            frame_dist_penalty = dist_penalty_sts(pred, labels, args).to(device)
            # front_adj_penalty = adjcent_penalty_sts(pred,labels,'pre',args).to(device)
            # back_adj_penalty = adjcent_penalty_sts(pred,labels,'sub',args).to(device)
            # phase_penalty = phaseweight_penalty_sts(pred).to(device)
        elif args.dataset == 'Asian':
            frame_dist_penalty = dist_penalty_asian(pred, labels).to(device)
            # front_adj_penalty = adjcent_penalty_asian(pred,labels,'pre').to(device)
            # back_adj_penalty = adjcent_penalty_asian(pred,labels,'sub').to(device)

        # loss
        if args.network == 'mstcn' or args.network == 'sstcn' or args.network == 'msgcn':
            loss = 0
            MseLoss = torch.nn.MSELoss(reduction='none')
            for p in output:
                loss += criterion(p.transpose(2, 1).contiguous().view(-1, args.class_num), labels.view(-1))
                loss += 0.15*torch.mean(torch.clamp(MseLoss(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*tcn_mask[:, :, 1:])

        elif args.dataset == 'Asian':
            MseLoss = torch.nn.MSELoss(reduction='none')
            output = output.permute(0,2,1) #N,T,C -> N,C,T
            celoss = criterion(output.transpose(2, 1).contiguous().view(-1, args.class_num), labels.view(-1))
            mseloss = 0.15*torch.mean(torch.clamp(MseLoss(F.log_softmax(output[:, :, 1:], dim=1), F.log_softmax(output.detach()[:, :, :-1], dim=1)), min=0, max=16))
            loss = celoss + mseloss

        else:
            output = output.permute(0, 2, 1)  # N,T,C -> N,C,T
            CEloss = criterion(output, labels)
            framewise_loss = torch.mul(CEloss, frame_dist_penalty)
            loss = torch.mean(framewise_loss)


        # loss recorder
        lossmeter.append(loss.item())
        loss_mean = np.mean(lossmeter)

        if (i+1) % args.printfreq == 0:
            print('Epoch[%d]:iteration(%d/%d) Acc:%.4f(%.4f)' %
                  (epoch,i+1,len(val_loader),accuracy,acc_mean))


    print('Average Accuracy: %.4f' % (acc_mean))
    if acc_mean > best_acc and acc_mean > 0.8:
        print('New Best Accuracy (%.4f > %.4f)! Saving New Best Model ...' % (acc_mean,best_acc))
        torch.save(model.state_dict(),r'%s\best_model_%s_%s_fold%d_@epoch_%d.pth' % (args.checkpoint_path,
                                                                                     args.dataset, args.network, fold, epoch))
        return acc_mean, acc_mean, loss_mean
    else:
        return best_acc, acc_mean, loss_mean



def main():
    # Parse the hyperparameters
    args = parser.parse_args()

    # Config hyperparameters for different datasets
    if args.dataset == 'TST':
        args.datapath = args.TSTpath
        args.class_num = 6
        args.jointnum = 25
        args.inputfeq = args.jointnum * 3
        args.fold = 5

    elif args.dataset == 'STS':
        args.datapath = args.STSpath
        if args.labels == 'fine':
            args.class_num = 8
        elif args.labels == 'coarse':
            args.class_num = 4
        args.jointnum = 32
        args.inputfeq = args.jointnum * 3
        args.fold = 5

    elif args.dataset == 'Asian':
        args.datapath = args.Asianpath
        args.class_num = 6
        args.jointnum = 19
        args.inputfeq = args.jointnum * 3
        args.fold = 10

    # Seeding
    if args.seed == 'random':
        seed_everything(random.randint(1, 10000))
        print('Random seeding\n')
    elif args.seed == 'fixed':
        seed_everything(args.seedid)
        print('Fixed seed number\n')

    # Device checking
    if torch.cuda.is_available() == True:
        device = torch.device('cuda:{}'.format(args.gpuid))
        print('Use GPU for training')
    else:
        device = torch.device('cpu')
        print('Use CPU for training')

    print('Experiment--dataset:{0}; network:{1}; numofclass:{2}; lr:{3}; wd:{4}'.format(args.dataset, args.network, args.class_num, args.lr, args.wd))

    train_loss_record_5F = []
    val_loss_record_5F = []
    train_acc_record_5F = []
    val_acc_record_5F = []
    best_acc_5F = []

    for fold in range(0,args.fold):
        print('=============================== Fold:[{0}] ==============================='.format(fold+1))
        # Initialize network
        if args.network == 'bilstm':
            model = Network_bilstm(args)
        elif args.network == 'sstcn':
            args.tcnstages = 1
            model = Network_MSTCN(args)
        elif args.network == 'mstcn':
            model = Network_MSTCN(args)
        elif args.network == 'msgcn':
            model = Network_MSGCN(args)
        elif args.network == 'edtcn':
            model = Network_EDTCN(args)
        elif args.network == 'edgcn':
            model = Network_EDGCN(args)

        model = model.to(device)

        # loading pretrained network
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading pretrained model: '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint, strict=False)
                # optimizer.load_state_dict(checkpoint['optimizer'])
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        # Initialize optimizer and optimization criterion
        if args.network == 'mstcn' or args.network == 'sstcn' or args.network == 'msgcn' or args.dataset == 'Asian':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100).to(device)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
            criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

        # Loading data
        if args.dataset == 'TST':
            train_dataset = TST_Feeder_5F(args.datapath, datatype='train', fold=fold)
            val_dataset = TST_Feeder_5F(args.datapath, datatype='val', fold=fold)
        elif args.dataset == 'STS':
            train_dataset = STS_Feeder_5F(args.datapath, datatype='train', fold=fold)
            val_dataset = STS_Feeder_5F(args.datapath, datatype='val', fold=fold)
        elif args.dataset == 'Asian':
            train_dataset = Asian_Feeder_10F(args.datapath, datatype='train', fold=fold)
            val_dataset = Asian_Feeder_10F(args.datapath, datatype='val', fold=fold)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)

        # training and validating
        best_acc = 0
        train_loss_record = []
        val_loss_record = []
        train_acc_record = []
        val_acc_record = []
        for epoch in range(0,args.epochs):
            print('===================== Epoch:[{0}] ====================='.format(epoch))
            print('Training ......')
            train_loss, train_acc = train(args, train_loader, model, optimizer, criterion, device, epoch, fold)
            train_loss_record.append(train_loss)
            train_acc_record.append(train_acc)

            print('Validating ......')
            best_acc, epoch_acc, val_loss = validate(args, val_loader, model, criterion, device, epoch, best_acc, fold)
            val_loss_record.append(val_loss)
            val_acc_record.append(epoch_acc)

        best_acc_5F.append(best_acc)
        train_loss_record_5F.append(train_loss_record)
        train_acc_record_5F.append(train_acc_record)
        val_loss_record_5F.append(val_loss_record)
        val_acc_record_5F.append(val_acc_record)
        print('Final Best Accuracy: %.4f\n' % (best_acc))
        print('===================================== Fold END ========================================')

    print('Five-fold validation accuracy: ', best_acc_5F)
    print('Five-fold average validation accuracy: ', np.mean(best_acc_5F))
    print('=========================================  END  ============================================')


'''
    # plot training curve
    for i in range(0,5):
        # plot the training curve
        fig, ax1 = plt.subplots()
        ax1.plot(train_loss_record_5F[i],'b', label='train loss')
        ax1.plot(val_loss_record_5F[i],'r', label='validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        plt.legend()

        ax2 = ax1.twinx()
        ax2.plot(val_acc_record_5F[i], 'g', label='validation accuracy')
        ax2.plot(train_acc_record_5F[i], 'm', label='training accuracy')
        ax2.set_ylabel('Accuracy')
        plt.legend()

        plt.title('Fold {0}'.format(i + 1))
        plt.show()
'''

if __name__ == '__main__':
    T = time.localtime()
    Time = '%d-%d-%d-%d-%d-%d' % (T[0],T[1],T[2],T[3],T[4],T[5])
    sys.stdout = Logger(r'logbooks/logbook_{}.txt'.format(Time), sys.stdout)
    main()

