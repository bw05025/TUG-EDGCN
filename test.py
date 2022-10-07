import argparse
import sys
import random
import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.Feeders import TST_Feeder_5F, STS_Feeder_5F, Asian_Feeder_10F
from utils.tools import Logger, seed_everything
from utils.metrics import edit_score, f_score, jitter_score, shift_score
from model import Network_bilstm, Network_MSTCN, Network_MSGCN, Network_EDTCN, Network_EDGCN
from utils.visualization import colormatch_tug, colormatch_sts


parser = argparse.ArgumentParser()

# test configuration (what you need to modify before testing each model)
parser.add_argument('--resume', type=str, default=r'...\TUG-EDGCN\checkpoints\best_model_TST_edgcn_fold1_@epoch_42.pth',
                    metavar='PATH', help='path of saved model')
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--dataset', type=str, default='TST', choices=['STS','TST','Asian'])
parser.add_argument('--network',type=str,default='edgcn', choices=['edgcn','bilstm','sstcn','edtcn','mstcn','msgcn'])

# data
parser.add_argument('--datapath', type=str, default='')
parser.add_argument('--STSpath', type=str, default=r'...\TUG-EDGCN\data\STS\Five_fold')
parser.add_argument('--TSTpath', type=str, default=r'...\TUG-EDGCN\data\TST_TUG\Five_fold')
parser.add_argument('--Asianpath', type=str, default=r'...\TUG-EDGCN\data\Asian_TUG\Ten_fold')
parser.add_argument('--labels', type=str, default='fine', choices=['coarse','fine'],help='whether segmenting the STS into finer sub-phases')
parser.add_argument('--class_num', type=int, default=6)
parser.add_argument('--jointnum', type=int, default=25)

# model
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

# testing
parser.add_argument('--seed', type=str, default='random', choices=['random','fixed'], help='seeding option')
parser.add_argument('--seedid', type=int, default=0, help='seed number for fixed seeding')
parser.add_argument('--jitter_weight', type=float, default=0.5, help='lambda: control weight of segment order and jitter issue in jitter score')
parser.add_argument('--visual',type=str, default='single', choices=['single','all'], help='save all the segmentation plots to local or plot one selected result')

# device
parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use.')
parser.add_argument('--workers', type=int, default=0, help='number of workers')



def test(args, val_loader, model, device):
    accmeter = []
    model.eval().to(device)

    # initialize tensors for recording results
    if args.dataset == 'TST':
        pred_record = torch.zeros([18, 512])
        label_record = torch.zeros([18, 512])
        fc_book = torch.zeros([18])   # fc=frames count for each example

    elif args.dataset == 'STS':
        pred_record = torch.zeros([65, 300])
        label_record = torch.zeros([65, 300])
        fc_book = torch.zeros([65])

    elif args.dataset == 'Asian':
        pred_record = torch.zeros([3, 3500])
        label_record = torch.zeros([3, 3500])
        fc_book = torch.zeros([3])


    for i, (skeleton, labels, frames_count) in enumerate(val_loader):
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


        # Prediction
        if args.network == 'mstcn' or args.network == 'sstcn' or args.network == 'msgcn':
            output = model(skeleton, tcn_mask)
            _, pred = torch.max(output[-1].data, 1)

        else:
            output, outputsoftmax = model(skeleton)
            pred = torch.argmax(outputsoftmax, dim=2).long()


        # Accuracy and recording the results
        correct = pred.clone().eq_(labels).view(-1)
        accuracy = float(torch.sum(correct) / len(correct))

        accmeter.append(accuracy)

        pred_record[i, 0:frames_count] = pred.clone()
        label_record[i, 0:frames_count] = labels.clone()
        fc_book[i] = frames_count

    acc_mean = np.mean(accmeter)
    acc_max = np.max(accmeter)
    acc_min = np.min(accmeter)

    print('Mean Acc: %.4f; Best Acc: %.4f; Worst Acc: %4f.' % (acc_mean, acc_max, acc_min))

    return pred_record, label_record, fc_book, accmeter



def main():
    # Parse the hyperparameters
    args = parser.parse_args()

    # Config hyperparameters for different datasets
    if args.dataset == 'TST':
        args.datapath = args.TSTpath
        args.class_num = 6
        args.jointnum = 25
        args.inputfeq = args.jointnum * 3

    elif args.dataset == 'STS':
        args.datapath = args.STSpath
        if args.labels == 'fine':
            args.class_num = 8
        elif args.labels == 'coarse':
            args.class_num = 4
        args.jointnum = 32
        args.inputfeq = args.jointnum * 3

    elif args.dataset == 'Asian':
        args.datapath = args.Asianpath
        args.class_num = 6
        args.jointnum = 19
        args.inputfeq = args.jointnum * 3


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
        print('Use GPU for prediction')
    else:
        device = torch.device('cpu')
        print('Use CPU for prediction')

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
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Loading data
    if args.dataset == 'TST':
        test_dataset = TST_Feeder_5F(args.datapath, datatype='val', fold=args.fold)
    elif args.dataset == 'STS':
        test_dataset = STS_Feeder_5F(args.datapath, datatype='val', fold=args.fold)
    elif args.dataset == 'Asian':
        test_dataset = Asian_Feeder_10F(args.datapath, datatype='val', fold=args.fold)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    # testing
    print('testing ......')
    pred_record, label_record, fc_record, accmeter = test(args,test_loader,model,device)
    fc_record = fc_record.long()
    best_idx = np.argmax(accmeter) # index of example with the best accuracy
    worst_idx = np.argmin(accmeter) # index of example with the worst accuracy

    # Calculate metrics results
    f1_50 = 0
    f1_80 = 0
    edit = 0
    jitter50 = 0
    jitter100 = 0
    shift = 0

    video_num = pred_record.size(0)
    video_len = pred_record.size(1)

    for i in range(0,video_num):
        video = pred_record[i,0:fc_record[i]]
        gt = label_record[i,0:fc_record[i]]

        f1, pre, rec = f_score(video, gt, 0.5)
        f1_50 += f1

        f1, pre, rec = f_score(video, gt, 0.8)
        f1_80 += f1

        edit += edit_score(video, gt)
        jitter50 += jitter_score(video, gt, 0.5)
        jitter100 += jitter_score(video, gt, 1)
        shift += shift_score(video,gt)

    f1_50 = f1_50/video_num
    f1_80 = f1_80/video_num
    edit = edit/video_num
    jitter50 = jitter50/video_num
    jitter100 = jitter100/video_num
    shift = shift/video_num

    print('F1 Score @ 50: ', f1_50)
    print('F1 Score @ 80: ', f1_80)
    print('Edit Score: ', edit)
    print('Jitter 50% Score: ', jitter50)
    print('Jitter 100% Score: ', jitter100)
    print('Shift Score: ', shift)
    print('Accuracy_record: ', accmeter)


    # Calculate prediction time error for each phase
    pred_time = np.zeros([video_num,args.class_num-1])
    gt_time = np.zeros([video_num,args.class_num-1])

    for i in range(0,video_num):
        for j in range(0, video_len):
            for k in range(1, args.class_num):
                if pred_record[i,j] == k:
                    pred_time[i,k-1] += 1
                    break
    for i in range(0, video_num):
        for j in range(0, video_len):
            for k in range(1, args.class_num):
                if int(label_record[i,j]) == k:
                    gt_time[i,k-1] += 1
                    break

    error_time = np.abs(pred_time-gt_time)
    mean_error_time = np.mean(error_time,axis=0)
    if args.dataset == 'Asian':
        print("Mean error time(seconds): ", (1 / 50) * mean_error_time)
        # Asian-TUG: 50fps
    else:
        print("Mean error time(seconds): ", (1 / 30) * mean_error_time)
        # TST-TUG/STS: 30fps


    # Visualization of segmentation results (upper: ground truth; lower: prediction)
    if args.visual == 'all':
        for plotidx in range(video_num):
            pred_plt = pred_record[plotidx,0:fc_record[plotidx]]
            labels_plt = label_record[plotidx,0:fc_record[plotidx]]

            fig1 = plt.figure()
            ax1 = fig1.add_subplot()
            for i in range(0,fc_record[plotidx]):
                if args.dataset == 'TST' or args.dataset == 'Asian':
                    predcolor = colormatch_tug(pred_plt[i])
                    gtcolor = colormatch_tug(labels_plt[i])
                elif args.dataset == 'STS':
                    predcolor = colormatch_sts(pred_plt[i])
                    gtcolor = colormatch_sts(labels_plt[i])

                ax1.add_patch(patches.Rectangle(((i + 0.5), 0.1), 1, 0.2, color=predcolor))
                ax1.add_patch(patches.Rectangle(((i + 0.5), 0.6), 1, 0.2, color=gtcolor))
            plt.axis([0,fc_record[plotidx]+1,0,1])
            plt.xlabel('Frames')
            plt.savefig(r'...\visual\{0}_video{1}.png'.format(args.dataset, plotidx))

    elif args.visual == 'single':
        plotidx = worst_idx     # which example to plot
        pred_plt = pred_record[plotidx, 0:fc_record[plotidx]]
        labels_plt = label_record[plotidx, 0:fc_record[plotidx]]

        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        for i in range(0, fc_record[plotidx]):
            if args.dataset == 'TST' or args.dataset == 'Asian':
                predcolor = colormatch_tug(pred_plt[i])
                gtcolor = colormatch_tug(labels_plt[i])
            elif args.dataset == 'STS':
                predcolor = colormatch_sts(pred_plt[i])
                gtcolor = colormatch_sts(labels_plt[i])

            ax1.add_patch(patches.Rectangle(((i + 0.5), 0.1), 1, 0.2, color=predcolor))
            ax1.add_patch(patches.Rectangle(((i + 0.5), 0.6), 1, 0.2, color=gtcolor))
        plt.axis([0, fc_record[plotidx] + 1, 0, 1])
        plt.xlabel('Frames')
        plt.show()



if __name__ == '__main__':
    T = time.localtime()
    Time = '%d-%d-%d-%d-%d-%d' % (T[0],T[1],T[2],T[3],T[4],T[5])
    sys.stdout = Logger(r'logbooks/logbook_test_{}.txt'.format(Time), sys.stdout)
    main()