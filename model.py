import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import copy

from utils.graph import Graph_kv2, Graph_kv3, Graph_asian


'''===========================================  ED-GCN  ============================================='''
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def temporal_interp_repeat(x):
    N, F, T = x.size()
    output = torch.zeros([N,F,2*T])
    for i in range(0,T):
        output[:, :, i * 2] = x[:, :, i]
        output[:, :, i * 2 + 1] = x[:, :, i]

    return output


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        #x = self.conv(x)
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class SingleStageModel_TUNet(nn.Module):
    def __init__(self, num_layers, diminput, num_f_maps):
        super(SingleStageModel_TUNet, self).__init__()
        self.conv_1x1 = nn.Conv1d(diminput, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer_TUNet(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return out


class DilatedResidualLayer_TUNet(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer_TUNet, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        return (x + out)


def gcn2tcn(x):
    N, C, T, V = x.size()
    x = x.permute(0, 1, 3, 2)
    x = torch.reshape(x, (N, C * V, T)).contiguous()  # N, C, T, V -> N, C, V, T -> N, C, T

    return x

def tcn2gcn(x, jointnum):
    N, C, T = x.size()
    x = torch.reshape(x, (N, int(C/jointnum), jointnum, T))
    x = x.permute(0, 1, 3, 2).contiguous()  # N, C, T -> N, C, V, T -> N, C, T, V

    return x


class Network_EDGCN(nn.Module):
    def __init__(self, args):
        super(Network_EDGCN, self).__init__()
        # device
        if torch.cuda.is_available() == True:
            self.device = torch.device('cuda:{}'.format(args.gpuid))
        else:
            self.device = torch.device('cpu')

        # AGCN
        if args.dataset == 'TST':
            self.graph = Graph_kv2(labeling_mode='spatial')
        elif args.dataset == 'STS':
            self.graph = Graph_kv3(labeling_mode='spatial')
        elif args.dataset == 'Asian':
            self.graph = Graph_asian(labeling_mode='spatial')
        A = self.graph.A

        self.gcn1 = TCN_GCN_unit(3, 32, A, stride=2)
        self.gcn2 = TCN_GCN_unit(32, 32, A, stride=2)
        self.gcn3 = TCN_GCN_unit(32, 32, A)
        self.gcn4 = TCN_GCN_unit(32, 32, A)

        self.tcn1 = SingleStageModel_TUNet(args.dtcn_layers, 32 * args.jointnum, 32 * args.jointnum)
        self.tcn2 = SingleStageModel_TUNet(args.dtcn_layers, 32 * args.jointnum, 32 * args.jointnum)
        self.tcn3 = SingleStageModel_TUNet(args.dtcn_layers, 32 * args.jointnum, 32 * args.jointnum)
        self.tcn4 = SingleStageModel_TUNet(args.dtcn_layers, 32 * args.jointnum, 3 * args.jointnum)

        self.fc = nn.Linear(3 * args.jointnum, args.class_num)
        self.softmax = nn.Softmax(dim=2)


    def forward(self,x):
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, C, T, V)

        # encoder
        x = self.gcn1(x)
        x = gcn2tcn(x)
        x = self.tcn1(x)
        x = tcn2gcn(x, V)

        x = self.gcn2(x)
        x = gcn2tcn(x)
        x = self.tcn2(x)
        x = tcn2gcn(x, V)

        # decoder
        x = self.gcn3(x)
        x = gcn2tcn(x)
        x = temporal_interp_repeat(x).to(self.device)
        x = self.tcn3(x)
        x = tcn2gcn(x, V)

        x = self.gcn4(x)
        x = gcn2tcn(x)
        x = temporal_interp_repeat(x).to(self.device)
        x = self.tcn4(x)

        x = x.permute(0, 2, 1)   # N, C, T -> N, T, C
        x = self.fc(x)

        outputsoftmax = self.softmax(x)

        return x, outputsoftmax


'''===================================================================================================='''
'''===========================================  Baseline  ============================================='''
'''===================================================================================================='''


'''===========================================  Bi-LSTM  ============================================='''
class Network_bilstm(nn.Module):
    def __init__(self, args):
        super(Network_bilstm, self).__init__()

        self.encoder = nn.LSTM(input_size=args.inputfeq, hidden_size=args.lstmhidden, num_layers=args.lstmlayers,
                               bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(args.lstmhidden*2,48)
        self.linear2 = nn.Linear(48, args.class_num)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        N,T,V,C = x.size()
        x = torch.reshape(x, (N,T,V*C))

        x, (hn, cn) = self.encoder(x)

        x = self.linear2(self.linear1(x))

        outputsoftmax = self.softmax(x)

        return x,outputsoftmax


'''===========================================  ED-TCN  ============================================='''
def temporal_interp_rep(x):
    N, F, T = x.size()
    output = torch.zeros([N,F,2*T])
    for i in range(0,T):
        output[:, :, i * 2] = x[:, :, i]
        output[:, :, i * 2 + 1] = x[:, :, i]

    return output


class Network_EDTCN(nn.Module):
    def __init__(self, args):
        super(Network_EDTCN, self).__init__()
        # device
        if torch.cuda.is_available() == True:
            self.device = torch.device('cuda:{}'.format(args.gpuid))
        else:
            self.device = torch.device('cpu')

        self.conv1 = nn.Conv1d(args.inputfeq, 64, args.edkernel, padding=args.edkernel-3)
        self.conv2 = nn.Conv1d(64, 96, args.edkernel, padding=args.edkernel-3)
        self.conv3 = nn.Conv1d(96, 96, args.edkernel, padding=args.edkernel-3)
        self.conv4 = nn.Conv1d(96, 64, args.edkernel, padding=args.edkernel-3)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(64, args.class_num)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        N, T, V, C = x.size()
        x = torch.reshape(x,(N,T,V*C))
        x = x.permute(0, 2, 1)  # NTVC -> NTF -> NFT

        # encoder
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # decoder
        x = temporal_interp_rep(self.relu(self.conv3(x))).to(self.device)
        x = temporal_interp_rep(self.relu(self.conv4(x))).to(self.device)

        x = x.permute(0, 2, 1) # N, C, T -> N, T, C
        x = self.fc(x)
        outputsoftmax = self.softmax(x)

        return x, outputsoftmax


'''===========================================  MS-TCN  ============================================='''
# MS-TCN
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        return (x + out) * mask[:, 0:1, :]


class Network_MSTCN(nn.Module):
    def __init__(self, args):
        super(Network_MSTCN, self).__init__()
        self.num_stages = args.tcnstages
        self.num_layers = args.tcnlayers
        self.num_f_maps = args.tcnhidden
        self.dim = args.inputfeq
        self.num_classes = args.class_num

        self.stage1 = SingleStageModel(self.num_layers, self.num_f_maps, self.dim, self.num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SingleStageModel(self.num_layers, self.num_f_maps, self.num_classes, self.num_classes)) for s in
             range(self.num_stages - 1)])

    def forward(self, x, mask):
        N, T, V, C = x.size()
        x = torch.reshape(x,(N,T,V*C))
        x = x.permute(0, 2, 1)  # N,T,V,C -> N,F,T

        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs



'''===========================================  MS-GCN  ============================================='''

class SingleStageModel_MSGCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel_MSGCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer_MSGCN(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer_MSGCN(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer_MSGCN, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, mask):
        out = self.conv_dilated(x)
        out = self.bn(out)
        out = self.relu(out)
        out = out + x
        return out * mask[:, 0:1, :]


class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert np.shape(A)[0] == self.kernel_size
        #assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 A=None,
                 dilation=1,
                 residual=True):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad = int((dilation*(kernel_size[0]-1))/2)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=(pad, 0),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        x = self.relu(x)
        x = x + res
        return x, A


class ST_GCN_Model(nn.Module):
    def __init__(self, in_channels=3, num_class=2, dil=[1,2,4,8,16], filters=64,
                 edge_importance_weighting=True, args=0):
        super(ST_GCN_Model, self).__init__()
        graph_args = {'layout': 'tp-vicon', 'strategy': 'spatial'}
        # load graph
        # print('--------')
        # print(graph_args)
        if args.dataset == 'TST':
            self.graph = Graph_kv2(labeling_mode='spatial')
        elif args.dataset == 'STS':
            self.graph = Graph_kv3(labeling_mode='spatial')
        elif args.dataset == 'Asian':
            self.graph = Graph_asian(labeling_mode='spatial')
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.conv_1x1 = nn.Conv2d(in_channels, filters, 1)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[0], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[1], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[2], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[3], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[4], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[5], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[6], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[7], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[8], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[9], residual=True),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.conv_out = nn.Conv1d(filters, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, C, T, V)

        # forward
        x = self.conv_1x1(x)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1, V))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        v = x.size(3)
        x = torch.reshape(x.permute(0,1,3,2), (N,c*v,t))
        out = self.conv_out(x)
        return out


class Network_MSGCN(nn.Module):
    def __init__(self, args):
        super(Network_MSGCN, self).__init__()
        self.dil = args.msgcn_dil
        self.num_layers_R = args.msgcn_layers
        self.num_R = args.tcnstages
        self.num_f_maps = args.msgcn_hidden
        self.dim = 3
        self.num_classes = args.class_num

        self.stream = ST_GCN_Model(in_channels=self.dim, num_class=self.num_classes, filters=self.num_f_maps, dil=self.dil, args=args)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel_MSGCN(self.num_layers_R, self.num_f_maps, self.num_classes, self.num_classes)) for s in range(self.num_R-1)])

    def forward(self, x, mask):
        out = self.stream(x) * mask[:, 0:1, :]
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs























