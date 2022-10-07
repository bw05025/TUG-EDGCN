import torch

phaseweight_coarse = torch.tensor([3.9892, 3.1029, 8.8178, 3.1884])
phaseweight_fine = torch.tensor([3.9892, 9.2023, 24.7431, 5.7739, 8.8178, 6.6271, 18.8705, 9.1117])

def dist_penalty_sts(pred,labels,args):
    N, T = pred.size()
    diff = torch.abs(pred - labels)

    penalty = 10 * torch.ones([N, T])
    if args.labels == 'coarse':
        penalty[diff != 2] = 1
    elif args.labels == 'fine':
        penalty[diff < 2] = 1
        penalty[diff > 6] = 1

    return penalty

def adjcent_penalty_sts(pred,labels,stat,args):
    N, framenum = pred.size()
    penalty_mask = torch.ones([N,framenum])
    penalty_mask[pred!=labels]=0

    penalty = 2.5 * torch.ones([N,framenum-1])
    tail = torch.zeros([N,1])

    if args.labels == 'coarse':
        end2sit = 3
    elif args.labels == 'fine':
        end2sit = 7

    assert stat=='pre' or stat=='sub'
    if stat == 'pre':
        penalty_mask = penalty_mask[:,1:]
        diff = pred[:,1:]-pred[:,0:framenum-1]
        penalty[diff == 0] = 1
        penalty[diff == 1] = 1
        penalty[diff == -end2sit] = 1

        penalty = torch.mul(penalty,penalty_mask)
        penalty = torch.cat((penalty,tail),1)

    elif stat == 'sub':
        penalty_mask = penalty_mask[:,0:framenum-1]
        diff = pred[:,0:framenum-1]-pred[:,1:]
        penalty[diff == 0] = 1
        penalty[diff == -1] = 1
        penalty[diff == end2sit] = 1

        penalty = torch.mul(penalty, penalty_mask)
        penalty = torch.cat((tail,penalty), 1)

    penalty[penalty == 0] += 1
    return penalty

def phaseweight_penalty_sts(pred,args):
    N, framenum = pred.size(dim=0)
    weights = torch.ones([N, framenum])

    if args.labels == 'coarse':
        weights[pred == 0] = phaseweight_coarse[0]
        weights[pred == 1] = phaseweight_coarse[1]
        weights[pred == 2] = phaseweight_coarse[2]
        weights[pred == 3] = phaseweight_coarse[3]
    elif args.labels == 'fine':
        weights[pred == 0] = phaseweight_fine[0]
        weights[pred == 1] = phaseweight_fine[1]
        weights[pred == 2] = phaseweight_fine[2]
        weights[pred == 3] = phaseweight_fine[3]
        weights[pred == 4] = phaseweight_fine[4]
        weights[pred == 5] = phaseweight_fine[5]
        weights[pred == 6] = phaseweight_fine[6]
        weights[pred == 7] = phaseweight_fine[7]

    return weights


phaseweight_tst = torch.tensor([6.5194, 8.6211, 5.3934, 7.0811, 5.7503, 4.3463])

def dist_penalty_tst(pred,labels):
    N,T = pred.size()
    diff = torch.abs(pred - labels)

    penalty = 10 * torch.ones([N,T])
    penalty[diff < 2] = 1
    penalty[diff > 4] = 1

    return penalty

def adjcent_penalty_tst(pred,labels,stat):
    N, framenum = pred.size()
    penalty_mask = torch.ones([N, framenum])
    penalty_mask[pred!=labels]=0

    penalty = 2.5 * torch.ones([N, framenum-1])
    tail = torch.zeros([N, 1])

    assert stat=='pre' or stat=='sub'
    if stat == 'pre':
        penalty_mask = penalty_mask[:, 1:]
        diff = pred[:, 1:]-pred[:, 0:framenum-1]
        penalty[diff == 0] = 1
        penalty[diff == 1] = 1
        penalty[diff == -5] = 1

        penalty = torch.mul(penalty,penalty_mask)
        penalty = torch.cat((penalty,tail),1)

    elif stat == 'sub':
        penalty_mask = penalty_mask[:, 0:framenum-1]
        diff = pred[:, 0:framenum-1]-pred[:, 1:]
        penalty[diff == 0] = 1
        penalty[diff == -1] = 1
        penalty[diff == 5] = 1

        penalty = torch.mul(penalty, penalty_mask)
        penalty = torch.cat((tail,penalty), 1)

    penalty[penalty == 0] += 1
    return penalty

def phaseweight_penalty_tst(pred):
    N, framenum = pred.size()
    weights = torch.ones([N, framenum])
    weights[pred == 0] = phaseweight_tst[0]
    weights[pred == 1] = phaseweight_tst[1]
    weights[pred == 2] = phaseweight_tst[2]
    weights[pred == 3] = phaseweight_tst[3]
    weights[pred == 4] = phaseweight_tst[4]
    weights[pred == 5] = phaseweight_tst[5]

    return weights


def dist_penalty_asian(pred,labels):
    N, T = pred.size()
    diff = torch.abs(pred - labels)

    penalty = 10 * torch.ones([N,T])
    penalty[diff < 2] = 1

    return penalty


def adjcent_penalty_asian(pred,labels,stat):
    N, framenum = pred.size()
    penalty_mask = torch.ones([N, framenum])
    penalty_mask[pred!=labels]=0

    penalty = 2.5 * torch.ones([N, framenum-1])
    tail = torch.zeros([N, 1])

    assert stat=='pre' or stat=='sub'
    if stat == 'pre':
        penalty_mask = penalty_mask[:, 1:]
        diff = pred[:, 1:]-pred[:, 0:framenum-1]
        penalty[diff == 0] = 1
        penalty[diff == 1] = 1

        penalty = torch.mul(penalty,penalty_mask)
        penalty = torch.cat((penalty,tail),1)

    elif stat == 'sub':
        penalty_mask = penalty_mask[:, 0:framenum-1]
        diff = pred[:, 0:framenum-1]-pred[:, 1:]
        penalty[diff == 0] = 1
        penalty[diff == -1] = 1

        penalty = torch.mul(penalty, penalty_mask)
        penalty = torch.cat((tail,penalty), 1)

    penalty[penalty == 0] += 1
    return penalty