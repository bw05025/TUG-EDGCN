import numpy as np


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends  # shared function for all scores


'''================================ Jitter score ================================'''

def jitter_score(pred, label, weight, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(pred, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(label, bg_class)

    p_len = np.subtract(p_end, p_start)
    y_len = np.subtract(y_end, y_start)
    video_len = np.sum(y_len)
    y_segment = len(y_len)

    py_correspond = np.linspace(0,-y_segment,num=y_segment+1, dtype=int)

    for i in range(len(y_len)):
        pd_len = 0
        for j in range(len(p_len)):
            if p_label[j] != y_label[i]:
                continue
            elif p_len[j] > pd_len:
                intersection = np.minimum(p_end[j], y_end[i]) - np.maximum(p_start[j], y_start[i])
                if intersection > 0:
                    py_correspond[i+1] = j
                    pd_len = p_len[j]
                else:
                    continue

    margin = []
    for i in range(len(y_len)):
        margin.append(py_correspond[i + 1] - py_correspond[i])
    order_penalty = sum(np.maximum(-1, (np.minimum(0, margin))))/(y_segment)

    pred_diff = []
    jitter_len = []

    n_cls_1 = y_label[0].item()
    n_cls_2 = y_label[0].item()

    print(len(p_len))
    for i in range(0,len(p_len)):
        print(i)
        if i in py_correspond[1:]:
            seg = int(np.argwhere(py_correspond[1:] == i)[0])
            n_cls_1 = y_label[seg].item()
            if seg == len(y_len)-1:
                n_cls_2 = y_label[seg].item()
            else:
                n_cls_2 = y_label[seg + 1].item()
            print("Corresponded segment: i={0}, n_cls_1={1}, n_cls_2={2}.".format(i, n_cls_1, n_cls_2))
        else:
            if p_label[i].item() == n_cls_1 or p_label[i].item() == n_cls_2:
                pred_diff.append(1/y_segment)
                print("Jitter segment 1: i={0}, n_cls_1={1}, n_cls_2={2}.".format(i, n_cls_1, n_cls_2))
            else:
                pred_diff.append(10/y_segment)
                print("Jitter segment 10: i={0}, n_cls_1={1}, n_cls_2={2}.".format(i, n_cls_1, n_cls_2))
            jitter_len.append(p_len[i])

    jitter_penalty = np.sum(np.multiply(pred_diff, jitter_len))

    score = (1 - weight) * (1 + order_penalty) + weight * (1 - np.tanh(0.05 * jitter_penalty))

    return score


'''================================ Shift score ================================'''

def shift_score(recognized, ground_truth, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    p_len = np.subtract(p_end, p_start)
    y_len = np.subtract(y_end, y_start)

    py_correspond = -1 * np.ones([len(y_len)],dtype=int)

    for i in range(len(y_len)):    # Search correspondence
        pd_len = 0
        for j in range(len(p_len)):
            if p_label[j] != y_label[i]:
                continue
            elif p_len[j] > pd_len:
                intersection = np.minimum(p_end[j], y_end[i]) - np.maximum(p_start[j], y_start[i])
                if intersection > 0:
                    py_correspond[i] = j
                    pd_len = p_len[j]
                else:
                    continue

    IoU = np.zeros(len(y_label)) # Initialize by zeros, if correspondence is not found (no intersection), IoU=default=0

    for j in range(len(py_correspond)):
        if py_correspond[j] >= 0:
            intersection = np.minimum(p_end[py_correspond[j]], y_end[j]) - np.maximum(p_start[py_correspond[j]], y_start[j])
            union = np.maximum(p_end[py_correspond[j]], y_end[j]) - np.minimum(p_start[py_correspond[j]], y_start[j])
            IoU[j] = 1.0 * intersection / union
    score = np.mean(IoU)

    return score


'''================================ F1 score ================================'''

def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1) * 100

    return f1, precision, recall


'''================================ Edit score ================================'''

def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score



