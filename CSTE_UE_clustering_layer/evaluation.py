import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics

from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc2(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def pairwise_counts(true_labels, predicted_labels,userid):
    tn, fn, fp, tp = 0.0, 0.0, 0.0, 0.0

    assert(len(true_labels) == len(predicted_labels))

    for i in range(0, len(true_labels) - 1):
      for j in range(i + 1, len(true_labels)):
          # if userid[i] == userid[j]:
          if true_labels[i] == true_labels[j] and predicted_labels[i] == predicted_labels[j]:
            tp += 1.0
          elif true_labels[i] != true_labels[j] and predicted_labels[i] != predicted_labels[j]:
            tn += 1.0
          elif true_labels[i] == true_labels[j] and predicted_labels[i] != predicted_labels[j]:
            fn += 1.0
          elif true_labels[i] != true_labels[j] and predicted_labels[i] == predicted_labels[j]:
            fp += 1.0
          # else:
          #     break
    #print(tp,fp,fn,tn)
    return tn, fn, fp, tp

from sklearn.metrics import fbeta_score
def cluster_acc(true_labels, predicted_labels, userid):
    tn, fn, fp, tp = pairwise_counts(true_labels, predicted_labels,userid)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    pairwise_fscore = 2 * precision * recall / (precision + recall + 1e-10)
    pairwise_fscore6 = ((1+0.6*0.6) * precision * recall) / ( (0.6*0.6)*precision + recall + 1e-10)
    return (tp+tn)/(tp+tn+fp+fn), pairwise_fscore,pairwise_fscore6  #, (tp + tn)/(tp+tn+fp+fn)

def get_num_clusters(labels):
  """ Compute the number of clusters from the dataset labels """
  num_clusters = np.max(labels)
  if np.min(labels) == 0:
    num_clusters += 1
  return num_clusters

def unsupervised_accuracy(true_labels, predicted_labels):
    """
    Calculate unsupervised clustering accuracy using the Hungarian algorithm
    in scikit-learn

    [1] https://github.com/XifengGuo/IDEC
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    n_labels = predicted_labels.size
    n_clusters = max(
      get_num_clusters(true_labels), get_num_clusters(predicted_labels)) + 1
    weights = np.zeros((n_clusters, n_clusters), dtype=np.int64)

    for i in range(n_labels):
      weights[predicted_labels[i], true_labels[i]] += 1

    indices_p = linear_assignment(weights.max() - weights)
    indices = list(map(lambda *x: x, indices_p[0], indices_p[1]))
    accuracy = float(sum([weights[i, j] for i, j in indices])) / n_labels
    return accuracy

# def eva(y_true, y_pred, userid, epoch=0):
#     acc, f1 = cluster_acc(y_true, y_pred,userid)
#     #acc = unsupervised_accuracy(y_true,y_pred)
#     nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
#     ari = ari_score(y_true, y_pred)
#     print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
#             ', f1 {:.4f}'.format(f1))
#     return  acc, nmi, ari, f1


def eva(y_true, y_pred, userid, epoch=0):
    acc, f1,f6 = cluster_acc(y_true, y_pred,userid)
    #acc = unsupervised_accuracy(y_true,y_pred)
    acc_un = unsupervised_accuracy(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)

    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ':acc_un {:.4f}'.format(acc_un),', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
            ', f1 {:.4f}'.format(f1),', f6 {:.4f}'.format(f6) )
    return  acc, acc_un, nmi, ari, f1

