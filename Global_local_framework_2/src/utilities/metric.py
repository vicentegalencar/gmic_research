import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def compute_metric(prediction, threshold):
    y_score = prediction[:, 0]
    y_pred = np.where(y_score > threshold, 1, 0)
    y_true = prediction[:, 1]
    print(y_pred, y_pred.sum())
    print(y_true, y_true.sum())

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    TP = (y_pred * y_true).sum()
    FN = ((1-y_pred) * y_true).sum()
    TN = ((1-y_pred) * (1-y_true)).sum()
    FP = (y_pred * (1-y_true)).sum()
    
    # ================================================================================= #
    # NOTE: If auc < 0.5, you need to change the order of 'neg' and 'pos' in cls_list,
    # and recalculate auc, acc, TP, FN, TN, FP
    # OR you can get the correct metrics in the following way
    #           1-auc ==> correct auc
    #           1-acc ==> correct acc
    #           TP ===> correct FP
    #           FN ===> correct TN
    #           TN ===> correct FN
    #           FP ===> correct TP
    #=================================================================================== #
    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc
        tmp = TP
        TP = FP
        FP = tmp
        tmp = FN
        FN = TN
        TN = tmp
        
    return TP, FN, TN, FP, acc, roc_auc