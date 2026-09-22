import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, confusion_matrix

def evaluate_model(preds, targets, inference_times=None):
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    preds_class = (preds_np >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(targets_np, preds_np)
    pr_auc = average_precision_score(targets_np, preds_np)
    
    f1 = f1_score(targets_np, preds_class)
    mcc = matthews_corrcoef(targets_np, preds_class)
    
    tn, fp, fn, tp = confusion_matrix(targets_np, preds_class).ravel()
    fpr = fp / (fp + tn + 1e-7) * 100
    
    print("\n" + "="*40)
    print("نتایج ارزیابی مدل (Evaluation Metrics):")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"FPR:       {fpr:.2f}%")
    
    if inference_times:
        avg_latency = np.mean(inference_times) * 1000
        print(f"Average Inference Latency: {avg_latency:.2f} ms")
    print("="*40 + "\n")
    
    return roc_auc, pr_auc, f1, mcc, fpr