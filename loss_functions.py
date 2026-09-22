import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassWeightedFocalLoss(nn.Module):
    """
    پیاده‌سازی فرمول (12) مقاله: Class-Weighted Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(ClassWeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

class ContinuousMCCLoss(nn.Module):
    """
    پیاده‌سازی فرمول (13 تا 15) مقاله: Continuous Differentiable MCC Loss
    """
    def __init__(self, eps=1e-7):
        super(ContinuousMCCLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        # تبدیل خروجی خام مدل به احتمالات پیوسته بین 0 و 1
        probs = torch.sigmoid(inputs)
        
        # محاسبات نرم (Soft) برای ماتریس درهم‌ریختگی (Continuous Approximation)
        TP = torch.sum(probs * targets)
        FP = torch.sum(probs * (1 - targets))
        TN = torch.sum((1 - probs) * (1 - targets))
        FN = torch.sum((1 - probs) * targets)

        # فرمول (15) - مخرج کسر
        numerator = (TP * TN) - (FP * FN)
        denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + self.eps

        mcc = numerator / denominator
        
        # چون هدف بهینه‌ساز کم کردن (Minimize) است، MCC را از 1 کم می‌کنیم
        return 1.0 - mcc

class CompoundRiskLoss(nn.Module):
    """
    پیاده‌سازی فرمول (16) مقاله: Total Objective Function
    ترکیب Focal Loss و MCC Loss برای مقابله با عدم تعادل شدید کلاس‌ها
    """
    def __init__(self, lambda_mcc=1.5):
        super(CompoundRiskLoss, self).__init__()
        self.focal_loss = ClassWeightedFocalLoss(gamma=2.0)
        self.mcc_loss = ContinuousMCCLoss()
        self.lambda_mcc = lambda_mcc

    def forward(self, preds, targets):
        L_focal = self.focal_loss(preds, targets)
        L_mcc = self.mcc_loss(preds, targets)
        
        # فرمول نهایی زیان ترکیبی
        total_loss = L_focal + (self.lambda_mcc * L_mcc)
        return total_loss