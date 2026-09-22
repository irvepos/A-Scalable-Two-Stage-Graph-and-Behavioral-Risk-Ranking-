import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math

class ContinuousTimeEncoder(nn.Module):
    """
    پیاده‌سازی فرمول (7): رمزگذاری پیوسته زمان بر اساس قضیه بوخنر
    """
    def __init__(self, time_dim):
        super(ContinuousTimeEncoder, self).__init__()
        self.time_dim = time_dim
        # پارامترهای فرکانس قابل یادگیری (omega)
        self.omega = nn.Parameter(torch.Tensor(time_dim // 2))
        nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def forward(self, delta_t):
        t_omega = delta_t * self.omega.view(1, -1)
        time_encoded = torch.cat([torch.cos(t_omega), torch.sin(t_omega)], dim=-1)
        return time_encoded * math.sqrt(1.0 / self.time_dim)

class TemporalAttentionLayer(MessagePassing):
    """
    پیاده‌سازی فرمول‌های (6)، (8) و (9): پیام‌رسانی و توجه زمانی
    """
    def __init__(self, node_dim, edge_dim, time_dim, out_dim, heads=4):
        super(TemporalAttentionLayer, self).__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_dim = out_dim
        
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.attn_q = nn.Linear(node_dim, out_dim * heads)
        self.attn_k = nn.Linear(node_dim + time_dim, out_dim * heads)
        self.attn_v = nn.Linear(out_dim, out_dim * heads)

    def forward(self, x, edge_index, edge_attr, delta_t, time_encoder):
        time_enc = time_encoder(delta_t)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, time_enc=time_enc)

    def message(self, x_i, x_j, edge_attr, time_enc, index, ptr, size_i):
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        
        q = self.attn_q(x_i).view(-1, self.heads, self.out_dim)
        k_input = torch.cat([x_j, time_enc], dim=-1)
        k = self.attn_k(k_input).view(-1, self.heads, self.out_dim)
        v = self.attn_v(msg).view(-1, self.heads, self.out_dim)
        
        alpha = (q * k).sum(dim=-1) / math.sqrt(self.out_dim)
        alpha = torch.nn.functional.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        
        return (v * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_dim)

class Stage2RiskRanker(nn.Module):
    """
    متصل کردن لایه‌ها، آپدیت GRU (فرمول 10) و امتیاز نهایی ترکیبی (فرمول 11)
    """
    def __init__(self, node_dim, edge_dim, time_dim, hidden_dim, alpha_fusion=0.5):
        super(Stage2RiskRanker, self).__init__()
        self.time_encoder = ContinuousTimeEncoder(time_dim)
        self.temporal_attn = TemporalAttentionLayer(node_dim, edge_dim, time_dim, hidden_dim)
        
        self.gru = nn.GRUCell(hidden_dim * self.temporal_attn.heads, node_dim)
        
        self.risk_classifier = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.alpha = alpha_fusion

    def forward(self, x, edge_index, edge_attr, delta_t, stage1_beh_score):
        h_N = self.temporal_attn(x, edge_index, edge_attr, delta_t, self.time_encoder)
        x_updated = self.gru(h_N, x)
        graph_risk = self.risk_classifier(x_updated).squeeze(-1)
        
        final_risk = (self.alpha * stage1_beh_score) + ((1 - self.alpha) * graph_risk)
        return final_risk, x_updated