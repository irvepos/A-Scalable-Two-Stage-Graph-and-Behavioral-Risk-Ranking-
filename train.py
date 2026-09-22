import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit._script")

import torch
import torch.optim as optim
import time
import pandas as pd
import numpy as np

from data_preprocessing import create_graph_data
from stage1_screening import BehavioralScreeningFilter
from stage2_stgnn import Stage2RiskRanker
from loss_functions import CompoundRiskLoss
from metrics_and_evaluation import evaluate_model

def run_pipeline():
    print("شروع پایپ‌لاین دو مرحله‌ای ضد پولشویی (AML)...")
    
    num_samples = 1000
    df = pd.DataFrame({
        'source_account': np.random.randint(0, 100, num_samples),
        'destination_account': np.random.randint(0, 100, num_samples),
        'amount': np.random.uniform(10, 5000, num_samples),
        'timestamp': np.sort(np.random.uniform(1600000000, 1600086400, num_samples)),
        'Is laundering': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    })
    
    stage1 = BehavioralScreeningFilter(tau_screen=0.35)
    df_features = stage1.calculate_behavioral_features(df)
    
    features_col = ['amount', 'Velocity', 'Flow_Skewness', 'Amount_Entropy']
    stage1.train_filter(df_features, features_col, 'Is laundering')
    
    df_features['S_beh'] = stage1.beh_model.predict_proba(df_features[features_col])[:, 1]
    s_beh_tensor = torch.tensor(df_features['S_beh'].values, dtype=torch.float)
    
    graph_data = create_graph_data(df_features, target_col='Is laundering')
    
    model = Stage2RiskRanker(node_dim=16, edge_dim=4, time_dim=32, hidden_dim=128, alpha_fusion=0.5)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = CompoundRiskLoss(lambda_mcc=1.5)
    
    epochs = 10
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        final_risk, _ = model(graph_data.x, 
                              graph_data.edge_index, 
                              graph_data.edge_attr, 
                              graph_data.delta_t, 
                              s_beh_tensor)
        
        loss = criterion(final_risk, graph_data.y)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Total Loss: {loss.item():.4f}")
        
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        for _ in range(50):
            start_time = time.perf_counter()
            preds, _ = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.delta_t, s_beh_tensor)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
            
    evaluate_model(preds, graph_data.y, inference_times)

if __name__ == "__main__":
    run_pipeline()