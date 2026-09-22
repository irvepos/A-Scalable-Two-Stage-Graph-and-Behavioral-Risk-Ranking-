import pandas as pd
import torch
from torch_geometric.data import Data

def create_graph_data(df, target_col='Is laundering'):
    edge_index = torch.tensor([df['source_account'].values, 
                               df['destination_account'].values], dtype=torch.long)
    
    edge_features = ['amount', 'Velocity', 'Flow_Skewness', 'Amount_Entropy']
    edge_attr = torch.tensor(df[edge_features].values, dtype=torch.float)
    
    delta_t = torch.tensor(df['timestamp'].diff().fillna(0).values, dtype=torch.float).view(-1, 1)
    y = torch.tensor(df[target_col].values, dtype=torch.float)
    
    num_nodes = max(df['source_account'].max(), df['destination_account'].max()) + 1
    x = torch.randn((num_nodes, 16))
    
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, delta_t=delta_t)
    return graph_data