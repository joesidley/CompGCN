import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def parse_data(file_path, dataset_type='WN18RR'):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            src, rel, dst = line.strip().split('\t')
            if dataset_type == 'WN18RR':
                edges.append((int(src), rel, int(dst)))
            else:
                edges.append((src, rel, dst))
    return edges

def remap_edges(edges):
    unique_nodes = set()
    unique_relations = set()
    for src, rel, dst in edges:
        unique_nodes.update([src, dst])
        unique_relations.add(rel)

    node_mapping = {node: idx for idx, node in enumerate(sorted(unique_nodes))}
    rel_mapping = {rel: idx for idx, rel in enumerate(sorted(unique_relations))}
    remapped_edges = [(node_mapping[src], rel_mapping[rel], node_mapping[dst]) 
                     for src, rel, dst in edges]

    return remapped_edges, len(unique_nodes), len(unique_relations), rel_mapping

def init_dataset(dataset_type='WN18RR'):
    base_path = f'{dataset_type}/'
    train_raw = parse_data(f'/kaggle/input/{base_path}train.txt', dataset_type)
    valid_raw = parse_data(f'/kaggle/input/{base_path}valid.txt', dataset_type)
    test_raw  = parse_data(f'/kaggle/input/{base_path}test.txt',  dataset_type)
    
    train_remapped, num_nodes, num_relations, rel_mapping = remap_edges(train_raw)
    
    node_mapping = {node: idx for idx, node in enumerate(sorted(set(
        node for edge in train_raw for node in [edge[0], edge[2]]
    )))}
    rel_mapping = {rel: idx for idx, rel in enumerate(sorted(set(
        edge[1] for edge in train_raw
    )))}
    
    valid_edges = [(node_mapping[src], rel_mapping[rel], node_mapping[dst]) 
                   for src, rel, dst in valid_raw 
                   if src in node_mapping and dst in node_mapping and rel in rel_mapping]
    
    test_edges = [(node_mapping[src], rel_mapping[rel], node_mapping[dst]) 
                  for src, rel, dst in test_raw 
                  if src in node_mapping and dst in node_mapping and rel in rel_mapping]
    
    return train_remapped, valid_edges, test_edges, len(node_mapping), len(rel_mapping), rel_mapping

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, edge_index, edge_features):
        src_nodes, dst_nodes = edge_index
        src_features = x[src_nodes]
        messages = self.linear(src_features)
        
        out = torch.zeros((x.shape[0], self.linear.out_features), 
                         device=x.device, 
                         dtype=messages.dtype)
        out.index_add_(0, dst_nodes, messages)
        
        neighbor_counts = torch.zeros(x.shape[0], device=x.device, dtype=messages.dtype)
        neighbor_counts.index_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=messages.dtype))
        neighbor_counts = torch.clamp(neighbor_counts, min=1)
        out = out / neighbor_counts.unsqueeze(1)
        
        return torch.relu(out)

class CompositionalOperator:
    @staticmethod
    def subtraction(h, r):
        h = h if h.dim() > 1 else h.unsqueeze(0)
        r = r if r.dim() > 1 else r.unsqueeze(0)
        result = h - r
        return result.squeeze(0) if result.size(0) == 1 else result

    @staticmethod
    def circular_correlation(h, r):
        h = h if h.dim() > 1 else h.unsqueeze(0)
        r = r if r.dim() > 1 else r.unsqueeze(0)
        result = torch.fft.ifft(torch.fft.fft(h, dim=-1) * torch.fft.fft(r, dim=-1).conj(), dim=-1).real
        return result.squeeze(0) if result.size(0) == 1 else result

    @staticmethod
    def multiplication(h, r):
        h = h if h.dim() > 1 else h.unsqueeze(0)
        r = r if r.dim() > 1 else r.unsqueeze(0)
        result = h * r
        return result.squeeze(0) if result.size(0) == 1 else result

class GCN(nn.Module):
    def __init__(self, num_nodes, num_relations, in_features, hidden_features, out_features, 
                 composition='circular_correlation', dropout=0.2):
        super(GCN, self).__init__()
        self.node_embedding = nn.Embedding(num_nodes, in_features)
        self.relation_embedding = nn.Embedding(num_relations, in_features)
        
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)
        
        self.layer_norm1 = nn.LayerNorm(hidden_features)
        self.layer_norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        self.composition = {
            'circular_correlation': CompositionalOperator.circular_correlation,
            'subtraction': CompositionalOperator.subtraction,
            'multiplication': CompositionalOperator.multiplication
        }[composition]
        
        self.skip1 = nn.Linear(in_features, hidden_features)
        self.skip2 = nn.Linear(hidden_features, out_features)

    def score_triplet(self, head_emb, rel_emb, tail_emb):
        composed = self.composition(head_emb, rel_emb)
        return torch.sum(composed * tail_emb, dim=-1)

    def forward(self, edges):
        node_features = self.node_embedding.weight
        edge_index = edges[:, [0, 2]].t()
        edge_features = self.relation_embedding(edges[:, 1])
        
        x1 = self.gcn1(node_features, edge_index, edge_features)
        x1 = x1 + self.skip1(node_features)
        x1 = self.layer_norm1(x1)
        x1 = self.dropout(x1)
        
        x2 = self.gcn2(x1, edge_index, edge_features)
        x2 = x2 + self.skip2(x1)
        x2 = self.layer_norm2(x2)
        x2 = self.dropout(x2)
        
        return x2

def compute_ranks(model, eval_edges, train_edges, num_nodes, batch_size=4096):
    model.eval()
    ranks = []
    eval_edges = eval_edges.to(device)
    train_edges = train_edges.to(device)
    
    with torch.no_grad():
        node_embeddings = model(train_edges).detach()
        
        for batch_start in range(0, len(eval_edges), batch_size):
            batch_end = min(batch_start + batch_size, len(eval_edges))
            batch_edges = eval_edges[batch_start:batch_end]
            
            src_nodes = batch_edges[:, 0]
            relations = batch_edges[:, 1]
            dst_nodes = batch_edges[:, 2]
            
            src_embeds = node_embeddings[src_nodes]
            rel_embeds = model.relation_embedding(relations)
            
            composed = model.composition(src_embeds, rel_embeds)
            all_scores = torch.mm(composed, node_embeddings.t())
            
            b_range = torch.arange(len(batch_edges), device=device)
            target_pred = all_scores[b_range, dst_nodes]
            
            mask = torch.zeros_like(all_scores, dtype=torch.bool)
            for i, (src, rel, dst) in enumerate(batch_edges):
                true_tails = train_edges[(train_edges[:, 0] == src) & 
                                       (train_edges[:, 1] == rel)][:, 2]
                mask[i][true_tails] = True
            
            all_scores[mask] = float('-inf')
            all_scores[b_range, dst_nodes] = target_pred
            
            ranks_batch = 1 + torch.argsort(
                torch.argsort(all_scores, dim=1, descending=True),
                dim=1,
                descending=False
            )[b_range, dst_nodes]
            
            ranks.extend(ranks_batch.cpu().tolist())
    
    return ranks

def evaluate(model, eval_edges, train_edges, num_nodes, return_dict=False):
    ranks = compute_ranks(model, eval_edges, train_edges, num_nodes)
    count = len(ranks)
    
    metrics = {
        'mrr': sum(1.0/rank for rank in ranks) / count,
        'mr': sum(ranks) / count,
        'h10': sum(1 for r in ranks if r <= 10) / count,
        'h3': sum(1 for r in ranks if r <= 3) / count,
        'h1': sum(1 for r in ranks if r <= 1) / count
    }
    
    return metrics if return_dict else format_metrics(metrics)

def format_metrics(metrics):
    return f'MRR: {metrics["mrr"]:.4f}, MR: {metrics["mr"]:.1f}, H@10: {metrics["h10"]:.4f}, H@3: {metrics["h3"]:.4f}, H@1: {metrics["h1"]:.4f}'

def train(model, train_edges, valid_edges, num_nodes, num_epochs=1000, learning_rate=0.001, batch_size=16192):
    wandb.finish()
    wandb.login(key="ADD KEY HERE")
    wandb.init(project="knowledge-graph-gcn", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_nodes": num_nodes,
        "model_type": model.__class__.__name__,
        "composition": model.composition.__name__,
        "device": device.type
    })
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    train_edges = train_edges.to(device)
    valid_edges = valid_edges.to(device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        perm = torch.randperm(len(train_edges), device=device)
        shuffled_edges = train_edges[perm]
        
        for batch_start in range(0, len(shuffled_edges), batch_size):
            batch_end = min(batch_start + batch_size, len(shuffled_edges))
            batch_edges = shuffled_edges[batch_start:batch_end]
            current_batch_size = len(batch_edges)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                node_embeddings = model(batch_edges)
                
                src_embeds = node_embeddings[batch_edges[:, 0]]
                rel_embeds = model.relation_embedding(batch_edges[:, 1])
                dst_embeds = node_embeddings[batch_edges[:, 2]]
                
                composed_src = model.composition(src_embeds, rel_embeds)
                pos_scores = torch.sum(composed_src * dst_embeds, dim=1)
                
                neg_dst = torch.randint(0, num_nodes, (current_batch_size,), device=device)
                neg_dst_embeds = node_embeddings[neg_dst]
                neg_scores = torch.sum(composed_src * neg_dst_embeds, dim=1)
                
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([
                    torch.ones(current_batch_size, device=device),
                    torch.zeros(current_batch_size, device=device)
                ])
                
                loss = criterion(scores, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * current_batch_size

        avg_loss = total_loss / len(train_edges)
        
        if (epoch + 1) % 10 == 0:
            eval_metrics = evaluate(model, valid_edges, train_edges, num_nodes, return_dict=True)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
            print(f'Validation Metrics: {format_metrics(eval_metrics)}')
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_mrr": eval_metrics['mrr'],
                "val_mr": eval_metrics['mr'],
                "val_hits@10": eval_metrics['h10'],
                "val_hits@3": eval_metrics['h3'],
                "val_hits@1": eval_metrics['h1']
            })
    
    wandb.finish()

def create_and_train_model(dataset_type='WN18RR', num_epochs=1000):
    train_edges, valid_edges, test_edges, num_nodes, num_relations, rel_mapping = init_dataset(dataset_type)
    
    train_edges = torch.tensor(train_edges, dtype=torch.long)
    valid_edges = torch.tensor(valid_edges, dtype=torch.long)
    test_edges = torch.tensor(test_edges, dtype=torch.long)
    
    model = GCN(
        num_nodes=num_nodes, 
        num_relations=num_relations, 
        in_features=256,
        hidden_features=512,
        out_features=256,
        composition='multiplication',
        dropout=0.2
    ).to(device)
    
    torch.backends.cudnn.benchmark = True
    train(model, train_edges, valid_edges, num_nodes, num_epochs=num_epochs)
    
    print("Test Evaluation:")
    print(evaluate(model, test_edges, train_edges, num_nodes))
    
    return model, rel_mapping


model, rel_mapping = create_and_train_model(dataset_type='fb15k-237')

