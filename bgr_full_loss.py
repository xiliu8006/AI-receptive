import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import label_ranking_average_precision_score
import math

def evaluate_all_metrics(model, k=10):
    model.eval()
    with torch.no_grad():
        user_emb = model.user_emb.weight
        item_emb = model.item_emb.weight
        scores = torch.matmul(user_emb, item_emb.T)  # [n_users, n_items]

        recall_list, precision_list, ndcg_list, hit_list, mrr_list = [], [], [], [], []

        for user in test_user_pos:
            score = scores[user].clone()
            train_items = train_user_pos.get(user, set())
            test_items = test_user_pos[user]

            # Mask training items
            score[list(train_items)] = -np.inf

            # Get top-k ranked item indices
            _, top_k_items = torch.topk(score, k)
            top_k_items = top_k_items.cpu().numpy().tolist()

            hit_count = len(set(top_k_items) & test_items)
            recall = hit_count / min(len(test_items), k)
            precision = hit_count / k
            hit = 1.0 if hit_count > 0 else 0.0

            # NDCG
            dcg = 0.0
            idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(test_items), k))])
            for rank, item in enumerate(top_k_items):
                if item in test_items:
                    dcg += 1.0 / math.log2(rank + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            # MRR
            rr = 0.0
            for rank, item in enumerate(top_k_items):
                if item in test_items:
                    rr = 1.0 / (rank + 1)
                    break

            recall_list.append(recall)
            precision_list.append(precision)
            hit_list.append(hit)
            ndcg_list.append(ndcg)
            mrr_list.append(rr)

        print(f"\nEvaluation Results @ {k}:")
        print(f"recall@{k:<4}: {np.mean(recall_list):.4f}")
        print(f"mrr@{k:<7}: {np.mean(mrr_list):.4f}")
        print(f"ndcg@{k:<6}: {np.mean(ndcg_list):.4f}")
        print(f"hit@{k:<7}: {np.mean(hit_list):.4f}")
        print(f"precision@{k}: {np.mean(precision_list):.4f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Config ====
base_dir = './ml-100k'
inter_file = os.path.join(base_dir, 'ml-100k.inter')
embedding_size = 64
epochs = 50
batch_size = 2048
lr = 0.001
reg_weight = 1e-4

# ==== Load Data ====
df = pd.read_csv(inter_file, sep='\t')
user2id = {u: i for i, u in enumerate(df['user_id:token'].unique())}
item2id = {i: j for j, i in enumerate(df['item_id:token'].unique())}
df['user'] = df['user_id:token'].map(user2id)
df['item'] = df['item_id:token'].map(item2id)

n_users = len(user2id)
n_items = len(item2id)

# Train/test split
train_list, test_list = [], []
for u, g in df.groupby('user'):
    if len(g) < 2: continue
    train, test = train_test_split(g, test_size=0.2, random_state=42)
    train_list.append(train)
    test_list.append(test)
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)
train_user_pos = train_df.groupby('user')['item'].apply(set).to_dict()
test_user_pos = test_df.groupby('user')['item'].apply(set).to_dict()

# ==== BPR Model ====
class BPRMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, embedding_size):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embedding_size)
        self.item_emb = nn.Embedding(n_items, embedding_size)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, items):
        return self.user_emb(users), self.item_emb(items)

# ==== Loss ====
class BPRLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        return -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).mean()

# ==== Sampling ====
def sample_batch():
    users = np.random.choice(list(train_user_pos.keys()), batch_size)
    pos_items, neg_items = [], []
    for u in users:
        pos = np.random.choice(list(train_user_pos[u]))
        neg = np.random.randint(n_items)
        while neg in train_user_pos[u]:
            neg = np.random.randint(n_items)
        pos_items.append(pos)
        neg_items.append(neg)
    return torch.LongTensor(users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)

# ==== Evaluation ====
def recall_at_k(model, k=10):
    model.eval()
    with torch.no_grad():
        user_e = model.user_emb.weight
        item_e = model.item_emb.weight
        scores = torch.matmul(user_e, item_e.t())
        recall = []
        for u in test_user_pos:
            s = scores[u].clone()
            s[list(train_user_pos.get(u, []))] = -np.inf
            topk = torch.topk(s, k)[1].cpu().numpy()
            hits = len(set(topk) & test_user_pos[u])
            recall.append(hits / min(len(test_user_pos[u]), k))
        return np.mean(recall)

# ==== Train Loop ====
model = BPRMatrixFactorization(n_users, n_items, embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
bpr_loss_fn = BPRLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for _ in range(len(train_df) // batch_size):
        u, pi, ni = sample_batch()
        u, pi, ni = u.to(device), pi.to(device), ni.to(device)
        ue, pie = model(u, pi)
        nie = model.item_emb(ni)
        pos = (ue * pie).sum(-1)
        neg = (ue * nie).sum(-1)
        loss = bpr_loss_fn(pos, neg) + reg_weight * (ue.norm(2) + pie.norm(2) + nie.norm(2)) / u.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    evaluate_all_metrics(model, k=10)
    # print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}, Recall@10: {recall:.4f}")

print("✅ BPR training completed.")
