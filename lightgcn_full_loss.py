import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import scipy.sparse as sp
import numpy as np
import os
import math
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Config ====
base_dir = './ml-100k'
inter_file = os.path.join(base_dir, 'ml-100k.inter')
latent_dim = 64
n_layers = 3
epochs = 50
batch_size = 2048
lr = 0.001
reg_weight = 1e-4
emb_weight = 1e-4

# ==== Load Data ====
df = pd.read_csv(inter_file, sep='\t')
user_list = df['user_id:token'].unique()
item_list = df['item_id:token'].unique()
user2id = {u: i for i, u in enumerate(user_list)}
item2id = {it: i for i, it in enumerate(item_list)}
df['user'] = df['user_id:token'].map(user2id)
df['item'] = df['item_id:token'].map(item2id)
n_users = len(user2id)
n_items = len(item2id)

# ==== Train/Test Split ====
train_list, test_list = [], []
for user, group in df.groupby('user'):
    if len(group) < 2: continue
    train, test = train_test_split(group, test_size=0.2, random_state=42)
    train_list.append(train)
    test_list.append(test)
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)
train_user_pos = train_df.groupby('user')['item'].apply(set).to_dict()
test_user_pos = test_df.groupby('user')['item'].apply(set).to_dict()
user_pos_dict = train_user_pos  # for sampling

# ==== Build Adjacency Matrix from Train ====
def build_adj_matrix(train_df):
    mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    for user, item in zip(train_df['user'], train_df['item']):
        mat[user, n_users + item] = 1.0
        mat[n_users + item, user] = 1.0
    mat = mat.tocsr()
    rowsum = np.array(mat.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    norm_mat = d_mat.dot(mat).dot(d_mat).tocoo()
    idx = torch.LongTensor([norm_mat.row, norm_mat.col])
    val = torch.FloatTensor(norm_mat.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(norm_mat.shape)).to(device)

norm_adj_mat = build_adj_matrix(train_df)

# ==== LightGCN Model ====
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, latent_dim, n_layers):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, latent_dim)
        self.item_emb = nn.Embedding(n_items, latent_dim)
        self.n_layers = n_layers
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_x = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj_mat, x)
            all_x.append(x)
        out = torch.stack(all_x, dim=1).mean(1)
        return out[:n_users], out[n_users:]

# ==== Losses ====
class BPRLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        return -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).mean()

class EmbLoss(nn.Module):
    def forward(self, *embs, require_pow=False):
        loss = torch.zeros(1, device=embs[0].device)
        for e in embs:
            loss += torch.norm(e, p=2)
        return loss / embs[0].shape[0]

class RegLoss(nn.Module):
    def forward(self, params):
        return sum(p.norm(2) for p in params)

# ==== Evaluation ====
def evaluate_all_metrics(model, k=10):
    model.eval()
    with torch.no_grad():
        user_e, item_e = model()
        scores = torch.matmul(user_e, item_e.T)
        recall_list, precision_list, ndcg_list, hit_list, mrr_list = [], [], [], [], []
        for u in test_user_pos:
            score = scores[u].clone()
            score[list(train_user_pos.get(u, []))] = -np.inf
            topk = torch.topk(score, k)[1].cpu().numpy().tolist()
            hits = len(set(topk) & test_user_pos[u])
            recall = hits / len(test_user_pos[u])
            precision = hits / k
            hit = 1.0 if hits > 0 else 0.0
            dcg = sum([1.0 / math.log2(rank + 2) for rank, i in enumerate(topk) if i in test_user_pos[u]])
            idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(test_user_pos[u]), k))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            rr = 0.0
            for rank, i in enumerate(topk):
                if i in test_user_pos[u]:
                    rr = 1.0 / (rank + 1)
                    break
            recall_list.append(recall)
            precision_list.append(precision)
            hit_list.append(hit)
            ndcg_list.append(ndcg)
            mrr_list.append(rr)
        print(f"recall@{k:<4}: {np.mean(recall_list):.4f}")
        print(f"mrr@{k:<7}: {np.mean(mrr_list):.4f}")
        print(f"ndcg@{k:<6}: {np.mean(ndcg_list):.4f}")
        print(f"hit@{k:<7}: {np.mean(hit_list):.4f}")
        print(f"precision@{k}: {np.mean(precision_list):.4f}")

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

# ==== Training ====
model = LightGCN(n_users, n_items, latent_dim, n_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
bpr_loss = BPRLoss()
emb_loss = EmbLoss()
reg_loss = RegLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for _ in range(len(train_df) // batch_size):
        u, pi, ni = sample_batch()
        u, pi, ni = u.to(device), pi.to(device), ni.to(device)
        ue, ie = model()
        pos = (ue[u] * ie[pi]).sum(-1)
        neg = (ue[u] * ie[ni]).sum(-1)
        loss = bpr_loss(pos, neg) + emb_weight * emb_loss(ue[u], ie[pi], ie[ni]) + reg_weight * reg_loss(model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    evaluate_all_metrics(model, k=10)
    
user_emb, item_emb = model()
torch.save(user_emb, "user_emb.pt")
torch.save(item_emb, "item_emb.pt")

print("✅ Training completed.")
