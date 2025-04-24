# AI-receptive
# BPR and LightGCN from Scratch (PyTorch)

This repository provides a complete from-scratch implementation of the **LightGCN** and **BPR**recommendation model using **PyTorch**, without any dependency on RecBole. It is adapted to use datasets in **RecBole `.inter` format**, such as MovieLens 100K (`ml-100k`).

---

## 🔧 Features

- ✅ Train-test split **per user (80/20)**
- ✅ Graph construction from **training set only**
- ✅ BPR loss with **embedding + L2 regularization**
- ✅ Evaluation with **Recall@10, Precision@10, Hit@10, NDCG@10, MRR@10**
- ✅ Ready to run: just point to your `.inter` file

---

## 📁 File

- `bgr_full_loss.py` - Main script for training and evaluation of LightGCN
- `lightgcn_full_loss.py` - Main script for training and evaluation of LightGCN

---

## 📂 Dataset Format

This script expects data in RecBole `.inter` format with at least the following columns:

```
user_id:token   item_id:token   rating:float
```

Example file: `ml-100k.inter`

---

## 🚀 How to Run

1. Make sure your `.inter` dataset is saved under:

```
./ml-100k/ml-100k.inter
```

2. Run the script:

```bash
python bgr_full_loss.py
python lightgcn_full_loss.py
```

---

## 🧪 Evaluation Output

Each epoch prints metrics on the test set:

```
Epoch 1/50, Loss: 138.9241
recall@10  : 0.2342
mrr@10     : 0.3181
ndcg@10    : 0.2859
hit@10     : 0.7683
precision@10: 0.1842
```

---

## 🛠 Dependencies

- Python 3.7+
- PyTorch >= 1.7
- pandas, numpy, scipy, scikit-learn

You can install required packages with:

```bash
pip install torch pandas numpy scipy scikit-learn
```

---

## 📌 Notes

- Adjacency matrix is built from training data only to prevent information leakage.
- Train/test split is done per-user for realistic cold-start evaluation.

---

## 📣 Credit

Based on the SIGIR 2020 paper:

**LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**  
*Xiangnan He et al.*

**Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback."**  
*S Rendle et al.*
For academic use only.
