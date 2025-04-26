# main.py (FastAPI backend for recommendation system)

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import torch
import pandas as pd
import numpy as np
import uvicorn
import json
import os
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import re

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load embeddings
# model = torch.load('your_model.pth', map_location=torch.device('cpu'))
user_emb = torch.load("bpr_user_emb.pt",  map_location=torch.device('cpu'))
item_emb = torch.load("bpr_item_emb.pt",  map_location=torch.device('cpu'))

# Load dataset and build lookup tables
inter_file = './ml-100k/ml-100k.inter'
item_file = './ml-100k/ml-100k.item'
user_file = './ml-100k/ml-100k.user'

inter_df = pd.read_csv(inter_file, sep='\t')
item_df = pd.read_csv(item_file, sep='\t')
user_df = pd.read_csv(user_file, sep='\t')

# Build user and item mapping
user2id = {u: i for i, u in enumerate(inter_df['user_id:token'].unique())}
item2id = {i: j for j, i in enumerate(inter_df['item_id:token'].unique())}
id2user = {i: u for u, i in user2id.items()}
id2item = {i: i_name for i_name, i in item2id.items()}

# Load movie titles and genres
item_title = item_df.set_index('item_id:token')['movie_title:token_seq'].to_dict()
item_genre_raw = item_df.set_index('item_id:token')['class:token_seq'].fillna('').to_dict()

# Improved genre parsing: split by any non-word or non-comma characters
item_genre = {
    k: re.split(r"[|;/\\]+|,", v) for k, v in item_genre_raw.items()
}
id2title = {item2id[k]: v for k, v in item_title.items() if k in item2id}
id2genres = {
    item2id[k]: [g.strip() for g in v if g.strip()] for k, v in item_genre.items() if k in item2id
}

# Map to internal ids
inter_df['user'] = inter_df['user_id:token'].map(user2id)
inter_df['item'] = inter_df['item_id:token'].map(item2id)

# Map user info
user_df['user'] = user_df['user_id:token'].map(user2id)
user_info = user_df.set_index('user')[['age:token', 'gender:token', 'occupation:token', 'zip_code:token']].to_dict(orient='index')

# Compute item-level stats: genres + view count
item_genres = defaultdict(list, id2genres)
item_view_count = inter_df['item'].value_counts().to_dict()

# Train/test split
train_list, test_list = [], []
for u, g in inter_df.groupby('user'):
    if len(g) < 2: continue
    train, test = train_test_split(g, test_size=0.2, random_state=42)
    train_list.append(train)
    test_list.append(test)
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)
train_user_pos = train_df.groupby('user')['item'].apply(set).to_dict()
test_user_pos = test_df.groupby('user')['item'].apply(set).to_dict()

# Update ratings based on training set
item_ratings = train_df.groupby('item')['rating:float'].mean().round(2).to_dict()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    poster_dir = "static/movies_poster"
    posters = sorted([
        f"movies_poster/{fname}" for fname in os.listdir(poster_dir)
        if fname.endswith(".jpg")
    ])
    user_ids = sorted(test_user_pos.keys())[:100]
    return templates.TemplateResponse("index.html", {"request": request, "posters": posters, "user_ids": user_ids})

@app.post("/login")
async def login(user_id: int = Form(...)):
    return RedirectResponse(f"/user?user_id={user_id}", status_code=302)

@app.get("/user", response_class=HTMLResponse)
async def user_page(request: Request, user_id: int):
    user_hist_items = train_user_pos.get(user_id, set())
    history_info = inter_df[inter_df['user'] == user_id].sort_values(by='item')
    top_items = history_info['item'].value_counts().head(5).index.tolist()
    top_titles = [
        {
            "title": id2title.get(i, f"Item {i}"),
            "genres": ' '.join(item_genres.get(i, [])),
            "rating": item_ratings.get(i, 0),
            "year": item_df.set_index('item_id:token').get('release_year:token', {}).get(id2item.get(i, ''), '')
        }
        for i in top_items
    ]

    genre_counter = Counter()
    for i in user_hist_items:
        genre_string = ' '.join(item_genres.get(i, []))
        genres = genre_string.split()
        genre_counter.update(set(genres))
    top_genres = [g for g, _ in genre_counter.most_common(5)]

    detailed_history = []
    for i in sorted(user_hist_items):
        detailed_history.append({
            "item_id": i,
            "title": id2title.get(i, f"Item {i}"),
            "genres": ' '.join(item_genres.get(i, [])),
            "views": item_view_count.get(i, 0),
            "rating": item_ratings.get(i, 0)
        })

    scores = torch.matmul(user_emb[user_id], item_emb.T)
    scores[list(user_hist_items)] = -np.inf
    topk = torch.topk(scores, 10).indices.tolist()
    recommended_items = [
        {
            "title": id2title.get(i, f"Item {i}"),
            "genres": ' '.join(item_genres.get(i, [])),
            "rating": item_ratings.get(i, 0),
            "year": item_df.set_index('item_id:token').get('release_year:token', {}).get(id2item.get(i, ''), '')
        }
        for i in topk
    ]

    return templates.TemplateResponse("user.html", {
        "request": request,
        "user_id": user_id,
        "user_profile": user_info.get(user_id, {}),
        "history": detailed_history,
        "top_movies": top_titles,
        "top_genres": top_genres,
        "recommended": recommended_items,
        "tag_mode": True
    })

@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, user_id: int):
    user_hist_items = train_user_pos.get(user_id, set())
    history = [(i, id2title.get(i, f"Item {i}")) for i in sorted(user_hist_items)]

    scores = torch.matmul(user_emb[user_id], item_emb.T)
    scores[list(user_hist_items)] = -np.inf
    topk = torch.topk(scores, 10).indices.tolist()
    top_items = [(i, id2title.get(i, f"Item {i}")) for i in topk]

    return templates.TemplateResponse("recommend.html", {
        "request": request,
        "user_id": user_id,
        "history": history,
        "top_items": top_items
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 从环境变量里拿PORT，默认8000
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)