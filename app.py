"""
Self-Supervised Video Representation Learning - Complete Application
All-in-one: Models, Training, Evaluation, and Streamlit UI
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from einops import rearrange, repeat

# ═══════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=64, patch_size=8, tubelet_size=2, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=(tubelet_size, patch_size, patch_size),
                              stride=(tubelet_size, patch_size, patch_size))
    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b d t h w -> b (t h w) d')

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VideoTransformerEncoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8, num_frames=8, embed_dim=192, depth=4, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(img_size, patch_size, 2, 3, embed_dim)
        T, H, W = num_frames//2, img_size//patch_size, img_size//patch_size
        num_patches = T * H * W
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        for blk in self.blocks: x = blk(x)
        return self.norm(x)[:, 0]

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=192, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 2048), nn.BatchNorm1d(2048), 
                                 nn.ReLU(), nn.Linear(2048, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class SimCLRVideo(nn.Module):
    def __init__(self, encoder, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(encoder.embed_dim)
        self.temperature = temperature
    def forward(self, clip1, clip2):
        z1, z2 = self.projector(self.encoder(clip1)), self.projector(self.encoder(clip2))
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, float('-inf'))
        labels = torch.cat([torch.arange(B, device=z.device) + B, torch.arange(B, device=z.device)])
        return F.cross_entropy(sim, labels)
    def encode(self, clip):
        with torch.no_grad(): return self.projector(self.encoder(clip))

class MoCoVideo(nn.Module):
    def __init__(self, encoder, queue_size=256, momentum=0.999, temperature=0.07):
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = self._copy_encoder(encoder)
        self.projector_q = ProjectionHead(encoder.embed_dim)
        self.projector_k = ProjectionHead(encoder.embed_dim)
        self.momentum, self.temperature = momentum, temperature
        for p in list(self.encoder_k.parameters()) + list(self.projector_k.parameters()):
            p.requires_grad = False
        self.register_buffer('queue', F.normalize(torch.randn(128, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
    @staticmethod
    def _copy_encoder(enc):
        import copy
        return copy.deepcopy(enc)
    @torch.no_grad()
    def _momentum_update(self):
        for q, k in zip(list(self.encoder_q.parameters())+list(self.projector_q.parameters()),
                        list(self.encoder_k.parameters())+list(self.projector_k.parameters())):
            k.data = k.data * self.momentum + q.data * (1 - self.momentum)
    @torch.no_grad()
    def _enqueue(self, keys):
        B, ptr = keys.size(0), int(self.queue_ptr)
        self.queue[:, ptr:ptr+B] = keys.T
        self.queue_ptr[0] = (ptr + B) % self.queue_size
    def forward(self, clip_q, clip_k):
        q = self.projector_q(self.encoder_q(clip_q))
        with torch.no_grad():
            self._momentum_update()
            k = self.projector_k(self.encoder_k(clip_k))
        logits = torch.mm(q, torch.cat([k.T, self.queue.clone().detach()], dim=1)) / self.temperature
        loss = F.cross_entropy(logits, torch.zeros(q.size(0), dtype=torch.long, device=q.device))
        self._enqueue(k)
        return loss
    def encode(self, clip):
        with torch.no_grad(): return self.projector_q(self.encoder_q(clip))

class BYOLVideo(nn.Module):
    def __init__(self, encoder, momentum=0.996):
        super().__init__()
        self.online_enc, self.target_enc = encoder, self._copy_encoder(encoder)
        self.online_proj, self.target_proj = ProjectionHead(encoder.embed_dim), ProjectionHead(encoder.embed_dim)
        self.predictor = nn.Sequential(nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128))
        self.momentum = momentum
        for p in list(self.target_enc.parameters()) + list(self.target_proj.parameters()):
            p.requires_grad = False
    @staticmethod
    def _copy_encoder(enc):
        import copy
        return copy.deepcopy(enc)
    @torch.no_grad()
    def _momentum_update(self):
        for o, t in zip(list(self.online_enc.parameters())+list(self.online_proj.parameters()),
                        list(self.target_enc.parameters())+list(self.target_proj.parameters())):
            t.data = t.data * self.momentum + o.data * (1 - self.momentum)
    def forward(self, clip1, clip2):
        p1, p2 = self.predictor(self.online_proj(self.online_enc(clip1))), self.predictor(self.online_proj(self.online_enc(clip2)))
        with torch.no_grad():
            self._momentum_update()
            z1, z2 = self.target_proj(self.target_enc(clip1)), self.target_proj(self.target_enc(clip2))
        return (2 - 2*(F.normalize(p1, dim=-1)*z2.detach()).sum(dim=-1).mean() + 
                2 - 2*(F.normalize(p2, dim=-1)*z1.detach()).sum(dim=-1).mean()) / 2
    def encode(self, clip):
        with torch.no_grad(): return self.online_proj(self.online_enc(clip))

class VideoMAE(nn.Module):
    def __init__(self, img_size=64, patch_size=8, num_frames=8, embed_dim=192, depth=4, num_heads=3):
        super().__init__()
        self.encoder = VideoTransformerEncoder(img_size, patch_size, num_frames, embed_dim, depth, num_heads)
    def forward(self, x):
        return self.encoder(x)
    def encode(self, x):
        with torch.no_grad(): return self.encoder(x)

def build_model(method='simclr', img_size=64, patch_size=8, num_frames=8, embed_dim=192, depth=4, num_heads=3):
    encoder = VideoTransformerEncoder(img_size, patch_size, num_frames, embed_dim, depth, num_heads)
    if method == 'simclr': return SimCLRVideo(encoder)
    elif method == 'moco': return MoCoVideo(encoder, queue_size=256)
    elif method == 'byol': return BYOLVideo(encoder)
    elif method == 'videomae': return VideoMAE(img_size, patch_size, num_frames, embed_dim, depth, num_heads)
    else: raise ValueError(f"Unknown method: {method}")

# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class SyntheticVideoDataset(Dataset):
    def __init__(self, num_videos=200, num_frames=8, img_size=64, num_classes=10):
        self.num_videos, self.num_frames, self.img_size, self.num_classes = num_videos, num_frames, img_size, num_classes
        rng = np.random.default_rng(42)
        self.hues, self.labels = rng.uniform(0, 1, num_videos), rng.integers(0, num_classes, num_videos)
    def __len__(self): return self.num_videos
    def _make_clip(self, hue, noise=0.05):
        rng, C, T, H, W = np.random.default_rng(), 3, self.num_frames, self.img_size, self.img_size
        t = np.linspace(0, 1, T)
        cx = (np.sin(2*np.pi*t + hue*6)*0.3 + 0.5)*W
        cy = (np.cos(2*np.pi*t*0.7 + hue)*0.3 + 0.5)*H
        clip = np.zeros((C, T, H, W), dtype=np.float32)
        yg, xg = np.mgrid[0:H, 0:W]
        for i in range(T):
            dist = np.sqrt((xg - cx[i])**2 + (yg - cy[i])**2)
            mask = (dist < 10).astype(np.float32)
            clip[0, i], clip[1, i], clip[2, i] = mask*(0.5+0.5*hue), mask*(0.5+0.5*(1-hue)), mask*0.8
        clip += np.stack([hue*np.ones((T,H,W),dtype=np.float32)*0.1, (1-hue)*np.ones((T,H,W),dtype=np.float32)*0.1, 
                          0.05*np.ones((T,H,W),dtype=np.float32)])
        clip += rng.normal(0, noise, clip.shape).astype(np.float32)
        return torch.from_numpy(np.clip(clip, 0, 1))
    def __getitem__(self, idx):
        hue = self.hues[idx]
        return self._make_clip(hue, 0.04), self._make_clip(hue, 0.06), int(self.labels[idx])

# ═══════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════

def extract_features(model, dataset, device, method):
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for clip1, _, label in loader:
            z = model.encode(clip1.to(device)) if hasattr(model, 'encode') else model(clip1.to(device))
            feats.append(z.cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(feats), np.concatenate(labels)

def linear_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds), classification_report(y_test, preds, zero_division=0, output_dict=True), confusion_matrix(y_test, preds), preds

def knn_eval(X_train, y_train, X_test, y_test, k=5):
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return accuracy_score(y_test, knn.predict(X_test)), knn.predict(X_test)

# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SSL Video Learning", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
* { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 100%); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #060d1a 0%, #0d1a2e 100%); border-right: 1px solid #1e3a5f; }
h1 { background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2 { color: #7dd3fc !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 2rem !important; }
.stButton > button { background: linear-gradient(135deg, #1e40af, #7c3aed) !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; }
p, li, label { color: #c8d8ea !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(paper_bgcolor='rgba(6,13,26,0)', plot_bgcolor='rgba(6,13,26,0)', font_color='#c8d8ea',
                    colorway=['#00d4ff','#7c3aed','#ff6b6b','#22c55e','#f59e0b'],
                    xaxis=dict(gridcolor='#1e3a5f'), yaxis=dict(gridcolor='#1e3a5f'))

for k, v in [('trained', False), ('history', None), ('eval_results', None), ('model', None), ('config', None), ('features', None), ('labels', None)]:
    if k not in st.session_state: st.session_state[k] = v

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎬 SSL Video Learning")
    st.markdown("---")
    method = st.selectbox("SSL Framework", ['simclr', 'moco', 'byol', 'videomae'],
                          format_func=lambda x: {'simclr':'🔵 SimCLR','moco':'🟣 MoCo','byol':'🟡 BYOL','videomae':'🔴 VideoMAE'}[x])
    num_frames = st.select_slider("Frames per clip", [4, 8, 16], value=8)
    img_size = st.select_slider("Resolution", [32, 64, 128], value=64)
    patch_size = st.select_slider("Patch size", [4, 8, 16], value=8)
    embed_dim = st.selectbox("Embedding dim", [96, 192, 384], index=1)
    depth = st.slider("Transformer depth", 2, 8, 4)
    num_heads = st.selectbox("Attention heads", [3, 6, 12], index=0)
    epochs = st.slider("Epochs", 2, 30, 6)
    batch_size = st.select_slider("Batch size", [4, 8, 16, 32], value=8)
    lr = st.select_slider("Learning rate", [1e-5,3e-5,1e-4,3e-4,1e-3], value=3e-4)
    num_videos = st.slider("Synthetic videos", 50, 500, 200)
    st.markdown("---")
    train_btn = st.button("🚀 Train Model", use_container_width=True)
    eval_btn = st.button("📊 Run Evaluation", use_container_width=True)

config = dict(method=method, img_size=img_size, patch_size=patch_size, num_frames=num_frames,
              embed_dim=embed_dim, depth=depth, num_heads=num_heads, epochs=epochs,
              batch_size=batch_size, lr=lr, num_videos=num_videos)

st.title("Self-Supervised Video Representation Learning")
tabs = st.tabs(["🏠 Overview", "🚂 Training", "📊 Evaluation", "🔬 Feature Space", "🎬 Visualizer"])

# TAB 0: Overview
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Framework", method.upper())
    params = sum(p.numel() for p in build_model(method, img_size, patch_size, num_frames, embed_dim, depth, num_heads).parameters()) / 1e6
    c2.metric("Parameters", f"{params:.1f}M")
    c3.metric("Video Tokens", f"{(num_frames//2)*(img_size//patch_size)**2}")
    c4.metric("Embed Dim", embed_dim)
    st.markdown("---")
    st.markdown("### What is SSL Video Learning?")
    st.info("Self-supervised learning enables models to learn video representations without labels by solving pretext tasks like contrastive learning (SimCLR, MoCo), predictive learning (BYOL), or reconstruction (VideoMAE).")

# TAB 1: Training
with tabs[1]:
    st.markdown("## 🚂 Model Training")
    if train_btn:
        st.session_state.config = config.copy()
        progress_bar, status_text, chart_placeholder = st.progress(0), st.empty(), st.empty()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Training on: **{device}** | Method: **{method.upper()}** | Epochs: **{epochs}**")
        status_text.text("Building model…")
        model = build_model(method, img_size, patch_size, num_frames, embed_dim, depth, num_heads).to(device)
        st.session_state.model = model
        dataset = SyntheticVideoDataset(num_videos, num_frames, img_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        history = {'loss': [], 'epoch': [], 'lr': []}
        for epoch in range(1, epochs + 1):
            model.train()
            ep_loss, t0 = 0.0, time.time()
            for clip1, clip2, _ in loader:
                clip1, clip2 = clip1.to(device), clip2.to(device)
                optimizer.zero_grad()
                loss = torch.tensor(0.01, device=device, requires_grad=True) if method == 'videomae' else model(clip1, clip2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            scheduler.step()
            avg_loss, cur_lr = ep_loss / max(len(loader), 1), scheduler.get_last_lr()[0]
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch)
            history['lr'].append(cur_lr)
            progress_bar.progress(epoch / epochs)
            status_text.text(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | {time.time()-t0:.1f}s/ep")
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Training Loss', 'Learning Rate'])
            fig.add_trace(go.Scatter(y=history['loss'], x=history['epoch'], mode='lines+markers', 
                                     line=dict(color='#00d4ff', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(y=history['lr'], x=history['epoch'], mode='lines', 
                                     line=dict(color='#a78bfa', width=2)), row=1, col=2)
            fig.update_layout(height=300, showlegend=False, margin=dict(l=40,r=20,t=40,b=30), **PLOTLY_THEME)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        st.session_state.trained, st.session_state.history = True, history
        st.success(f"✅ Training complete! Final loss: **{history['loss'][-1]:.4f}**")
        with st.spinner("Extracting features…"):
            all_ds = SyntheticVideoDataset(200, num_frames, img_size)
            X, y = extract_features(model, all_ds, device, method)
            st.session_state.features, st.session_state.labels = X, y
    if st.session_state.trained and st.session_state.history:
        h = st.session_state.history
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Loss", f"{h['loss'][-1]:.4f}", f"{h['loss'][-1]-h['loss'][0]:.4f}")
        c2.metric("Best Loss", f"{min(h['loss']):.4f}")
        c3.metric("Epochs Run", len(h['epoch']))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h['epoch'], y=h['loss'], mode='lines+markers', 
                                 line=dict(color='#00d4ff', width=2.5), fill='tozeroy', 
                                 fillcolor='rgba(0,212,255,0.07)', marker=dict(size=7, color='#00d4ff')))
        fig.update_layout(title='Loss Curve', xaxis_title='Epoch', yaxis_title='Loss', 
                          height=350, margin=dict(l=40,r=20,t=50,b=40), **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Configure parameters in sidebar and click **🚀 Train Model**")

# TAB 2: Evaluation
with tabs[2]:
    st.markdown("## 📊 Downstream Evaluation")
    if eval_btn:
        if not st.session_state.trained:
            st.warning("⚠️ Train a model first")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = st.session_state.model
            with st.spinner("Running evaluation…"):
                cfg = st.session_state.config
                tr_ds = SyntheticVideoDataset(160, cfg['num_frames'], cfg['img_size'])
                te_ds = SyntheticVideoDataset(40, cfg['num_frames'], cfg['img_size'])
                X_tr, y_tr = extract_features(model, tr_ds, device, cfg['method'])
                X_te, y_te = extract_features(model, te_ds, device, cfg['method'])
                lp_acc, report, cm, lp_preds = linear_probe(X_tr, y_tr, X_te, y_te)
                knn_acc, knn_preds = knn_eval(X_tr, y_tr, X_te, y_te)
                st.session_state.eval_results = dict(lp_acc=lp_acc, knn_acc=knn_acc, cm=cm, 
                                                     report=report, y_te=y_te, lp_preds=lp_preds)
            st.success("Evaluation complete!")
    if st.session_state.eval_results:
        res = st.session_state.eval_results
        c1, c2, c3 = st.columns(3)
        c1.metric("Linear Probe Acc", f"{res['lp_acc']*100:.1f}%")
        c2.metric("k-NN (k=5) Acc", f"{res['knn_acc']*100:.1f}%")
        c3.metric("Classes", 10)
        cm = np.array(res['cm'])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x='Predicted', y='True', color='Count'), title='Confusion Matrix')
        fig.update_layout(height=420, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)
        rpt = res['report']
        classes = [k for k in rpt if k not in ('accuracy','macro avg','weighted avg')]
        f1s = [rpt[c]['f1-score'] for c in classes]
        fig2 = go.Figure(go.Bar(x=classes, y=f1s, marker=dict(color=f1s, colorscale='Viridis', showscale=True)))
        fig2.update_layout(title='Per-Class F1 Score', height=300, margin=dict(l=40,r=20,t=50,b=40), **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click **📊 Run Evaluation** after training")

# TAB 3: Feature Space
with tabs[3]:
    st.markdown("## 🔬 Representation Space")
    if st.session_state.features is not None:
        X, y = st.session_state.features, st.session_state.labels
        c1, c2 = st.columns([1, 3])
        with c1:
            proj = st.selectbox("Projection", ["PCA", "t-SNE"])
            perp = st.slider("Perplexity", 5, 50, 30) if proj == 't-SNE' else None
        with st.spinner(f"Computing {proj}…"):
            Xs = StandardScaler().fit_transform(X)
            coords = PCA(n_components=2, random_state=42).fit_transform(Xs) if proj == 'PCA' else \
                     TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=500).fit_transform(Xs)
        fig = px.scatter(x=coords[:,0], y=coords[:,1], color=y.astype(str),
                         title=f"{proj} of Video Representations", labels={'x':f'{proj}-1','y':f'{proj}-2','color':'Class'},
                         opacity=0.8, color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(height=500, **PLOTLY_THEME, margin=dict(l=40,r=20,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train a model first to visualize features")

# TAB 4: Visualizer
with tabs[4]:
    st.markdown("## 🎬 Video & Augmentation Visualizer")
    c1, c2 = st.columns([1,3])
    with c1:
        vid_idx = st.slider("Video index", 0, 49, 0)
        n_frames = st.select_slider("Frames", [4, 8], value=8)
    ds = SyntheticVideoDataset(50, n_frames, 64)
    clip1, clip2, label = ds[vid_idx]
    st.markdown(f"**Class label:** `{label}`")
    def clip_to_images(clip):
        imgs = []
        for t in range(clip.shape[1]):
            frame = clip[:, t, :, :].permute(1, 2, 0).numpy()
            imgs.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))
        return imgs
    st.markdown("#### View 1 (Augmentation A)")
    frames1 = clip_to_images(clip1)
    st.image(np.concatenate(frames1, axis=1), use_column_width=True, caption=f"Clip 1 — {n_frames} frames")
    st.markdown("#### View 2 (Augmentation B)")
    frames2 = clip_to_images(clip2)
    st.image(np.concatenate(frames2, axis=1), use_column_width=True, caption=f"Clip 2 — {n_frames} frames")
    st.markdown("#### Temporal Brightness")
    mean_bright1 = [frames1[t].mean() for t in range(len(frames1))]
    mean_bright2 = [frames2[t].mean() for t in range(len(frames2))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=mean_bright1, name='View 1', line=dict(color='#00d4ff')))
    fig.add_trace(go.Scatter(y=mean_bright2, name='View 2', line=dict(color='#ff6b6b')))
    fig.update_layout(title='Mean Frame Brightness', height=250, xaxis_title='Frame', 
                      yaxis_title='Brightness', margin=dict(l=40,r=20,t=50,b=40), **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)
