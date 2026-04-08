import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer, T5EncoderModel
def extract_features(
    cnn_model,
    rating_tensor,
    item_labels_tensor,
    device,
    chunk_size=256,
    to_cpu=True,
    use_amp=True,
):
    cnn_model.eval()
    rating_tensor = rating_tensor.to(device)
    item_labels_tensor = item_labels_tensor.to(device)
    fixed_bs = getattr(cnn_model, "batch_size", None)
    eff_chunk = fixed_bs if (fixed_bs is not None and fixed_bs > 0) else chunk_size
    features_list = []
    num_samples = rating_tensor.size(0)
    autocast_enabled = (use_amp and device.type == "cuda")
    with torch.no_grad():
        for start in range(0, num_samples, eff_chunk):
            chunk = rating_tensor[start:start + eff_chunk]
            b = chunk.size(0)
            if fixed_bs is not None and b < fixed_bs:
                pad = chunk[-1:].repeat(fixed_bs - b, 1)
                chunk_in = torch.cat([chunk, pad], dim=0)
            else:
                chunk_in = chunk
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                if hasattr(cnn_model, "forward_features"):
                    out = cnn_model.forward_features(chunk_in, item_labels_tensor)
                else:
                    out = cnn_model(chunk_in, item_labels_tensor)
            if fixed_bs is not None and b < fixed_bs:
                out = out[:b]
            out = out.detach().float()
            if to_cpu:
                out = out.cpu()
            features_list.append(out)
    features = torch.cat(features_list, dim=0)
    return features

def load_item_label_data(features_file, num_items,
                         model_name="t5-small",
                         out_dim=16,
                         batch_size=64,
                         max_length=64,
                         device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(
        features_file,
        delimiter="::",
        engine="python",
        header=None,
        usecols=[0, 2],
        encoding="ISO-8859-1",
        names=["item_id", "features"],
    )
    df["item_id"] = df["item_id"].astype(int)
    texts = df["features"].fillna("").astype(str).str.replace("|", ", ", regex=False).tolist()
    item_ids = df["item_id"].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()
    with torch.no_grad():
        dummy = tokenizer(["test"], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        dummy = {k: v.to(device) for k, v in dummy.items()}
        out = model(**dummy)
        hidden_dim = out.last_hidden_state.shape[-1]
    proj = nn.Linear(hidden_dim, out_dim, bias=False).to(device)
    proj.eval()
    E = np.zeros((num_items, out_dim), dtype=np.float32)
    req_rows, req_texts = [], []
    for iid, t in zip(item_ids, texts):
        row = iid - 1
        if 0 <= row < num_items:
            t = t.strip()
            if t:
                req_rows.append(row)
                req_texts.append(t)
    if not req_texts:
        return E
    for s in range(0, len(req_texts), batch_size):
        batch_texts = req_texts[s:s + batch_size]
        batch_rows  = req_rows[s:s + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = proj(pooled)
        pooled_list = pooled.detach().cpu().tolist()
        for row, vec in zip(batch_rows, pooled_list):
            E[row] = np.asarray(vec, dtype=np.float32)
    return E


def amazon_item(
    category_file_path,
    brand_file_path,
    num_items=2752,
    model_name="t5-small",
    out_dim=16,
    batch_size=64,
    max_length=64,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    item_to_categories = defaultdict(list)
    with open(category_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            item_str, cat_str = parts
            try:
                item_id = int(item_str)
                cat_id = int(cat_str)
            except ValueError:
                continue
            item_to_categories[item_id].append(cat_id)
    for item_id in item_to_categories:
        item_to_categories[item_id] = list(dict.fromkeys(item_to_categories[item_id]))
    item_to_brand = {}
    with open(brand_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            item_str, brand_str = parts
            try:
                item_id = int(item_str)
                brand_id = int(brand_str)
            except ValueError:
                continue
            item_to_brand[item_id] = brand_id
    all_item_ids = sorted(set(item_to_categories.keys()) | set(item_to_brand.keys()))
    item_texts = {}
    for item_id in all_item_ids:
        parts = []
        if item_id in item_to_categories:
            for cat_id in item_to_categories[item_id]:
                parts.append(f"category_{cat_id}")

        if item_id in item_to_brand:
            parts.append(f"brand_{item_to_brand[item_id]}")
        text = ", ".join(parts).strip()
        if text:
            item_texts[item_id] = text
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device)
    encoder.eval()
    with torch.no_grad():
        dummy = tokenizer(
            ["test"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        hidden_dim = encoder(**dummy).last_hidden_state.shape[-1]
    proj = nn.Linear(hidden_dim, out_dim, bias=False).to(device)
    proj.eval()
    E = np.zeros((num_items, out_dim), dtype=np.float32)
    if not item_texts:
        return E
    item_ids = sorted(item_texts.keys())
    text_list = [item_texts[i] for i in item_ids]
    for s in range(0, len(text_list), batch_size):
        batch_ids = item_ids[s:s + batch_size]
        batch_texts = text_list[s:s + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = encoder(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vecs = proj(pooled)
        vecs = vecs.detach().cpu().numpy().astype(np.float32)
        for item_id, vec in zip(batch_ids, vecs):
            if 0 <= item_id < num_items:
                E[item_id] = vec
    return E