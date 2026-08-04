# Шаг 1 (titans_qwen.md): классификатор know/not-know по hidden на позиции ДО значения факта.
# Вход: hidden последнего слоя (896) + энтропия + logprob первого токена значения.
# 5-fold CV, сравнение с эвристиками (порог энтропии / порог logprob).
# Это онлайн-сигнал новизны для η/θ-гейта: один forward, без второго прогона.

import torch, torch.nn as nn, numpy as np
from sklearn.metrics import roc_auc_score

DEV = "cuda"

class Clf(nn.Module):
    def __init__(self, d=896, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 2, h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def zscore(X, mu, sd):
    return (X - mu) / (sd + 1e-8)

def best_threshold(y, s):
    """лучший порог по accuracy на train"""
    best, bt = -1, 0.0
    for t in np.quantile(s.numpy(), np.linspace(0.05, 0.95, 50)):
        acc = ((s > t) == (y > 0.5)).float().mean().item()
        if acc > best:
            best, bt = acc, t
    return bt, best

def main():
    d = torch.load("dataset_knowledge.pt")
    X_raw = torch.cat([d["hidden"], d["entropy"].unsqueeze(1), d["logprob"].unsqueeze(1)], dim=1)
    y = d["labels"].float()
    n = len(y)
    print(f"датасет: {n} примеров (know={int(y.sum())}, notknow={int(n - y.sum())})")

    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 5)

    accs, aucs = [], []
    acc_ent, acc_lp = [], []
    for fi in range(5):
        val_i = folds[fi]; tr_i = np.concatenate([f for j, f in enumerate(folds) if j != fi])
        tr_i, val_i = torch.tensor(tr_i), torch.tensor(val_i)
        mu, sd = X_raw[tr_i].mean(0), X_raw[tr_i].std(0)
        Xtr, Xva = zscore(X_raw[tr_i], mu, sd), zscore(X_raw[val_i], mu, sd)
        ytr, yva = y[tr_i], y[val_i]

        model = Clf().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        Xtr_c, ytr_c = Xtr.to(DEV), ytr.to(DEV)
        for ep in range(300):
            perm = torch.randperm(len(Xtr), device=DEV)
            model.train()
            for b in range(0, len(Xtr), 16):
                bi = perm[b:b+16]
                opt.zero_grad()
                loss = lossf(model(Xtr_c[bi]), ytr_c[bi])
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva.to(DEV))).cpu()
        acc = ((p > 0.5) == (yva > 0.5)).float().mean().item()
        auc = roc_auc_score(yva.numpy(), p.numpy()) if yva.sum() not in (0, len(yva)) else 0.5
        accs.append(acc); aucs.append(auc)

        # эвристики: порог по entropy и по logprob, подобранный на train
        t_ent, _ = best_threshold(ytr, d["entropy"][tr_i])
        t_lp, _ = best_threshold(ytr, d["logprob"][tr_i])
        acc_ent.append(((d["entropy"][val_i] > t_ent) == (yva > 0.5)).float().mean().item())
        acc_lp.append(((d["logprob"][val_i] > t_lp) == (yva > 0.5)).float().mean().item())
        print(f"fold {fi}: clf acc={acc:.2f} auc={auc:.3f} | ent-порог acc={acc_ent[-1]:.2f} "
              f"| logprob-порог acc={acc_lp[-1]:.2f}")

    print(f"\n5-fold: классификатор acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"auc={np.mean(aucs):.3f}")
    print(f"        entropy-порог  acc={np.mean(acc_ent):.3f}±{np.std(acc_ent):.3f}")
    print(f"        logprob-порог  acc={np.mean(acc_lp):.3f}±{np.std(acc_lp):.3f}")

    # полная модель на всех данных — сохранить
    mu, sd = X_raw.mean(0), X_raw.std(0)
    X = zscore(X_raw, mu, sd).to(DEV)
    model = Clf().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    y_c = y.to(DEV)
    for ep in range(300):
        perm = torch.randperm(n, device=DEV)
        model.train()
        for b in range(0, n, 16):
            bi = perm[b:b+16]
            opt.zero_grad()
            loss = lossf(model(X[bi]), y_c[bi])
            loss.backward(); opt.step()
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, "clf_knowledge.pt")
    print("Сохранено: clf_knowledge.pt")

if __name__ == "__main__":
    main()
