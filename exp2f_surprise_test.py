# Эксперимент 2f: механизм УДИВЛЕНИЯ (Titans) на Qwen + наш MLP — тест различения.
# Механизм Titans (neural_memory.py): surprise_t = ∇_θ ‖M(k_t)−v_t‖² (per-sample градиент,
# vmap), запись модулируется η_t (обучаемым). ГИПОТЕЗА для нас: ВЫРОВНЕННАЯ память
# (M(h)≈h на типичных hidden Qwen) должна «удивляться» аномальным токенам (секрет
# «X7K9Q2») сильнее, чем типичному мусору — тогда η-гейт работает БЕЗ учителя.
# Метрика: ROC-AUC различения секрет/мусор по маске mask_secret (датасет знает).
# Варианты удивления:
#   (a) loss_t = ‖M(h_t) − h_t‖²              — самовосстановление (дёшево)
#   (b) grad_t = ‖∇_θ ‖M(h_t)−h_t‖²‖          — Titans-стиль (per-sample градиент)
#   (c) loss_next = ‖M(h_t) − h_{t+1}‖²       — предсказание следующего токена

import torch, torch.nn as nn, torch.nn.functional as F, random
from sklearn.metrics import roc_auc_score

D = 896
N_TRAIN = 320
BATCH_TOK = 16
LR = 3e-3
ALIGN_STEPS = 2400


class MemMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < 200]
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = len(ex) // 10
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]
    Xtr = torch.cat([d["ctx_hidden"] for d in train]).to("cuda")
    print(f"train {len(Xtr)} токенов | val {len(val)} примеров", flush=True)

    torch.manual_seed(0)
    M = MemMLP().to("cuda")
    opt = torch.optim.Adam(M.parameters(), lr=LR)
    lossf = nn.MSELoss()
    for st in range(ALIGN_STEPS):
        opt.zero_grad()
        idx = torch.randint(0, len(Xtr), (BATCH_TOK,))
        x = Xtr[idx]
        loss = lossf(M(x), x)
        loss.backward()
        opt.step()
        if st % 800 == 799:
            print(f"  align {st}: {loss.item():.4f}", flush=True)
    print("выравнивание готово (M(h)≈h)", flush=True)

    # ---- surprise по val ----
    aucs = {"loss_self": [], "grad_self": [], "loss_next": []}
    n_secret = n_noise = 0
    for d in val:
        ctx = d["ctx_hidden"].to("cuda")
        mask = d["mask_secret"]                 # 1=секрет, 0=мусор (в токенах 0..n-1)
        with torch.no_grad():
            Mo = M(ctx)
            loss_self = ((Mo - ctx) ** 2).mean(-1)          # [n]
            loss_next = ((M(ctx[:-1]) - ctx[1:]) ** 2).mean(-1)   # [n-1]
        # per-sample градиент (Titans-стиль): для каждого токена отдельно — ВНЕ no_grad
        grads = []
        for t in range(len(ctx)):
            x = ctx[t].unsqueeze(0)
            l = lossf(M(x), x)
            g = torch.autograd.grad(l, list(M.parameters()))
            grads.append(sum(gi.norm() for gi in g))
        grad_self = torch.stack(grads)
        m = mask[:len(ctx)]
        for name, sc in (("loss_self", loss_self), ("grad_self", grad_self), ("loss_next", loss_next)):
            mm = m[:len(sc)]
            if (mm == 1).sum() > 0 and (mm == 0).sum() > 0:
                aucs[name].append(roc_auc_score(mm.cpu().numpy(), sc.cpu().numpy()))
        n_secret += (m == 1).sum().item(); n_noise += (m == 0).sum().item()

    print(f"токены: секрет {n_secret} | мусор {n_noise}", flush=True)
    for name, a in aucs.items():
        print(f"  {name}: AUC = {sum(a) / len(a):.4f} (по {len(a)} примерам)", flush=True)

    # наглядность: среднее удивление по классам (loss_self)
    ls_s, ls_n = [], []
    for d in val:
        ctx = d["ctx_hidden"].to("cuda")
        mask = d["mask_secret"]
        with torch.no_grad():
            loss_self = ((M(ctx) - ctx) ** 2).mean(-1)
        m = mask[:len(ctx)]
        ls_s.append(loss_self[m == 1].mean().item())
        ls_n.append(loss_self[m == 0].mean().item())
    print(f"  loss_self: секрет {sum(ls_s)/len(ls_s):.4f} | мусор {sum(ls_n)/len(ls_n):.4f} "
          f"| отношение {sum(ls_s)/len(ls_s) / (sum(ls_n)/len(ls_n) + 1e-9):.2f}x", flush=True)


if __name__ == "__main__":
    main()
