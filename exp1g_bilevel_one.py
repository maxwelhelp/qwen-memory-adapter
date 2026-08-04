# Эксперимент 1g: bilevel с 6-ЧИСЛОВОЙ головой, N_SLOTS=1 (проверка гипотезы маршрутизации).
# exp1d/1e: 8 слотов — запись (случайные route_keys, argmax) и чтение (обучаемый W_route)
# разведены по слотам → сигнал теряется. 1 слот = запись и чтение в одном месте
# (как в exp1c, где bilevel работал). Если acc6 растёт — чинить маршрутизацию;
# если нет — чинить поток внешнего градиента в записи.
# Мотивация: exp1d (bilevel→896-вектор target_in) не закодировал направление (cos шум).
# 6-числовая голова (target6 = сдвиг логпробов кандидатов с/без контекста) предъявляет
# МЕНЬШЕ требований к направлению → слоты должны легче закодировать то, что нужно чтению.
# Запись: арифметика без detach (как exp1c/1d), внешний лосс течёт в θ₀/η/γ/W_out/W_route.
# Усиления vs exp1d: ITERS=4, N_TRAIN=64, 15 эпох.
# Критерий: val_acc (6 кандидатов) растёт выше шанса + ‖mix6−target6‖ падает.

import torch, torch.nn as nn, torch.nn.functional as F, random
from train_memory_distill import (
    last_layer_forward, cand_logits, SECRET_TO_IDX, SEC_TOK_VALUES, DEV,
)

D = 896
N_SLOTS = 1
ITERS = 4
CHUNK = 16
MAX_CTX_S = 200
N_TRAIN = 64
LR = 3e-3
ALIGN_ITERS = 300
PHASE2_EPOCHS = 15


class SlotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


class MemorySix(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([SlotMLP() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_out = nn.Linear(D, 6)                     # 6 чисел — сдвиг логпробов кандидатов
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.eta_raw = nn.Parameter(torch.tensor(-1.0))  # η = softplus ≈ 0.31
        self.gam_raw = nn.Parameter(torch.tensor(-2.0))  # γ = sigmoid ≈ 0.12

    def g(self):
        return torch.sigmoid(self.g_logit)

    def eta(self):
        return F.softplus(self.eta_raw)

    def gam(self):
        return torch.sigmoid(self.gam_raw)

    def read_theta(self, q):
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1 = torch.stack([self.slots[i].net[0].weight.t() for i in range(N_SLOTS)])
        W2 = torch.stack([self.slots[i].net[2].weight.t() for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        h = F.gelu(torch.einsum('bd,bdh->bh', qb, W1))
        out = torch.einsum('bh,bhd->bd', h, W2)
        return (out * w.unsqueeze(1)).sum(0)

    def write_bilevel(self, h_ctx):
        eta, gam = self.eta(), self.gam()
        work = {}
        for c in range(0, len(h_ctx), CHUNK):
            groups = {}
            for j, h in enumerate(h_ctx[c:c + CHUNK]):
                s = (h @ self.route_keys.t()).argmax().item()
                groups.setdefault(s, []).append((j, h))
            for s, items in groups.items():
                kk = torch.stack([h for _, h in items])
                if s in work:
                    W1, W2 = work[s]
                else:
                    W1, W2 = self.slots[s].net[0].weight, self.slots[s].net[2].weight
                for _ in range(ITERS):
                    pred = F.gelu(kk @ W1.t()) @ W2.t()
                    iloss = F.mse_loss(pred, kk)
                    g1, g2 = torch.autograd.grad(iloss, [W1, W2], create_graph=True)
                    g1 = g1 / (g1.norm() + 1e-8)
                    g2 = g2 / (g2.norm() + 1e-8)
                    W1 = W1 * (1 - gam) - eta * g1
                    W2 = W2 * (1 - gam) - eta * g2
                work[s] = (W1, W2)
        return work

    def read_diff(self, q, work):
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1w = torch.stack([work[i][0] if i in work else self.slots[i].net[0].weight
                           for i in range(N_SLOTS)])
        W2w = torch.stack([work[i][1] if i in work else self.slots[i].net[2].weight
                           for i in range(N_SLOTS)])
        W1t = torch.stack([self.slots[i].net[0].weight.t() for i in range(N_SLOTS)])
        W2t = torch.stack([self.slots[i].net[2].weight.t() for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        hw = F.gelu(torch.einsum('bd,bdh->bh', qb, W1w))
        outw = torch.einsum('bh,bhd->bd', hw, W2w)
        ht = F.gelu(torch.einsum('bd,bdh->bh', qb, W1t))
        outt = torch.einsum('bh,bhd->bd', ht, W2t)
        return ((outw - outt) * w.unsqueeze(1)).sum(0)

    def mix6(self, q, work):
        return self.g() * self.W_out(self.read_diff(q, work))


def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = MemorySix().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    print("alignment ...", flush=True)
    for st in range(ALIGN_ITERS):
        opt.zero_grad()
        d = train[st % len(train)]
        ctx = d["ctx_hidden"].to(DEV)
        h = ctx[torch.randperm(len(ctx))[:8]]
        loss = lossf(model.read_theta(h.mean(0)), h.mean(0))
        loss.backward()
        opt.step()
        if st % 100 == 99:
            print(f"  align {st}: {loss.item():.4f}", flush=True)

    print("bilevel-фаза (6-числовая голова, запись обучаемая) ...", flush=True)
    for ep in range(PHASE2_EPOCHS):
        model.train()
        tot = 0.0
        for i, d in enumerate(train):
            opt.zero_grad()
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = torch.tensor(SECRET_TO_IDX[d["secret"]], device=DEV)
            work = model.write_bilevel(ctx)
            mv = model.mix6(q, work)
            base_cand = cand_logits(last_layer_forward(h_all)).detach()
            target6 = (cand_logits(y_out) - base_cand).float()
            final_cand = base_cand + mv.to(base_cand.dtype)
            loss = (lossf(mv.to(target6.dtype), target6)
                    + 0.5 * cef(final_cand, tgt))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"эпоха {ep}: train {tot / len(train):.4f}", flush=True)

        model.eval()
        vacc, vd, nv = 0, 0.0, 0
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_bilevel(ctx)          # ВНЕ no_grad (билевел-граф)
            with torch.no_grad():
                mv = model.mix6(q, work)
                base_cand = cand_logits(last_layer_forward(h_all))
                target6 = (cand_logits(y_out) - base_cand).float()
                final_cand = base_cand + mv.to(base_cand.dtype)
                vacc += final_cand.argmax(-1).item() == tgt
                vd += (mv.to(target6.dtype) - target6).norm().item()
                nv += 1
        print(f"  val: acc6 = {vacc / nv:.2f} | ‖mix6−target6‖ = {vd / nv:.2f} | "
              f"η = {model.eta().item():.3f} | γ = {model.gam().item():.3f} | "
              f"g = {model.g().item():.3f}", flush=True)

    torch.save(model.state_dict(), "exp1g_bilevel_one.pt")
    print("Сохранено: exp1g_bilevel_one.pt", flush=True)


if __name__ == "__main__":
    main()
