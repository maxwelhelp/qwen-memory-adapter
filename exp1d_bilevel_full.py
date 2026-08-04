# Эксперимент 1d (PLAN_MEMORY_V3, Ф1-вариант C, полный): ЗАПИСЬ ОБУЧАЕТСЯ ПОД ЧТЕНИЕ.
# Мотивация (cos-диагностика exp1b2): необучаемая v5-запись НЕ кодирует направление
# секрета — cos(read_diff, target_in) = 0.014 (шум). Векторная инъекция невозможна,
# пока слоты не вырабатывают нужный вектор.
# Bilevel: внешний лосс течёт ЧЕРЕЗ шаги записи (арифметика без detach, create_graph=True)
# в θ₀, η, γ — слоты учатся писать ТАК, чтобы чтение по q давало target_in.
# Схема (диагностическая, урезанная):
#   alignment (как v5, короткий) → bilevel-фаза:
#   запись: нормированные шаги самовосстановления ||M(h)−h||², η/γ ОБУЧАЕМЫЕ
#   чтение: read_diff(q, work) → W_out → mix (896); лосс = MSE(mix, target_in) + 0.5·CE(6)
#   backward через ВСЕ шаги записи; обучаются: θ₀ слотов, W_route, W_out, g, η, γ
# Критерий: cos(read_diff, target_in) на val РАСТЁТ с 0.014 (шум) — слоты кодируют.

import torch, torch.nn as nn, torch.nn.functional as F, random
from torch.func import functional_call
from train_memory_distill import (
    last_layer_forward, cand_logits, SECRET_TO_IDX, DEV, MAX_CTX,
)

D = 896
N_SLOTS = 4
ITERS = 2
CHUNK = 16
MAX_CTX_S = 150
N_TRAIN = 32
LR = 3e-3
ALIGN_ITERS = 200
PHASE2_EPOCHS = 8


class SlotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


class MemoryBilevel(nn.Module):
    """Полный bilevel: запись — арифметика без detach, все параметры обучаемы внешним лоссом."""

    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([SlotMLP() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_out = nn.Linear(D, D)                 # векторная выдача (как exp1b2)
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.eta_raw = nn.Parameter(torch.tensor(-1.0))   # η = softplus ≈ 0.31
        self.gam_raw = nn.Parameter(torch.tensor(-2.0))   # γ = sigmoid ≈ 0.12

    def g(self):
        return torch.sigmoid(self.g_logit)

    def eta(self):
        return F.softplus(self.eta_raw)

    def gam(self):
        return torch.sigmoid(self.gam_raw)

    def read_theta(self, q):
        """отклик живых θ₀ (для alignment; градиенты в θ₀ идут)"""
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1 = torch.stack([self.slots[i].net[0].weight.t() for i in range(N_SLOTS)])
        W2 = torch.stack([self.slots[i].net[2].weight.t() for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        h = F.gelu(torch.einsum('bd,bdh->bh', qb, W1))
        out = torch.einsum('bh,bhd->bd', h, W2)
        return (out * w.unsqueeze(1)).sum(0)

    def write_bilevel(self, h_ctx):
        """запись: арифметика БЕЗ detach — W1/W2 становятся графовыми узлами,
        граф идёт через η_raw/gam_raw и θ₀ (create_graph=True). Шаги нормированные."""
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
        """разность откликов (запись − θ₀) с графовыми весами; θ₀ — живые (градиенты идут)"""
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

    def mix(self, q, work):
        return self.g() * self.W_out(self.read_diff(q, work))


def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = MemoryBilevel().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # ---- ALIGNMENT (короткий): слоты ≈ identity на hidden Qwen ----
    print("alignment ...", flush=True)
    for st in range(ALIGN_ITERS):
        opt.zero_grad()
        d = train[st % len(train)]
        ctx = d["ctx_hidden"].to(DEV)
        h = ctx[torch.randperm(len(ctx))[:8]]
        loss = lossf(model.read_theta(h.mean(0)), h.mean(0))
        loss.backward()
        opt.step()
        if st % 50 == 49:
            print(f"  align {st}: {loss.item():.4f}", flush=True)

    # ---- ФАЗА 2: bilevel — внешний лосс через запись ----
    print("bilevel-фаза (запись обучается под чтение) ...", flush=True)
    for ep in range(PHASE2_EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for i, d in enumerate(train):
            opt.zero_grad()
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            tgt = torch.tensor(SECRET_TO_IDX[d["secret"]], device=DEV)
            tin = d["target_in"].to(DEV)
            work = model.write_bilevel(ctx)
            mv = model.mix(q, work)
            h_inj = d["h_inA_all"].to(DEV).clone()
            h_inj[-1] = h_inj[-1] + mv
            h_out_p = last_layer_forward(h_inj)
            loss = (lossf(mv, tin) + 0.5 * cef(cand_logits(h_out_p), tgt)
                    + 0.1 * lossf(h_out_p, d["h_outB"].to(DEV)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f"эпоха {ep}: train {tot / n:.4f}", flush=True)

        # ---- val: cos(read_diff, target_in) — КЛЮЧЕВАЯ метрика ----
        model.eval()
        c6, cs, nd, nm = 0, 0.0, 0, 0.0
        with torch.no_grad():
            for d in val:
                ctx = d["ctx_hidden"].to(DEV)
                q = d["q_hidden"].to(DEV)
                tgt = SECRET_TO_IDX[d["secret"]]
                work = model.write_bilevel(ctx)
                rd = model.read_diff(q, work)
                tin = d["target_in"].to(DEV)
                cs += F.cosine_similarity(rd, tin, dim=0).item()
                mv = model.mix(q, work)
                h_inj = d["h_inA_all"].to(DEV).clone()
                h_inj[-1] = h_inj[-1] + mv
                h_out_p = last_layer_forward(h_inj)
                c6 += cand_logits(h_out_p).argmax(-1).item() == tgt
                nd += (mv - tin).norm().item()
                nm += mv.norm().item()
        n = len(val)
        print(f"  val: cos(read_diff, target_in) = {cs / n:.4f} | acc6 = {c6 / n:.2f} | "
              f"‖mix−Δ_in‖ = {nd / n:.1f} | ‖mix‖ = {nm / n:.1f} | "
              f"η = {model.eta().item():.3f} | γ = {model.gam().item():.3f}", flush=True)
        print(f"  (эталон: cos=0.014 шум → растёт если слоты кодируют; oracle acc6=1.00)", flush=True)

    torch.save(model.state_dict(), "exp1d_bilevel.pt")
    print("Сохранено: exp1d_bilevel.pt", flush=True)


if __name__ == "__main__":
    main()
