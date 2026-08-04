# Эксперимент 2b: СОСЕДНЯЯ АССОЦИАЦИЯ (k_t → v_{t+1}).
# exp2a: пары (k_t → v_t) — токен→себя; чтение по вопросу даёт ~0 (вопрос не записывался).
# exp2b: каждый токен запоминает ЗНАЧЕНИЕ СЛЕДУЮЩЕГО: «Пароль:» → секрет.
# Пары строятся из контекста (на инференсе тоже!) — связь маркер→секрет кодируется
# напрямую, чтение по вопросу (семантически близкому к маркеру) извлекает её.
# Всё остальное как exp2a (проекции k/v/q, прямое чтение, bilevel-запись).
# Исправления против нашей схемы (v5/exp1*):
#  1. Запись АССОЦИАТИВНАЯ: лосс записи = ‖M(k) − v‖² (eq.12 статьи / neural_memory.py:402),
#     где k = проекция токена, v = проекция значения — НЕ самовосстановление ‖M(h)−h‖².
#  2. Обучаемые проекции: k=GELU(W_k(h)), v=GELU(W_v(h)), q=GELU(W_q(q_hidden))
#     (как to_keys/to_values/to_queries, neural_memory.py:420-428).
#  3. Чтение ПРЯМОЕ: mix6 = g·W_out(M(q; записанные веса)) — без «разности откликов».
#  4. Одна память (без слотов) — как в Titans; шаги записи — нормированные, η/γ обучаемые
#     (exp1c доказал: ассоциативная запись + bilevel-градиенты работают).
# Метрика: val_acc6 (6 кандидатов) vs шанс 0.17; ‖mix6−target6‖; персистентность:
#   запись по контексту (сессия 1) → чтение по вопросу (сессия 2) — как в eval_cross_session.

import torch, torch.nn as nn, torch.nn.functional as F, random
from train_memory_distill import (
    last_layer_forward, cand_logits, SECRET_TO_IDX, DEV,
)

D = 896
CHUNK = 16
MAX_CTX_S = 200
N_TRAIN = 64
LR = 3e-3
ALIGN_ITERS = 300
PHASE2_EPOCHS = 15
ITERS = 4          # шаги записи на чанк


class MemoryMLP(nn.Module):
    """одна память: Linear→GELU→Linear (как MemoryMLP в Titans, dim_head=896)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, w1, w2):
        return F.gelu(x @ w1.t()) @ w2.t()


class TitansCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = MemoryMLP()
        self.W_k = nn.Linear(D, D, bias=False)   # to_keys
        self.W_v = nn.Linear(D, D, bias=False)   # to_values
        self.W_q = nn.Linear(D, D, bias=False)   # to_queries
        self.W_out = nn.Linear(D, 6)             # сдвиг логпробов кандидатов
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.eta_raw = nn.Parameter(torch.tensor(-1.0))  # η = softplus ≈ 0.31
        self.gam_raw = nn.Parameter(torch.tensor(-2.0))  # γ = sigmoid ≈ 0.12

    def g(self):
        return torch.sigmoid(self.g_logit)

    def eta(self):
        return F.softplus(self.eta_raw)

    def gam(self):
        return torch.sigmoid(self.gam_raw)

    def write(self, h_ctx):
        """АССОЦИАТИВНАЯ запись: M(k) ≈ v, k=W_k(h), v=W_v(h); арифметика без detach
        (bilevel), шаги нормированные; η/γ обучаемые."""
        eta, gam = self.eta(), self.gam()
        k = F.gelu(self.W_k(h_ctx[:-1]))          # ключ: токен t
        v = F.gelu(self.W_v(h_ctx[1:]))           # значение: токен t+1 (СОСЕДНЯЯ ассоциация)
        W1, W2 = self.mem.net[0].weight, self.mem.net[2].weight
        for c in range(0, len(k), CHUNK):
            kk, vv = k[c:c + CHUNK], v[c:c + CHUNK]
            for _ in range(ITERS):
                pred = self.mem.forward(kk, W1, W2)
                iloss = F.mse_loss(pred, vv)          # |M(k) − v|²  — eq.12 Titans
                g1, g2 = torch.autograd.grad(iloss, [W1, W2], create_graph=True)
                g1 = g1 / (g1.norm() + 1e-8)
                g2 = g2 / (g2.norm() + 1e-8)
                W1 = W1 * (1 - gam) - eta * g1
                W2 = W2 * (1 - gam) - eta * g2
        return W1, W2

    def read(self, q_hidden, W1, W2):
        """ПРЯМОЕ чтение: mix6 = g·W_out(M(W_q(q))) — без разности откликов"""
        q = F.gelu(self.W_q(q_hidden))
        out = self.mem.forward(q, W1, W2)
        return self.g() * self.W_out(out)


def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = TitansCorrect().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # ---- пре-обучение проекций: k/v/q должны быть выровнены с hidden Qwen ----
    # (легкая дистилляция: W_k/W_v/W_q ≈ identity на hidden)
    print("пре-обучение проекций (≈identity на hidden Qwen) ...", flush=True)
    for st in range(ALIGN_ITERS):
        opt.zero_grad()
        d = train[st % len(train)]
        h = d["ctx_hidden"].to(DEV)[:16]
        loss = (lossf(F.gelu(model.W_k(h)), h) + lossf(F.gelu(model.W_v(h)), h)) / 2
        loss.backward()
        opt.step()
        if st % 100 == 99:
            print(f"  proj {st}: {loss.item():.4f}", flush=True)

    # ---- фаза: обучение памяти (запись+чтение) на парах контекст→вопрос ----
    print("обучение памяти (ассоциативная запись + прямое чтение) ...", flush=True)
    for ep in range(PHASE2_EPOCHS):
        model.train()
        tot = 0.0
        for i, d in enumerate(train):
            opt.zero_grad()
            ctx = d["ctx_hidden"].to(DEV)
            q_h = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = torch.tensor(SECRET_TO_IDX[d["secret"]], device=DEV)
            W1, W2 = model.write(ctx)
            mv = model.read(q_h, W1, W2)
            base_cand = cand_logits(last_layer_forward(h_all)).detach()
            target6 = (cand_logits(y_out) - base_cand).float()
            final_cand = base_cand + mv.to(base_cand.dtype)
            loss = lossf(mv.to(target6.dtype), target6) + 0.5 * cef(final_cand, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"эпоха {ep}: train {tot / len(train):.4f}", flush=True)

        model.eval()
        vacc, vd, nv = 0, 0.0, 0
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q_h = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            W1, W2 = model.write(ctx)          # ВНЕ no_grad (bilevel-граф)
            with torch.no_grad():
                mv = model.read(q_h, W1, W2)
                base_cand = cand_logits(last_layer_forward(h_all))
                target6 = (cand_logits(y_out) - base_cand).float()
                final_cand = base_cand + mv.to(base_cand.dtype)
                vacc += final_cand.argmax(-1).item() == tgt
                vd += (mv.to(target6.dtype) - target6).norm().item()
                nv += 1
        print(f"  val: acc6 = {vacc / nv:.2f} | ‖mix6−target6‖ = {vd / nv:.2f} | "
              f"η = {model.eta().item():.3f} | γ = {model.gam().item():.3f} | "
              f"g = {model.g().item():.3f}", flush=True)

    torch.save(model.state_dict(), "exp2b_next_assoc.pt")
    print("Сохранено: exp2b_next_assoc.pt", flush=True)


if __name__ == "__main__":
    main()
