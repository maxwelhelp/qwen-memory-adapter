# Шаг 3 (memory_layer_v2.md): дистилляция памяти — ПОДМЕШИВАНИЕ К ВХОДУ ПОСЛЕДНЕГО СЛОЯ.
# Память пишет контекст чанками (bilevel), читает по вопросу, подмешивает
# ТОЛЬКО разность «запись − θ₀» к ВХОДУ последнего слоя на последней позиции:
#   h'_in[-1] = h_in[-1] + g·W_out(M(q) − M_θ₀(q))
#   h_out' = last_layer(h'_in)[-1]   → loss = MSE(h_out', h_outB) + λ·CE(кандидаты)
# Контроль ТОЧНЫЙ: h_in + target_in = h_in_ctx → последний слой → h_out_ctx.
# Δ входа имеет норму ~5-10 (не 150 как Δ выхода) — реалистично для памяти.

import torch, torch.nn as nn, torch.nn.functional as F, random
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SECRETS = ["X7K9Q2", "M4N8Z1", "Q2P5R7", "T8W3V5", "R9L2X4", "Z6F3H8"]
tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
SEC_TOK_VALUES = [tok.encode(p, add_special_tokens=False)[0] for p in SECRETS]
SECRET_TO_IDX = {p: i for i, p in enumerate(SECRETS)}

D = 896
N_SLOTS = 32
H_SLOT = 896        # ширина скрытого слоя MLP-слота (полный тюнинг)
ITERS = 4           # итерации записи на чанк (нелинейности нужно >1 шага)
ETA = 0.3
GAMMA = 0.01
TOPK = 4
CHUNK = 16
MAX_CTX = 200
N_TRAIN = 320
BATCH = 4
EPOCHS = 40
LR = 3e-3
NORM_STEP = True

class SlotMLP(nn.Module):
    """MLP-слот в стиле Titans MemoryMLP: Linear(без bias)→GELU→Linear(без bias),
    xavier_uniform. БЕЗ residual-обёртки: mix = разность откликов (запись − θ₀),
    а residual (MLP(x)+x) сводил запись к нулю (||MLP(h)||² → 0) и обнулял mix."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, H_SLOT, bias=False),
                                 nn.GELU(),
                                 nn.Linear(H_SLOT, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)

class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([SlotMLP() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)  # ОБУЧАЕМАЯ маршрутизация
        self.W_out = nn.Linear(D, len(SECRETS))   # 6 чисел = СДВИГ ЛОГПРОБОВ кандидатов
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.W_eta = nn.Linear(D, 1)              # η-голова: КАК СИЛЬНО писать токен
        self.eta_min, self.eta_max = 0.1, 1.0     # мусор пишется слабо, секрет сильно

    def g(self):
        return torch.sigmoid(self.g_logit)

    def write_work(self, h_ctx):
        """запись чанками, ITERS итераций на чанк (нелинейности нужно >1 шага).
        Пары (k=h, v=h) — самовосстановление M(h)≈h; автономные клоны параметров
        (запись — необучаемая механика), шаги нормированные (веса не раздуваются).
        η-ГОЛОВА: η_t = σ(W_eta(h_t)) — «как сильно писать токен» (учитель:
        секрет=1, мусор=0); шаг токена масштабируется η_t (novelty гейтит θ_t)."""
        n = len(h_ctx)
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * torch.sigmoid(
            self.W_eta(h_ctx)).squeeze(-1)      # [n] — сила записи per-token
        work = {}
        for _ in range(ITERS):
            for c in range(0, n, CHUNK):
                groups = {}
                for j, h in enumerate(h_ctx[c:c+CHUNK]):
                    s = (h @ self.route_keys.t()).argmax().item()
                    groups.setdefault(s, []).append((j, h))
                    if s not in work:
                        work[s] = {k: v.detach().clone().requires_grad_(True)
                                   for k, v in self.slots[s].named_parameters()}
                for s, items in groups.items():
                    params = work[s]
                    jj = [j for j, _ in items]
                    hs = [h for _, h in items]
                    kk = torch.stack(hs)
                    pred = functional_call(self.slots[s], params, (kk,))
                    wts = eta_t[torch.tensor(jj, device=kk.device)]   # η по токенам группы
                    iloss = (wts * ((pred - kk) ** 2).mean(-1)).mean()  # M(h) ≈ h, взвешено η
                    gs = torch.autograd.grad(iloss, list(params.values()), create_graph=False)
                    for (name, p), g in zip(params.items(), gs):
                        if NORM_STEP:
                            g = g / (g.norm() + 1e-8)
                        params[name] = p * (1 - GAMMA) - ETA * g
        return work

    def refresh_theta_stacks(self):
        """стек θ₀-матриц всех слотов [32, 896, H] / [32, H, 896] — для батч-чтения.
        Вызывать после alignment (θ₀ меняются)."""
        with torch.no_grad():
            # weight — [out, in]; для einsum 'bd,bdh->bh' нужен [in, out] = .t()
            self._W1_theta = torch.stack([self.slots[i].net[0].weight.t() for i in range(N_SLOTS)])
            self._W2_theta = torch.stack([self.slots[i].net[2].weight.t() for i in range(N_SLOTS)])
        return self

    def _read_batch(self, q, work):
        """выходы ВСЕХ слотов батчем (einsum вместо цикла functional_call)"""
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1 = torch.stack([work[i]["net.0.weight"].t() if i in work else self._W1_theta[i]
                          for i in range(N_SLOTS)])
        W2 = torch.stack([work[i]["net.2.weight"].t() if i in work else self._W2_theta[i]
                          for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        h = F.gelu(torch.einsum('bd,bdh->bh', qb, W1))
        out = torch.einsum('bh,bhd->bd', h, W2)
        return out, w

    def read_theta(self, q):
        """отклик θ₀ (без записи): Σ w_i·slot_i(q) — для alignment-фазы.
        Стек строится ИЗ ЖИВЫХ θ₀ каждый вызов (без кэша/detach!) —
        в фазе 1 θ₀ обучаются через read_theta, градиенты должны идти."""
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1 = torch.stack([self.slots[i].net[0].weight.t() for i in range(N_SLOTS)])
        W2 = torch.stack([self.slots[i].net[2].weight.t() for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        h = F.gelu(torch.einsum('bd,bdh->bh', qb, W1))
        out = torch.einsum('bh,bhd->bd', h, W2)
        return (out * w.unsqueeze(1)).sum(0)

    def read_diff(self, q, work):
        """разность «рабочая память − θ₀» по всем слотам (что изменила запись)"""
        outw, w = self._read_batch(q, work)
        outt, _ = self._read_batch(q, {})
        return ((outw - outt) * w.unsqueeze(1)).sum(0)

    def mix(self, q, work):
        """[6] сдвигов логпробов кандидатов (память «голосует» за кандидатов)"""
        return self.g() * self.W_out(self.read_diff(q, work))

def last_layer_forward(h_all):
    """прогон последнего слоя без подмешивания (для базовых логпробов кандидатов)"""
    hh = h_all.unsqueeze(0).to(qwen.lm_head.weight.dtype)
    T = h_all.shape[0]
    pos = torch.arange(T, device=DEV).unsqueeze(0)
    cos, sin = qwen.model.rotary_emb(hh, pos)
    out = qwen.model.layers[-1](hh, position_embeddings=(cos, sin))   # [1, T, D]
    return out[0, -1].float()

def cand_logits(h_out):
    """[6] логпробов кандидатов из выхода последнего слоя"""
    logits = qwen.lm_head(h_out.unsqueeze(0).to(qwen.lm_head.weight.dtype))   # [1, V]
    return torch.stack([logits[0, t] for t in SEC_TOK_VALUES])

def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX]
    print(f"примеров с контекстом < {MAX_CTX}: {len(ex)} (из {len(data)})", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = Memory().to(DEV)
    model.refresh_theta_stacks()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    ynorm = torch.cat([d["target_in"] for d in val]).pow(2).mean().item()
    print(f"baseline ||Δ_in||² = {ynorm:.5f}  (по норме {ynorm ** 0.5:.4f})", flush=True)

    # ---- ФАЗА 1: alignment — слоты учатся ВОСПРОИЗВОДИТЬ пространство Qwen ----
    print("Фаза 1: alignment (M(h) ≈ h на скрытых состояниях)", flush=True)
    ALIGN_EPOCHS = 15
    for ep in range(ALIGN_EPOCHS):
        model.train()
        opt.zero_grad()
        tot, n = 0.0, 0
        for i, d in enumerate(train):
            ctx = d["ctx_hidden"].to(DEV)
            hs = ctx[torch.randperm(len(ctx))[:16]]
            loss = torch.tensor(0.0, device=DEV)
            for h in hs:
                loss = loss + lossf(model.read_theta(h), h)
            loss.backward()
            tot += loss.item(); n += 1
            if (i + 1) % BATCH == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        if n % BATCH:
            opt.step(); opt.zero_grad()
        vloss = 0.0
        model.eval()
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            with torch.no_grad():
                hs = ctx[:16]
                for h in hs:
                    vloss += lossf(model.read_theta(h), h).item()
        vloss /= (len(val) * 16)
        print(f"align {ep:2d}: train {tot / n:.4f} | val {vloss:.4f}", flush=True)

    # ---- ФАЗА 1.5: η-ГОЛОВА — «что записывать» (учитель: секрет=1, мусор=0) ----
    print("Фаза 1.5: η-голова (что писать)", flush=True)
    bcef = nn.BCEWithLogitsLoss()
    for ep in range(10):
        model.train()
        opt.zero_grad()
        tot, n = 0.0, 0
        for i, d in enumerate(train):
            ctx = d["ctx_hidden"].to(DEV)
            mask = d["mask_secret"].to(DEV)
            loss = bcef(model.W_eta(ctx).squeeze(-1), mask)
            loss.backward()
            tot += loss.item(); n += 1
            if (i + 1) % BATCH == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        if n % BATCH:
            opt.step(); opt.zero_grad()
        if ep in (0, 4, 9):
            with torch.no_grad():
                eta_m = model.eta_min + (model.eta_max - model.eta_min) * torch.sigmoid(
                    model.W_eta(train[0]["ctx_hidden"].to(DEV))).squeeze(-1)
                m_m = train[0]["mask_secret"]
                print(f"  eta-эпоха {ep}: loss {tot / n:.3f} | "
                      f"eta[секрет]≈{eta_m[m_m > 0].mean().item():.2f} "
                      f"eta[мусор]≈{eta_m[m_m == 0].mean().item():.2f}", flush=True)

    # ---- ЗАМОРОЗКА СЛОТОВ: запись — необучаемая механика, чтение — обучаемое.
    #      (градиенты через билевел-запись не доходят — клоны обрывают граф;
    #       обучаем только W_out, W_route, g, W_eta — прямая связь CE → извлечение) ----
    for p in model.slots.parameters():
        p.requires_grad = False
    model.refresh_theta_stacks()   # θ₀ изменились в фазе 1 — обновить кэш
    print("Слоты заморожены: обучаются только W_out, W_route, g", flush=True)

    # ---- ФАЗА 2: дистилляция ответов (память на выровненном пространстве) ----
    print("Фаза 2: дистилляция", flush=True)
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        tot, n = 0.0, 0
        for i, d in enumerate(train):
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_work(ctx)
            mix = model.mix(q, work)
            base_cand = cand_logits(last_layer_forward(h_all)).detach()   # без контекста
            target6 = (cand_logits(y_out) - base_cand).float()           # целевой сдвиг
            final_cand = base_cand + mix.to(base_cand.dtype)
            loss_ce = cef(final_cand, torch.tensor(tgt, device=DEV))
            loss_mse = lossf(mix, target6)
            loss = loss_ce + 0.5 * loss_mse
            loss.backward()
            tot += loss.item(); n += 1
            if (i + 1) % BATCH == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        if n % BATCH:
            opt.step(); opt.zero_grad()

        model.eval()
        vloss, vacc = 0.0, 0
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_work(ctx)
            mix = model.mix(q, work)
            base_cand = cand_logits(last_layer_forward(h_all)).detach()
            target6 = (cand_logits(y_out) - base_cand).float()
            final_cand = base_cand + mix.to(base_cand.dtype)
            vloss += lossf(mix, target6).item()
            vacc += final_cand.argmax(-1).item() == tgt
        vloss /= len(val); vacc /= len(val)
        print(f"эпоха {ep:2d}: train {tot / n:.4f} | val_mse {vloss:.4f} "
              f"| val_acc {vacc:.2f} | g={model.g().item():.3f}", flush=True)

    torch.save(model.state_dict(), "memory_distill.pt")
    print("Сохранено: memory_distill.pt")

if __name__ == "__main__":
    main()
