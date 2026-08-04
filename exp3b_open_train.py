# Эксперимент 3b: ВЕКТОРНАЯ ИНЪЕКЦИЯ, ОБУЧЕНИЕ НА ОТКРЫТОМ ДАТАСЕТЕ.
# exp3a (обучен на 6 кандидатах): open-тест 0/40 — W_out извлекает только знакомые
# секреты. Здесь: датасет со СЛУЧАЙНЫМИ секретами (dataset_yattn_open_train.pt),
# лосс = MSE(h_out', h_outB) + 0.5·CE(lm_head(h_out_p), ПЕРВЫЙ ТОКЕН СЕКРЕТА по всему
# словарю) — память учится извлекать произвольные ответы.
# Зачем: 6 чисел = голосование за закрытый список кандидатов; вектор во вход
# последнего слоя = Qwen сама доводит до логпробов ВСЕГО словаря (открытый словарь).
# Oracle показал: последний слой пропускает правильный вектор (acc6=1.00, усиление
# ~7.7x). Раньше (exp1b) не сходилось из-за раздутого mix — но то было ДО правильной
# структуры (линейные слоты+delta-rule+η). Здесь: W_out 896→896, mix_vec во h_in[-1],
# лосс = MSE(h_out', h_outB) + 0.5·CE(кандидаты). Метрики: val_mse, acc6, acc_open.
# exp2m: энтропия предсказания Qwen различает секрет/мусор (AUC 0.85); это сигнал
# «писать то, что удивляет ОСНОВНУЮ СЕТЬ» — бесплатно при forward, работает на
# инференсе (без разметки). η_t = 0.1 + 0.9·(ent_t/max_ent), порог ETA_MIN_SKIP.
# Всё остальное как exp2h (линейные слоты, delta-rule, retrieval).
# Ключевой урок серии: НЕЛИНЕЙНЫЙ слот не является ассоциативной памятью —
# M(q) для q≠записанного k произволен. ЛИНЕЙНЫЙ слот M(q)=W·q с записью
# W += η(v−Wk)kᵀ даёт M(q) = Wq + η(v−Wk)(k·q) — извлечение по похожести АВТОМАТИЧЕСКИ
# (поэтому линейные слоты v5 давали 34/40, а MLP — никогда).
# Здесь: 32 линейных слота (без GELU) + ассоциативная запись соседними парами
# (k=W_k(h_t) → v=W_v(h_{t+1})) с η-гейтом (mask_secret, секрет=1/мусор=0.1) + порог,
# чтение: w=softmax(q·W_route), mix6 = g·W_out(Σ w_i·W_i·q). Запись необучаемая (детach).

import torch, torch.nn as nn, torch.nn.functional as F, random
from train_memory_distill import (
    last_layer_forward, cand_logits, qwen, SEC_TOK_VALUES, SECRET_TO_IDX, DEV,
)


def qwen_lmhead_logits(h_out):
    """логпробы ПОЛНОГО словаря (открытый словарь) — для acc_open"""
    return qwen.lm_head(h_out.unsqueeze(0).to(qwen.lm_head.weight.dtype))[0]

D = 896
N_SLOTS = 32
CHUNK = 16
ETA = 0.3
GAMMA = 0.01
ETA_MIN_SKIP = 0.3
MAX_CTX_S = 200
N_TRAIN = 64
LR = 3e-3
PROJ_ITERS = 300
PHASE2_EPOCHS = 15


class LinearSlot(nn.Module):
    """один линейный слот: M(q) = W·q (без нелинейности — ассоциативная память)"""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(D, D))   # старт с нуля: чтение = 0 без записи

    def forward(self, q):
        return q @ self.W.t()


class LinSlotsAssoc(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([LinearSlot() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_out = nn.Linear(D, D)   # ВЕКТОР 896 (инъекция во вход последнего слоя)
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.W_eta = nn.Linear(D, 1)
        self.eta_min, self.eta_max = 0.1, 1.0

    def g(self):
        return torch.sigmoid(self.g_logit)

    def write_work(self, h_ctx, entropy=None):
        if entropy is not None:
            self.entropy = entropy
        """delta-rule запись по слотам: W_s += η_t·(v − W_s·k)⊗k, соседние пары (k_t→v_{t+1});
        η-гейт (mask_secret-учитель), порог ETA_MIN_SKIP — мусор не пишем"""
        k = F.gelu(self.W_k(h_ctx[:-1]))
        v = F.gelu(self.W_v(h_ctx[1:]))
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * (self.entropy / (self.entropy.max() + 1e-8))
        work = {}
        for c in range(0, len(k), CHUNK):
            groups = {}
            for j, (kk, vv) in enumerate(zip(k[c:c + CHUNK], v[c:c + CHUNK])):
                if eta_t[c + j] < ETA_MIN_SKIP:
                    continue
                s = (kk @ self.route_keys.t()).argmax().item()
                groups.setdefault(s, []).append((j, kk, vv))
            for s, items in groups.items():
                if s not in work:
                    work[s] = self.slots[s].W.detach().clone().requires_grad_(True)
                W = work[s]
                kks = torch.stack([kk for _, kk, _ in items])
                vvs = torch.stack([vv for _, _, vv in items])
                wts = eta_t[torch.tensor([j for j, _, _ in items], device=kks.device)]
                # delta-rule: W ← W(1−γ) − η·∇‖Wk−v‖² = W(1−γ) + η·Σ wts·(v−Wk)⊗k
                for _ in range(1):
                    pred = kks @ W.t()
                    iloss = (wts * (pred - vvs).pow(2).mean(-1)).mean()
                    gW = torch.autograd.grad(iloss, W, create_graph=False)[0]
                    gW = gW / (gW.norm() + 1e-8)
                    with torch.no_grad():
                        W = W * (1 - GAMMA) - ETA * gW
                    W = W.requires_grad_(True)
                work[s] = W
        return work

    def read_batch(self, q, work):
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        Ws = torch.stack([work[i] if i in work else self.slots[i].W.detach()
                          for i in range(N_SLOTS)])          # [32, D, D]
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        out = torch.einsum('bd,bdh->bh', qb, Ws.transpose(1, 2))  # [32, D]
        return out, w

    def mix_vec(self, q_hidden, work):
        """[896] — вектор инъекции во вход последнего слоя (открытый словарь)"""
        q = F.gelu(self.W_q(q_hidden))
        out, w = self.read_batch(q, work)
        return self.g() * self.W_out((out * w.unsqueeze(1)).sum(0))

    def read_theta(self, q):
        """отклик θ₀ (для пре-обучения проекций)"""
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        Ws = torch.stack([self.slots[i].W for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        out = torch.einsum('bd,bdh->bh', qb, Ws.transpose(1, 2))
        return (out * w.unsqueeze(1)).sum(0)


def main():
    data = torch.load("dataset_yattn_open_train.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = LinSlotsAssoc().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # пре-обучение проекций k/v/q ≈ identity (выравнивание)
    print("пре-обучение проекций (≈identity) ...", flush=True)
    for st in range(PROJ_ITERS):
        opt.zero_grad()
        d = train[st % len(train)]
        h = d["ctx_hidden"].to(DEV)[:16]
        hh = h.mean(0)
        loss = (lossf(F.gelu(model.W_k(hh)), hh) + lossf(F.gelu(model.W_v(hh)), hh)
                + lossf(F.gelu(model.W_q(hh)), hh)) / 3
        loss.backward()
        opt.step()
        if st % 100 == 99:
            print(f"  pre {st}: {loss.item():.4f}", flush=True)

    # ФАЗА 1.5 ОТСУТСТВУЕТ: η = нормированная энтропия Qwen (удивление
    # основной сети, exp2m AUC 0.85) — без учителя, работает на инференсе.
    print("фаза 2: линейные слоты + delta-rule + η-гейт + retrieval ...", flush=True)
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
            work = model.write_work(ctx, d["entropy"].to(DEV))
            mv = model.mix_vec(q_h, work)
            h_inj = h_all.clone()
            h_inj[-1] = h_inj[-1] + mv                       # ИНЪЕКЦИЯ во вход последнего слоя
            h_out_p = last_layer_forward(h_inj)
            # CE по ВСЕМУ словарю: первый токен секрета (открытый словарь!)
            lg = qwen.lm_head(h_out_p.unsqueeze(0).to(qwen.lm_head.weight.dtype))[0]
            tgt_tok = d["secret_tok0"].to(DEV)
            loss = lossf(h_out_p, y_out) + 0.5 * cef(lg, tgt_tok)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"эпоха {ep}: train {tot / len(train):.4f}", flush=True)

        model.eval()
        vacc, va_open, v_mse, vd, nv = 0, 0, 0.0, 0.0, 0
        per_type = {}
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q_h = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_work(ctx, d["entropy"].to(DEV))          # ВНЕ no_grad (клоны)
            with torch.no_grad():
                mv = model.mix_vec(q_h, work)
                h_inj = h_all.clone()
                h_inj[-1] = h_inj[-1] + mv
                h_out_p = last_layer_forward(h_inj)
                v_mse += lossf(h_out_p, y_out).item()
                lg = qwen_lmhead_logits(h_out_p)
                ok_open = lg.argmax(-1).item() == d["secret_tok0"].item()
                vacc += ok6
                va_open += ok_open
                vd += mv.norm().item()
                nv += 1
                per_type.setdefault(d["type"], [0, 0])
                per_type[d["type"]][0] += ok_open
                per_type[d["type"]][1] += 1
        pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
        print(f"  val: acc6 = {vacc / nv:.2f} | acc_open = {va_open / nv:.2f} | "
              f"val_mse = {v_mse / nv:.4f} | ‖mix‖ = {vd / nv:.2f} | "
              f"g = {model.g().item():.3f} | {pt}", flush=True)

    torch.save(model.state_dict(), "exp3b_open_train.pt")
    print("Сохранено: exp3b_open_train.pt", flush=True)


if __name__ == "__main__":
    main()
