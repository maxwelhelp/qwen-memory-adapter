# Эксперимент 4a: WINDOWED KEY — ключ памяти строится из локального окна вокруг
# удивившего токена, а не из одной точки h_t.
#
# Зачем: в exp2n/exp3a ключ k = W_k(h_t) — снимок ОДНОЙ позиции. Если тип факта
# ("от почты", "от телефона") стоит рядом с секретом, но сам не вызывает удивления
# у Qwen (η мал), он в ключ вообще не попадает — отсюда интерференция типов на
# масштабе (exp2n_full, 320 примеров: путаница «пароль/пин-код»).
#
# Идея (обсуждали): вместо k=h_t брать k = attn_pool(h_{t-w_before..t+w_after}) —
# обучаемый локальный attention-пулинг по окну вокруг t. Веса пулинга учатся сами
# (score = Linear(h), softmax по окну) — модель сама решает, какие соседние токены
# несут информацию "к чему относится" этот факт. Value (v = h_{t+1}) остаётся
# точечным — мы хотим ТОЧНО вернуть значение секрета, а не его контекст.
#
# Всё остальное — как в exp3a: линейные слоты + delta-rule + η-гейт из энтропии
# Qwen (без учителя) + векторная инъекция во вход последнего слоя (открытый словарь).
#
# Это ablation №1 из мини-плана (4a). Следующий шаг при подтверждении эффекта — 4c
# (per-type routing heads) поверх этой же схемы.

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

# --- новые гиперпараметры окна ---
WIN_BEFORE = 2   # сколько токенов ДО t включать в ключ (контекст "к чему относится")
WIN_AFTER = 1    # сколько токенов ПОСЛЕ t (иногда тип идёт сразу за секретом)


class LinearSlot(nn.Module):
    """один линейный слот: M(q) = W·q (без нелинейности — ассоциативная память)"""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(D, D))   # старт с нуля: чтение = 0 без записи

    def forward(self, q):
        return q @ self.W.t()


class LocalWindowPool(nn.Module):
    """Обучаемый attention-пулинг по локальному окну вокруг позиции t.
    score(h_i) = Linear(h_i) -> softmax по окну -> взвешенная сумма.
    Даёт ключу семантику соседних токенов ("от почты"/"от телефона"),
    а не только точку самого удивившего токена."""
    def __init__(self, d, w_before=WIN_BEFORE, w_after=WIN_AFTER):
        super().__init__()
        self.w_before, self.w_after = w_before, w_after
        self.score = nn.Linear(d, 1)

    def pooled_all(self, h_ctx):
        """Возвращает [T, D] — для каждой позиции t пуленный вектор окна.
        Векторизовано через unfold было бы быстрее, но T тут мало (<200),
        поэтому простой питоновский цикл ради читаемости."""
        T = h_ctx.shape[0]
        out = []
        for t in range(T):
            lo = max(0, t - self.w_before)
            hi = min(T, t + self.w_after + 1)
            window = h_ctx[lo:hi]                      # [w, D]
            scores = self.score(window).squeeze(-1)     # [w]
            weights = torch.softmax(scores, dim=0)
            pooled = (weights.unsqueeze(-1) * window).sum(0)  # [D]
            out.append(pooled)
        return torch.stack(out)                          # [T, D]


class LinSlotsAssocCtx(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([LinearSlot() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.key_pool = LocalWindowPool(D)     # NEW: локальный пулинг для ключа
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_out = nn.Linear(D, D)   # вектор 896 (инъекция во вход последнего слоя)
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.eta_min, self.eta_max = 0.1, 1.0

    def g(self):
        return torch.sigmoid(self.g_logit)

    def write_work(self, h_ctx, entropy=None):
        if entropy is not None:
            self.entropy = entropy
        """delta-rule запись по слотам: ключ теперь = W_k(pool(окно вокруг t)),
        значение по-прежнему точечное v = W_v(h_{t+1}) — мы хотим вернуть ТОЧНОЕ
        значение секрета, контекст нужен только для различения "какой именно"."""
        pooled = self.key_pool.pooled_all(h_ctx)          # [T, D]
        k = F.gelu(self.W_k(pooled[:-1]))                  # ключ из окна
        v = F.gelu(self.W_v(h_ctx[1:]))                     # значение точечное, как раньше
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * (
            self.entropy / (self.entropy.max() + 1e-8))
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


def main():
    torch.manual_seed(0)   # ДЕТЕРМИНИЗМ: фикс. seed для сравнения версий
    import sys as _sys
    _ds_path = _sys.argv[1] if len(_sys.argv) > 1 else "dataset_yattn.pt"
    _ds_tag = "multitype" if "multi" in _ds_path else "yattn"
    data = torch.load(_ds_path)
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = LinSlotsAssocCtx().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # пре-обучение проекций k/v/q ≈ identity (выравнивание)
    # ВАЖНО: k теперь строится из pooled-окна, поэтому выравниваем W_k тоже на
    # pooled-представлении (пулинг на старте почти uniform, так что hh годится
    # как приближение — донастроится в фазе 2 вместе с score-головой пулинга).
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

    print("фаза 2: windowed key + линейные слоты + delta-rule + η-гейт + retrieval ...",
          flush=True)
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
            h_inj[-1] = h_inj[-1] + mv
            h_out_p = last_layer_forward(h_inj)
            loss = lossf(h_out_p, y_out) + 0.5 * cef(cand_logits(h_out_p), tgt)
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
            work = model.write_work(ctx, d["entropy"].to(DEV))
            with torch.no_grad():
                mv = model.mix_vec(q_h, work)
                h_inj = h_all.clone()
                h_inj[-1] = h_inj[-1] + mv
                h_out_p = last_layer_forward(h_inj)
                v_mse += lossf(h_out_p, y_out).item()
                c6 = cand_logits(h_out_p)
                ok6 = c6.argmax(-1).item() == tgt
                lg = qwen_lmhead_logits(h_out_p)
                ok_open = lg.argmax(-1).item() == SEC_TOK_VALUES[tgt]
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

    torch.save(model.state_dict(), "exp4a_windowed_key.pt")
    print("Сохранено: exp4a_windowed_key.pt", flush=True)

    # --- сравнение с exp3a напечатать вручную ---
    print("\nСравни acc_open и per_type с exp3a (25/40, 62.5%) — если 4a лучше "
          "именно на разнотипных/масштабных случаях, окно даёт эффект. "
          "На текущем датасете (1 секрет на тип) эффекта может почти не быть — "
          "нужен multi-type тест-сет (см. план, эксп. отдельно генерировать).",
          flush=True)


if __name__ == "__main__":
    main()
