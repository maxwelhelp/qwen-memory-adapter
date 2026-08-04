# Эксперимент 5: WINDOWED KEY + ДИНАМИЧЕСКИЙ ГЕЙТ + ЧЕСТНАЯ КРОСС-СЕССИОННАЯ
# ПЕРСИСТЕНТНОСТЬ (через диск, не через RAM-словарь в одном процессе).
#
# Строится поверх exp4a (windowed key) + добавляет два новых куска:
#
# 1) ДИНАМИЧЕСКИЙ ГЕЙТ. Раньше g = sigmoid(g_logit) — один скаляр на всю модель,
#    не зависящий от примера: память подмешивалась с одинаковой силой и когда Qwen
#    сама уверена в ответе, и когда не уверена. Теперь гейт зависит от ДВУХ сигналов,
#    которые уже вычисляются в пайплайне, просто раньше не были связаны:
#      - энтропия Qwen на 6 кандидатах БЕЗ памяти (base_cand) — "Qwen не знает,
#        нужна подсказка";
#      - уверенность роутинга памяти — концентрация softmax(q·W_route) по 32
#        слотам (низкая энтропия = память нашла конкретную запись, а не
#        размазана по всем слотам "не знаю").
#    Подмешивать сильно — только когда ОБА сигнала говорят "надо": Qwen не знает
#    И память уверенно что-то нашла. Иначе — минимально, чтобы не портить то,
#    что Qwen и так предсказала бы верно.
#
# 2) ЧЕСТНАЯ ПЕРСИСТЕНТНОСТЬ. cross_session_eval() реально сохраняет `work` на
#    диск и заново с нуля инстанцирует модель + грузит work из файла — то есть
#    "сессия 2" физически не имеет доступа ни к какому питоновскому объекту из
#    "сессии 1", только к файлам. Это ловит баги вида "тензор с requires_grad
#    не сериализуется", "объект держит ссылку на живой граф" и т.п., которые
#    не видны, если просто гонять работу внутри одного process/dict.
#    Для ЕЩЁ более строгого теста (реально два разных процесса ОС) —
#    session_write.py / session_read.py рядом.

import torch, torch.nn as nn, torch.nn.functional as F, random, os
from train_memory_distill import (
    last_layer_forward, cand_logits, qwen, SECRETS, SEC_TOK_VALUES, SECRET_TO_IDX, DEV,
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
WIN_BEFORE = 2
WIN_AFTER = 1
N_CAND = 6   # число кандидатов в cand_logits (см. train_memory_distill.SECRET_TO_IDX)


class LinearSlot(nn.Module):
    """один линейный слот: M(q) = W·q (без нелинейности — ассоциативная память)"""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(D, D))

    def forward(self, q):
        return q @ self.W.t()


class LocalWindowPool(nn.Module):
    """Обучаемый attention-пулинг по локальному окну вокруг позиции t (эксп. 4a)."""
    def __init__(self, d, w_before=WIN_BEFORE, w_after=WIN_AFTER):
        super().__init__()
        self.w_before, self.w_after = w_before, w_after
        self.score = nn.Linear(d, 1)

    def pooled_all(self, h_ctx):
        T = h_ctx.shape[0]
        out = []
        for t in range(T):
            lo = max(0, t - self.w_before)
            hi = min(T, t + self.w_after + 1)
            window = h_ctx[lo:hi]
            scores = self.score(window).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            pooled = (weights.unsqueeze(-1) * window).sum(0)
            out.append(pooled)
        return torch.stack(out)


class DynamicGate(nn.Module):
    """Гейт силы инъекции: f(энтропия Qwen на кандидатах без памяти,
    уверенность роутинга памяти). Оба сигнала нормированы в [0,1].
    Старт: веса нулевые, bias нулевой -> gate≈0.5 в начале обучения,
    дальше обучается сам находить нужный баланс."""
    def __init__(self, n_cand=N_CAND, n_slots=N_SLOTS):
        super().__init__()
        self.n_cand = n_cand
        self.n_slots = n_slots
        self.lin = nn.Linear(2, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, base_cand_logits, route_w):
        p = torch.softmax(base_cand_logits, dim=-1)
        h_qwen = -(p * p.clamp_min(1e-8).log()).sum()
        h_qwen_norm = h_qwen / torch.log(torch.tensor(float(self.n_cand), device=p.device))
        pw = route_w.clamp_min(1e-8)
        h_route = -(route_w * pw.log()).sum()
        h_route_norm = h_route / torch.log(torch.tensor(float(self.n_slots), device=p.device))
        route_confidence = 1.0 - h_route_norm
        # gate высокий, когда Qwen НЕ уверена (h_qwen_norm высокая) И память уверена
        # (route_confidence высокая) -> используем h_qwen_norm напрямую, не (1-h_qwen_norm)
        feats = torch.stack([h_qwen_norm, route_confidence])
        return torch.sigmoid(self.lin(feats))


class LinSlotsAssocCtxGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([LinearSlot() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)
        self.key_pool = LocalWindowPool(D)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_out = nn.Linear(D, D)
        self.gate_net = DynamicGate()
        self.eta_min, self.eta_max = 0.1, 1.0

    def write_work(self, h_ctx, entropy=None):
        if entropy is not None:
            self.entropy = entropy
        pooled = self.key_pool.pooled_all(h_ctx)
        k = F.gelu(self.W_k(pooled[:-1]))
        v = F.gelu(self.W_v(h_ctx[1:]))
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
                          for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        out = torch.einsum('bd,bdh->bh', qb, Ws.transpose(1, 2))
        return out, w

    def mix_vec(self, q_hidden, work, base_cand_logits):
        """base_cand_logits: логиты 6 кандидатов БЕЗ памяти (для гейта, detached)."""
        q = F.gelu(self.W_q(q_hidden))
        out, w = self.read_batch(q, work)
        gate = self.gate_net(base_cand_logits, w)
        return gate * self.W_out((out * w.unsqueeze(1)).sum(0))


def _detach_work(work):
    return {k: v.detach().cpu() for k, v in work.items()}


def _load_work(work_cpu):
    return {k: v.to(DEV) for k, v in work_cpu.items()}


def cross_session_eval(ckpt_path, val, tag=""):
    """Настоящая проверка через диск: 'сессия 1' пишет work и сохраняет в файл,
    ссылка на объект уничтожается; 'сессия 2' с нуля инстанцирует модель из
    чекпоинта и грузит work только из файла. Контекст в сессии 2 не используется
    нигде — только q_hidden (вопрос) и h_inA_all (для базового прогноза Qwen)."""
    os.makedirs("/tmp/mem_sessions", exist_ok=True)
    vacc, va_open, nv = 0, 0, 0
    per_type = {}
    confusion = {}
    for i, d in enumerate(val):
        # --- СЕССИЯ 1: пишем и сохраняем на диск, дальше объекты не используем ---
        model_w = LinSlotsAssocCtxGate().to(DEV)
        model_w.load_state_dict(torch.load(ckpt_path, map_location=DEV))
        model_w.eval()
        ctx = d["ctx_hidden"].to(DEV)
        work = model_w.write_work(ctx, d["entropy"].to(DEV))
        mem_path = f"/tmp/mem_sessions/mem_{tag}_{i}.pt"
        torch.save(_detach_work(work), mem_path)
        del model_w, work, ctx

        # --- СЕССИЯ 2: свежая модель + work только из файла, БЕЗ контекста ---
        model_r = LinSlotsAssocCtxGate().to(DEV)
        model_r.load_state_dict(torch.load(ckpt_path, map_location=DEV))
        model_r.eval()
        work_loaded = _load_work(torch.load(mem_path, map_location="cpu"))
        q_h = d["q_hidden"].to(DEV)
        h_all = d["h_inA_all"].to(DEV)
        tgt = SECRET_TO_IDX[d["secret"]]
        with torch.no_grad():
            base_cand = cand_logits(last_layer_forward(h_all))
            mv = model_r.mix_vec(q_h, work_loaded, base_cand)
            h_inj = h_all.clone()
            h_inj[-1] = h_inj[-1] + mv
            h_out_p = last_layer_forward(h_inj)
            c6 = cand_logits(h_out_p)
            ok6 = c6.argmax(-1).item() == tgt
            lg = qwen_lmhead_logits(h_out_p)
            ok_open = lg.argmax(-1).item() == SEC_TOK_VALUES[tgt]
        vacc += ok6
        va_open += ok_open
        nv += 1
        per_type.setdefault(d["type"], [0, 0])
        per_type[d["type"]][0] += ok_open
        per_type[d["type"]][1] += 1
        if "referent" in d:
            pred_secret = SECRETS[c6.argmax(-1).item()]
            ref_of_pred = dict(d["pairs"]).get(pred_secret)
            if ref_of_pred is not None:
                confusion[(d["referent"], ref_of_pred)] = confusion.get((d["referent"], ref_of_pred), 0) + 1
        os.remove(mem_path)
        del model_r, work_loaded

    pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
    if confusion:
        cstr = " | ".join(f"{k[0]}->{k[1]}:{v}" for k, v in sorted(confusion.items()))
        print(f"  [cross-session] confusion: {cstr}", flush=True)
    print(f"[cross-session, диск] acc6 = {vacc/nv:.2%} | acc_open = {va_open/nv:.2%} | {pt}",
          flush=True)
    return vacc / nv, va_open / nv


def main():
    import sys as _sys
    data = torch.load(_sys.argv[1] if len(_sys.argv) > 1 else "dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = LinSlotsAssocCtxGate().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

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

    print("фаза 2: windowed key + delta-rule + η-гейт(write) + dynamic gate(read) ...",
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
            base_cand = cand_logits(last_layer_forward(h_all)).detach()
            mv = model.mix_vec(q_h, work, base_cand)
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
        confusion = {}
        per_type = {}
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q_h = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_work(ctx, d["entropy"].to(DEV))
            with torch.no_grad():
                base_cand = cand_logits(last_layer_forward(h_all))
                mv = model.mix_vec(q_h, work, base_cand)
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
                if "referent" in d:
                    pred_secret = SECRETS[c6.argmax(-1).item()]
                    ref_of_pred = dict(d["pairs"]).get(pred_secret)
                    if ref_of_pred is not None:
                        confusion[(d["referent"], ref_of_pred)] = \
                            confusion.get((d["referent"], ref_of_pred), 0) + 1
        pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
        print(f"  val (in-process): acc6 = {vacc/nv:.2f} | acc_open = {va_open/nv:.2f} | "
              f"val_mse = {v_mse/nv:.4f} | ‖mix‖ = {vd/nv:.2f} | {pt}", flush=True)
        if confusion:
            cstr = " | ".join(f"{k[0]}->{k[1]}:{v}" for k, v in sorted(confusion.items()))
            print(f"  confusion: {cstr}", flush=True)

    ckpt_path = "exp5_ctx_gate.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Сохранено: {ckpt_path}", flush=True)

    print("\n=== ЧЕСТНАЯ КРОСС-СЕССИОННАЯ ПРОВЕРКА (через диск, не RAM) ===", flush=True)
    cross_session_eval(ckpt_path, val, tag="exp5")


if __name__ == "__main__":
    main()
