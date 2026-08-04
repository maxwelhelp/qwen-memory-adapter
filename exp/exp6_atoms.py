# Эксперимент 6: RANK-1 GATED ATOMS вместо 32 плотных матриц-слотов.
#
# Идея (обсуждали): текущие 32 слота — это 32 полные матрицы D×D (~803K параметров
# КАЖДАЯ, ~25.7M суммарно), и роутинг между ними при записи — жёсткий argmax в ОДИН
# слот. Это грубая адресация: 896-мерное пространство фактов сжимается всего в 32
# бакета, и разные факты одного типа ("пароль от почты" / "пароль от телефона")
# легко попадают в один и тот же слот и портят друг друга там через общую матрицу.
#
# По аналогии со SwiGLU MLP Qwen (каждый нейрон = rank-1 объект: гейт-направление +
# write-направление, gate(x)=SiLU(Wg·x), вклад = gate(x)·read(x)·Wd[:,j]) — делаем
# банк из МНОГИХ (N_ATOMS=384) rank-1 атомов вместо малого числа "толстых" матриц.
# Атом = пара векторов (gate_key_j, write_j), D=896 каждый -> 2·D≈1792 параметров
# на атом. В бюджет ОДНОГО старого слота (~803K) помещается ~450 таких атомов.
#
# КЛЮЧЕВОЕ АРХИТЕКТУРНОЕ РЕШЕНИЕ (в отличие от провалившихся MLP-слотов v2/exp2g):
# нелинейность (SiLU) сидит ТОЛЬКО в скалярном гейте, а сам вклад атома в выход —
# это скаляр × фиксированное направление write_j. Retrieval по похожести (то, что
# давало 85-100% линейным слотам) сохраняется: близкий query активирует те же
# атомы, что были активны при записи похожего ключа — это уже не гарантировано
# математически как в чисто линейном слое (из-за SiLU), но эмпирически похожие
# по направлению векторы дают похожую активацию гейта, чего достаточно для
# ассоциативного extraction. Полноценно нелинейная, монолитная M(q)=MLP(q) — то,
# что провалилось раньше — здесь не используется вообще.
#
# ВАЖНЫЙ ФИКС по сравнению с exp2n/exp3a/exp5: там route_keys (запись, argmax) и
# W_route (чтение, softmax) — РАЗНЫЕ параметры; route_keys никогда не получал
# градиент (участвует только в недифференцируемом argmax). Здесь gate_key ОДИН
# параметр на запись и на чтение — так как чтение дифференцируемо (внутри
# mix_vec), градиент через gate_key доходит и обучает его отбирать атомы разумно,
# что автоматически улучшает и запись (тот же параметр).
#
# ДВЕ ФАЗЫ ОБУЧЕНИЯ (то, что уже было в пайплайне, здесь просто явно прокомментировано):
#   Фаза 1+2 (train_memory_distill / main()) — OFFLINE supervised: сеть видит
#     контекст, что она пишет (write_buf), что отдаёт (mix_vec), и НАСТОЯЩИЙ
#     правильный ответ (h_outB) — лосс MSE+CE. Обучаются slow weights: W_k, W_v,
#     W_q, W_out, gate_key, DynamicGate. write_buf НЕ обучается backprop'ом — он
#     каждый раз строится заново фиксированной delta-rule формулой (см. write_atoms).
#   cross_session_eval / реальный инференс — slow weights ЗАМОРОЖЕНЫ (load + eval()),
#     write_buf адаптируется на лету по той же формуле, без меток, без градиента.
#     Это и есть "test-time training" в духе Titans — тут ничего нового не введено,
#     просто теперь явно разделено в коде и комментариях.

import torch, torch.nn as nn, torch.nn.functional as F, random, os
from train_memory_distill import (
    last_layer_forward, cand_logits, qwen, SECRETS, SEC_TOK_VALUES, SECRET_TO_IDX, DEV,
)


def qwen_lmhead_logits(h_out):
    return qwen.lm_head(h_out.unsqueeze(0).to(qwen.lm_head.weight.dtype))[0]


D = 896
N_ATOMS = 384          # атомов гораздо больше, чем было слотов (32) -> тоньше адресация
TOP_K = 8               # разреженная активация: сколько атомов реально трогаем за токен
CHUNK = 16
ETA_WRITE = 0.3          # базовый шаг записи (аналог старого ETA для delta-rule)
GAMMA = 0.01             # забывание неактивных-но-задетых атомов
ETA_MIN_SKIP = 0.3       # порог: ниже — токен не пишем вообще (мусор)
ETA_LO, ETA_HI = 0.1, 1.0
MAX_CTX_S = 200
N_TRAIN = 64
LR = 3e-3
PROJ_ITERS = 300
PHASE2_EPOCHS = 15
WIN_BEFORE, WIN_AFTER = 2, 1     # то же окно контекста, что в exp4a/exp5
N_CAND = 6


class LocalWindowPool(nn.Module):
    """Обучаемый attention-пулинг по окну вокруг t (эксп. 4a, без изменений)."""
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
            w = torch.softmax(self.score(window).squeeze(-1), dim=0)
            out.append((w.unsqueeze(-1) * window).sum(0))
        return torch.stack(out)


class AtomBank(nn.Module):
    """Банк rank-1 гейтед-атомов. gate_key — ЕДИНЫЙ параметр для записи и чтения
    (см. комментарий вверху файла про фикс необучаемости старого route_keys)."""
    def __init__(self, d=D, n_atoms=N_ATOMS, top_k=TOP_K):
        super().__init__()
        self.d, self.n_atoms, self.top_k = d, n_atoms, top_k
        self.gate_key = nn.Parameter(torch.randn(n_atoms, d) * (d ** -0.5))

    def gate(self, x):
        """SiLU-гейт по всем атомам для вектора x [D] -> [n_atoms].
        Дифференцируемо по gate_key и по x (не topk-маска, а сами значения)."""
        return F.silu(self.gate_key @ x)

    def top_k_mask(self, g):
        k = min(self.top_k, self.n_atoms)
        idx = torch.topk(g.abs(), k).indices
        mask = torch.zeros_like(g)
        mask[idx] = 1.0
        return mask

    def init_buf(self):
        """'θ0' — пустая память на старте сессии/примера."""
        return torch.zeros(self.n_atoms, self.d, device=self.gate_key.device)

    def write_atoms(self, write_buf, k_seq, v_seq, eta_t):
        """Fast-weight обновление (НЕОБУЧАЕМАЯ формула, как и delta-rule в exp2n/
        exp3a/exp5 — 'запись необучаемая'). k_seq/v_seq: [T,D] (уже из windowed
        key / точечного value). eta_t: [T] — вес токена из энтропии Qwen (unsupervised)."""
        buf = write_buf
        # БЕЗ no_grad: buf — графовый узел от k/v/eta → key_pool, W_k, W_v, gate_key
        # обучаются через mix_vec → loss (фикс «запись не обучалась»).
        for t in range(len(k_seq)):
            if eta_t[t] < ETA_MIN_SKIP:
                continue
            g = self.gate(k_seq[t])                      # [n_atoms], нелинейный обученный гейт
            mask = self.top_k_mask(g)
            gm = g * mask                                   # разреженная взвешенная активация
            strength = (ETA_WRITE * eta_t[t] * gm).unsqueeze(-1)   # [n_atoms,1]
            decay = (GAMMA * mask).unsqueeze(-1)
            buf = buf * (1 - decay) + strength * (v_seq[t].unsqueeze(0) - buf)
        return buf

    def read_atoms(self, write_buf, q):
        """Дифференцируемо (используется внутри mix_vec, тут градиент и доходит
        до gate_key). Возвращает (вектор, confidence retrieval)."""
        g = self.gate(q)
        mask = self.top_k_mask(g)
        gm = g * mask
        out = (gm.unsqueeze(-1) * write_buf).sum(0)
        conf = gm.abs().sum() / (g.abs().sum() + 1e-8)   # доля массы гейта в top-k
        return out, conf


class DynamicGate(nn.Module):
    """Нелинейный (2-слойный) гейт силы инъекции: f(энтропия Qwen без памяти,
    уверенность retrieval атомов). Раньше был один Linear(2,1) — теперь с Tanh
    внутри, чтобы гейт мог учиться нелинейной комбинации сигналов, а не только
    их взвешенной сумме."""
    def __init__(self, n_cand=N_CAND, hidden=8):
        super().__init__()
        self.n_cand = n_cand
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # старт gate≈sigmoid(0)=0.5

    def forward(self, base_cand_logits, read_confidence):
        p = torch.softmax(base_cand_logits, dim=-1)
        h_qwen = -(p * p.clamp_min(1e-8).log()).sum()
        h_qwen_norm = h_qwen / torch.log(torch.tensor(float(self.n_cand), device=p.device))
        feats = torch.stack([h_qwen_norm, read_confidence])
        return torch.sigmoid(self.net(feats))


class AtomMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_pool = LocalWindowPool(D)
        self.atoms = AtomBank()
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_out = nn.Linear(D, D)
        self.gate_net = DynamicGate()

    def write_work(self, h_ctx, entropy=None):
        """Возвращает write_buf [N_ATOMS, D] — плотный тензор.
        ЗАПИСЬ ОБУЧАЕМАЯ (фикс бага «мёртвого key_pool»): buf строится с графом
        (без no_grad) — key_pool/W_k/W_v/gate_key получают градиент через mix_vec→loss."""
        if entropy is None:
            entropy = torch.ones(len(h_ctx) - 1, device=h_ctx.device)
        pooled = self.key_pool.pooled_all(h_ctx)
        k = F.gelu(self.W_k(pooled[:-1]))
        v = F.gelu(self.W_v(h_ctx[1:]))
        eta_t = ETA_LO + (ETA_HI - ETA_LO) * (entropy / (entropy.max() + 1e-8))
        buf = self.atoms.init_buf()
        return self.atoms.write_atoms(buf, k, v, eta_t)

    def mix_vec(self, q_hidden, write_buf, base_cand_logits):
        q = F.gelu(self.W_q(q_hidden))
        out, conf = self.atoms.read_atoms(write_buf, q)
        gate = self.gate_net(base_cand_logits, conf)
        return gate * self.W_out(out)


def cross_session_eval(ckpt_path, val, tag=""):
    """Как в exp5: реальный round-trip через диск, свежая модель на каждой стороне."""
    os.makedirs("/tmp/mem_sessions_atoms", exist_ok=True)
    vacc, va_open, nv = 0, 0, 0
    per_type = {}
    for i, d in enumerate(val):
        model_w = AtomMemory().to(DEV)
        model_w.load_state_dict(torch.load(ckpt_path, map_location=DEV))
        model_w.eval()
        ctx = d["ctx_hidden"].to(DEV)
        buf = model_w.write_work(ctx, d["entropy"].to(DEV))
        mem_path = f"/tmp/mem_sessions_atoms/mem_{tag}_{i}.pt"
        torch.save(buf.detach().cpu(), mem_path)
        del model_w, buf, ctx

        model_r = AtomMemory().to(DEV)
        model_r.load_state_dict(torch.load(ckpt_path, map_location=DEV))
        model_r.eval()
        buf_loaded = torch.load(mem_path, map_location=DEV)
        q_h = d["q_hidden"].to(DEV)
        h_all = d["h_inA_all"].to(DEV)
        tgt = SECRET_TO_IDX[d["secret"]]
        with torch.no_grad():
            base_cand = cand_logits(last_layer_forward(h_all))
            mv = model_r.mix_vec(q_h, buf_loaded, base_cand)
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
        os.remove(mem_path)
        del model_r, buf_loaded

    pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
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

    model = AtomMemory().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # ФАЗА 1: выравнивание проекций (≈identity) — без изменений относительно exp4a/5
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

    # ФАЗА 2: OFFLINE supervised обучение "логики" записи/чтения (см. комментарий
    # вверху файла) — сеть видит контекст, ЧТО она пишет (write_buf), ЧТО отдаёт
    # (mix_vec) и настоящий правильный ответ (y_out), лосс = MSE + CE.
    # write_buf НЕ обучается backprop'ом — строится заново фиксированной формулой
    # write_atoms на каждом примере; обучаются только slow weights модели.
    print("фаза 2: rank-1 атомы (N_ATOMS={}, TOP_K={}) + windowed key + dynamic gate ..."
          .format(N_ATOMS, TOP_K), flush=True)
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
            buf = model.write_work(ctx, d["entropy"].to(DEV))
            base_cand = cand_logits(last_layer_forward(h_all)).detach()
            mv = model.mix_vec(q_h, buf, base_cand)
            h_inj = h_all.clone()
            h_inj[-1] = h_inj[-1] + mv
            h_out_p = last_layer_forward(h_inj)
            loss = lossf(h_out_p, y_out) + 0.5 * cef(cand_logits(h_out_p), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        print(f"эпоха {ep}: train {tot/len(train):.4f}", flush=True)

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
            buf = model.write_work(ctx, d["entropy"].to(DEV))
            with torch.no_grad():
                base_cand = cand_logits(last_layer_forward(h_all))
                mv = model.mix_vec(q_h, buf, base_cand)
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
                        confusion[(d["referent"], ref_of_pred)] = confusion.get((d["referent"], ref_of_pred), 0) + 1
        if confusion:
            cstr = " | ".join(f"{k[0]}->{k[1]}:{v}" for k, v in sorted(confusion.items()))
            print(f"  confusion: {cstr}", flush=True)
        pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
        print(f"  val (in-process): acc6 = {vacc/nv:.2f} | acc_open = {va_open/nv:.2f} | "
              f"val_mse = {v_mse/nv:.4f} | ‖mix‖ = {vd/nv:.2f} | {pt}", flush=True)

    ckpt_path = "exp6_atoms.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Сохранено: {ckpt_path}", flush=True)

    print("\n=== ЧЕСТНАЯ КРОСС-СЕССИОННАЯ ПРОВЕРКА (через диск) ===", flush=True)
    cross_session_eval(ckpt_path, val, tag="exp6")


if __name__ == "__main__":
    main()
