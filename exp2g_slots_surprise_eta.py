# Эксперимент 2g: exp2d + η-ГЕЙТ ИЗ ВНУТРЕННЕГО УДИВЛЕНИЯ (БЕЗ УЧИТЕЛЯ).
# exp2f доказал: ‖M_θ₀(h)−h‖² выровненной памяти различает секрет/мусор (AUC 0.89,
# секрет в 2.86x сильнее) — это механизм Titans (surprise = внутр. лосс памяти),
# работающий БЕЗ mask_secret (на инференсе маски нет). Здесь: η_t из удивления
# + порог (перцентиль удивления на train), токены ниже порога НЕ пишутся.
# Всё остальное как exp2d/2e (слоты, ассоциация k_t→v_{t+1}, retrieval-чтение).
# Указание пользователя: сначала обучить память писать ТОЛЬКО удивляющее,
# потом тестировать извлечение. В exp2a/2b/2d запись была равномерной (ETA=0.3) —
# мусор забивал слоты. Здесь: η_t = σ(W_eta(h_t)), учитель mask_secret (секрет=1,
# мусор=0.1 — как v5), шаг записи масштабируется η_t, токены с η < ETA_MIN_SKIP
# НЕ ПИШУТСЯ вовсе. Всё остальное как exp2d (слоты, ассоциация k_t→v_{t+1}, retrieval).
# Что работает по отдельности (доказано) — объединяем:
#  - СЛОТЫ: разделение фактов (v5: 32 слота; линейные 34/40 работали как retrieval;
#           без слотов — «каша», exp2a/2b: одна память тонет в мусоре);
#  - АССОЦИАТИВНАЯ ЗАПИСЬ: M(k)≈v, k=GELU(W_k(h_t)), v=GELU(W_v(h_{t+1})) — соседние
#           пары «маркер→секрет» (exp2c: чтение по похожему ключу работает, cos 0.87);
#  - RETRIEVAL-ЧТЕНИЕ: w = softmax(q·W_route) по слотам — вопрос попадает в слот секрета
#           по похожести, mix = g·W_out(Σ_i w_i·M_i(q;запись)).
# Запись: необучаемая механика (клоны, как v5) — сначала БЕЗ bilevel (одна переменная).
# Метрики: val_acc6, ‖mix6−target6‖, per-type.

import torch, torch.nn as nn, torch.nn.functional as F, random
from train_memory_distill import (
    last_layer_forward, cand_logits, SECRET_TO_IDX, DEV,
)

D = 896
N_SLOTS = 32
CHUNK = 16
ITERS = 2
ETA = 0.3
GAMMA = 0.01
ETA_MIN_SKIP = 0.3        # порог: η ниже — токен не пишем
MAX_CTX_S = 200
N_TRAIN = 64
LR = 3e-3
PROJ_ITERS = 300
PHASE2_EPOCHS = 15


class SlotMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, w1=None, w2=None):
        if w1 is None:
            return self.net(x)
        return F.gelu(x @ w1.t()) @ w2.t()


class SlotsAssocRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([SlotMLP() for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)  # маршрут записи
        self.W_route = nn.Parameter(torch.randn(N_SLOTS, D) * 0.01)     # маршрут чтения
        self.W_k = nn.Linear(D, D, bias=False)   # ключи (проекции токенов)
        self.W_v = nn.Linear(D, D, bias=False)   # значения (проекции след. токена)
        self.W_out = nn.Linear(D, 6)
        self.g_logit = nn.Parameter(torch.zeros(()))
        self.W_eta = nn.Linear(D, 1)                 # η-голова: КАК СИЛЬНО писать
        self.eta_min, self.eta_max = 0.1, 1.0

    def g(self):
        return torch.sigmoid(self.g_logit)

    def write_work(self, h_ctx):
        """ассоциативная запись по слотам с η-ГЕЙТОМ: k=W_k(h_t), v=W_v(h_{t+1});
        шаг записи масштабируется η_t (surprise-гейт), η < ETA_MIN_SKIP → токен НЕ пишется"""
        k = F.gelu(self.W_k(h_ctx[:-1]))
        v = F.gelu(self.W_v(h_ctx[1:]))
        # УДИВЛЕНИЕ (Titans): насколько токен аномален для ВЫРОВНЕННОЙ памяти θ₀
        with torch.no_grad():
            surprise = (self.read_theta(h_ctx[:-1]) - h_ctx[:-1]).pow(2).mean(-1)
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * (surprise / (surprise.max() + 1e-8))
        work = {}
        for c in range(0, len(k), CHUNK):
            groups = {}
            for j, (kk, vv) in enumerate(zip(k[c:c + CHUNK], v[c:c + CHUNK])):
                if eta_t[c + j] < ETA_MIN_SKIP:
                    continue                                  # порог: не пишем то, что не удивило
                s = (kk @ self.route_keys.t()).argmax().item()
                groups.setdefault(s, []).append((j, kk, vv))
            for s, items in groups.items():
                if s not in work:
                    work[s] = {n: p.detach().clone().requires_grad_(True)
                               for n, p in self.slots[s].named_parameters()}
                params = work[s]
                kks = torch.stack([kk for _, kk, _ in items])
                vvs = torch.stack([vv for _, _, vv in items])
                wts = eta_t[torch.tensor([j for j, _, _ in items], device=kks.device)]
                for _ in range(ITERS):
                    pred = self.slots[s].forward(kks, params["net.0.weight"], params["net.2.weight"])
                    iloss = (wts * F.mse_loss(pred, vvs, reduction='none').mean(-1)).mean()  # η-взвеш.
                    g1, g2 = torch.autograd.grad(iloss,
                                                 [params["net.0.weight"], params["net.2.weight"]],
                                                 create_graph=False)
                    g1 = g1 / (g1.norm() + 1e-8)
                    g2 = g2 / (g2.norm() + 1e-8)
                    with torch.no_grad():
                        params["net.0.weight"] = params["net.0.weight"] * (1 - GAMMA) - ETA * g1
                        params["net.2.weight"] = params["net.2.weight"] * (1 - GAMMA) - ETA * g2
                    params["net.0.weight"] = params["net.0.weight"].requires_grad_(True)
                    params["net.2.weight"] = params["net.2.weight"].requires_grad_(True)
        return work

    def read_batch(self, q, work):
        """выходы ВСЕХ слотов батчем (einsum) + веса чтения w=softmax(q·W_route)"""
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1 = torch.stack([work[i]["net.0.weight"].t() if i in work
                          else self.slots[i].net[0].weight.detach().t()
                          for i in range(N_SLOTS)])
        W2 = torch.stack([work[i]["net.2.weight"].t() if i in work
                          else self.slots[i].net[2].weight.detach().t()
                          for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        h = F.gelu(torch.einsum('bd,bdh->bh', qb, W1))
        out = torch.einsum('bh,bhd->bd', h, W2)
        return out, w

    def mix6(self, q, work):
        """retrieval-чтение: взвешенная сумма по слотам → 6 чисел"""
        out, w = self.read_batch(q, work)
        return self.g() * self.W_out((out * w.unsqueeze(1)).sum(0))

    def read_theta(self, q):
        """отклик θ₀ (для пре-обучения проекций/выравнивания)"""
        w = torch.softmax(q @ self.W_route.t(), dim=0)
        W1 = torch.stack([self.slots[i].net[0].weight.t() for i in range(N_SLOTS)])
        W2 = torch.stack([self.slots[i].net[2].weight.t() for i in range(N_SLOTS)])
        qb = q.unsqueeze(0).expand(N_SLOTS, -1)
        h = F.gelu(torch.einsum('bd,bdh->bh', qb, W1))
        out = torch.einsum('bh,bhd->bd', h, W2)
        return (out * w.unsqueeze(1)).sum(0)


def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX_S]
    print(f"примеров с контекстом < {MAX_CTX_S}: {len(ex)}", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = SlotsAssocRetrieval().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # пре-обучение: проекции k/v ≈ identity на hidden (выравнивание), слоты ≈ identity
    print("пре-обучение (проекции ≈ identity, слоты ≈ identity) ...", flush=True)
    for st in range(PROJ_ITERS):
        opt.zero_grad()
        d = train[st % len(train)]
        h = d["ctx_hidden"].to(DEV)[:16]
        hh = h.mean(0)
        loss = (lossf(F.gelu(model.W_k(hh)), hh) + lossf(F.gelu(model.W_v(hh)), hh)
                + 0.5 * lossf(model.read_theta(hh), hh))
        loss.backward()
        opt.step()
        if st % 100 == 99:
            print(f"  pre {st}: {loss.item():.4f}", flush=True)

    # ФАЗА 1.5 ОТСУТСТВУЕТ: η берётся из внутреннего удивления (exp2f: AUC 0.89,
    # секрет в 2.86x сильнее) — без учителя, работает на инференсе.
    print("фаза 2: слоты+ассоциативная запись+η-гейт+retrieval-чтение ...", flush=True)
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
            work = model.write_work(ctx)
            mv = model.mix6(q_h, work)
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
        per_type = {}
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q_h = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_work(ctx)          # ВНЕ no_grad (запись строит клоны)
            with torch.no_grad():
                mv = model.mix6(q_h, work)
                base_cand = cand_logits(last_layer_forward(h_all))
                target6 = (cand_logits(y_out) - base_cand).float()
                final_cand = base_cand + mv.to(base_cand.dtype)
                ok = final_cand.argmax(-1).item() == tgt
                vacc += ok
                vd += (mv.to(target6.dtype) - target6).norm().item()
                nv += 1
                per_type.setdefault(d["type"], [0, 0])
                per_type[d["type"]][0] += ok
                per_type[d["type"]][1] += 1
        pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
        print(f"  val: acc6 = {vacc / nv:.2f} | ‖mix6−target6‖ = {vd / nv:.2f} | "
              f"g = {model.g().item():.3f} | {pt}", flush=True)

    torch.save(model.state_dict(), "exp2g_slots_surprise_eta.pt")
    print("Сохранено: exp2g_slots_surprise_eta.pt", flush=True)


if __name__ == "__main__":
    main()
