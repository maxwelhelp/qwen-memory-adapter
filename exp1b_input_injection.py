# Эксперимент 1b (PLAN_MEMORY_V3, Ф1-вариант B): ВЕКТОРНАЯ ИНЪЕКЦИЯ во ВХОД последнего слоя.
# Докстринг v5 заявляет (не реализовано): h'_in[-1] = h_in[-1] + g·W_out(M(q)−M_θ₀(q));
# h_out' = last_layer(h'_in)[-1]; loss = MSE(h_out', h_outB) + λ·CE(кандидаты).
# Отличие от прошлого провала (Δ выхода ~150, нормализация давила сигнал в 150 раз):
# инъекция во ВХОД последнего слоя; требуемый Δ входа ≈ 1.21 (замер Ф0, не 5-10 как заявлено).
# Цель: РЕАЛЬНАЯ векторная память (открытый словарь), не голосование за 6 кандидатов.
# Метрики: val_mse(h_out', h_outB); acc6 (по 6 кандидатам, для сравнения с v5);
#          acc_open (argmax ПОЛНОГО lm_head == первый токен секрета — открытый словарь);
#          norm(mix − target_in) — контроль точности инъекции; per-type acc_open.
# Запись — необучаемая механика (как v5); обучаются W_out(896→896), W_route, g, W_eta.

import torch, torch.nn as nn, random
from train_memory_distill import (
    Memory as MemoryV5, last_layer_forward, cand_logits, qwen,
    SECRETS, SEC_TOK_VALUES, SECRET_TO_IDX, DEV, MAX_CTX, N_TRAIN, BATCH, LR,
)


def qwen_lmhead_logits(h_out):
    """логпробы ПОЛНОГО словаря lm_head (открытый словарь) — для acc_open"""
    return qwen.lm_head(h_out.unsqueeze(0).to(qwen.lm_head.weight.dtype))[0]

EPOCHS_D = 12      # диагностика: урезанная дистилляция (полная — если сходится)
ALIGN_EPOCHS = 15
ETA_EPOCHS = 10


class MemoryB(MemoryV5):
    """v5 + векторный выход: W_out 896→896, mix — вектор в пространстве hidden."""

    def __init__(self):
        super().__init__()
        self.W_out = nn.Linear(896, 896)          # вместо 896→6
        self.g_logit = nn.Parameter(torch.zeros(()))

    def mix_vector(self, q, work):
        """[896] — вектор инъекции во вход последнего слоя на последней позиции"""
        return self.g() * self.W_out(self.read_diff(q, work))


def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX]
    print(f"примеров с контекстом < {MAX_CTX}: {len(ex)} (из {len(data)})", flush=True)
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = max(1, len(ex) // 10)
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]

    model = MemoryB().to(DEV)
    model.refresh_theta_stacks()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    cef = nn.CrossEntropyLoss()

    # ---- ФАЗА 1: alignment (тот же код, что в v5) ----
    print("Фаза 1: alignment (M(h) ≈ h)", flush=True)
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
        with torch.no_grad():
            for d in val:
                ctx = d["ctx_hidden"].to(DEV)
                for h in ctx[:16]:
                    vloss += lossf(model.read_theta(h), h).item()
        vloss /= (len(val) * 16)
        print(f"align {ep:2d}: train {tot / n:.4f} | val {vloss:.4f}", flush=True)

    # ---- ФАЗА 1.5: η-голова (тот же код, что в v5) ----
    print("Фаза 1.5: η-голова (что писать)", flush=True)
    bcef = nn.BCEWithLogitsLoss()
    for ep in range(ETA_EPOCHS):
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

    # ---- ЗАМОРОЗКА СЛОТОВ (как v5) ----
    for p in model.slots.parameters():
        p.requires_grad = False
    model.refresh_theta_stacks()
    print("Слоты заморожены: обучаются W_out(896→896), W_route, g, W_eta", flush=True)

    # ---- ФАЗА 2: дистилляция ВЕКТОРНОЙ инъекции ----
    print("Фаза 2: дистилляция (векторная инъекция во вход последнего слоя)", flush=True)
    for ep in range(EPOCHS_D):
        model.train()
        opt.zero_grad()
        tot, n = 0.0, 0
        for i, d in enumerate(train):
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = torch.tensor(SECRET_TO_IDX[d["secret"]], device=DEV)
            work = model.write_work(ctx)
            mv = model.mix_vector(q, work)
            h_inj = h_all.clone()
            h_inj[-1] = h_inj[-1] + mv
            h_out_p = last_layer_forward(h_inj)
            loss_mse = lossf(h_out_p, y_out)
            loss_ce = cef(cand_logits(h_out_p), tgt)
            loss = loss_mse + 0.5 * loss_ce
            loss.backward()
            tot += loss.item(); n += 1
            if (i + 1) % BATCH == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        if n % BATCH:
            opt.step(); opt.zero_grad()

        # ---- val: mse, acc6, acc_open (весь словарь), norm(mix − target_in), per-type ----
        # ВАЖНО: write_work НЕЛЬЗЯ вызывать под no_grad (клоны с requires_grad_ не строят
        # граф в no_grad → autograd.grad падает). Как в v5: write_work вне no_grad,
        # остальное (чтение/метрики) — под no_grad.
        model.eval()
        v_mse, a6, a_open, v_d, nv = 0.0, 0, 0, 0.0, 0
        per_type = {}
        for d in val:
            ctx = d["ctx_hidden"].to(DEV)
            q = d["q_hidden"].to(DEV)
            h_all = d["h_inA_all"].to(DEV)
            y_out = d["h_outB"].to(DEV)
            tgt = SECRET_TO_IDX[d["secret"]]
            work = model.write_work(ctx)          # вне no_grad — граф клонов нужен
            with torch.no_grad():
                mv = model.mix_vector(q, work)
                h_inj = h_all.clone()
                h_inj[-1] = h_inj[-1] + mv
                h_out_p = last_layer_forward(h_inj)
                v_mse += lossf(h_out_p, y_out).item()
                c6 = cand_logits(h_out_p)
                a6 += c6.argmax(-1).item() == tgt
                lg = qwen_lmhead_logits(h_out_p)
                pred_open = lg.argmax(-1).item()
                ok = pred_open == SEC_TOK_VALUES[tgt]
                a_open += ok
                v_d += (mv - d["target_in"].to(DEV)).norm().item()
                nv += 1
                per_type.setdefault(d["type"], [0, 0])
                per_type[d["type"]][0] += ok
                per_type[d["type"]][1] += 1
        pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
        print(f"эпоха {ep:2d}: train {tot / n:.4f} | val_mse {v_mse / nv:.4f} | "
              f"acc6 {a6 / nv:.2f} | acc_open {a_open / nv:.2f} | "
              f"‖mix−Δ_in‖ {v_d / nv:.4f} | g={model.g().item():.3f} | {pt}", flush=True)

    torch.save(model.state_dict(), "exp1b_memory_injection.pt")
    print("Сохранено: exp1b_memory_injection.pt", flush=True)


if __name__ == "__main__":
    main()
