# Эксперимент 2c: ОТДЕЛЬНЫЙ ТЕСТ дистилляции эмбеддингов в MLP — «ОДНО ПРОСТРАНСТВО?»
# Вопрос: достаточно ли наша дистилляция (M(h)≈h) выравнивает пространство памяти
# с пространством Qwen? Метрики на НЕВИДАННЫХ векторах: cos-sim(M(x), x), rel_err.
#   hidden (Б): контекстные hidden Qwen (вход последнего слоя) — из dataset_yattn.pt,
#               без загрузки Qwen — можно параллельно с другими прогонами.
#   emb (А):    статичные эмбеддинги токенов (embed_tokens) — нужна Qwen.
# Ablation: случайный MLP (без дистилляции) — ожидаем cos≈0.
# Механика: запись пары (k→v) в выровненный MLP, чтение по зашумлённому k'≈k —
#           работает ли чтение в «одном пространстве».

import torch, torch.nn as nn, torch.nn.functional as F, random, sys

D = 896
N_TRAIN = 320
BATCH_TOK = 16
LR = 3e-3


class MemMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


def load_data(mode):
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < 200]
    rng = random.Random(0)
    rng.shuffle(ex)
    n_val = len(ex) // 10
    train, val = ex[n_val:][:N_TRAIN], ex[:n_val]
    if mode == "hidden":
        Xtr = torch.cat([d["ctx_hidden"] for d in train])      # [~55k, 896]
        Xva = torch.cat([d["ctx_hidden"] for d in val])
    else:  # emb — см. main() части А (загружает Qwen)
        from train_memory_distill import tok, qwen
        ids = [tok.apply_chat_template([{"role": "user", "content": d["ctx_text"]}],
                                       tokenize=True, return_tensors="pt") for d in train]
        # контекстный текст в датасете не хранится — генерируем заново тем же шаблоном
        NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
                 "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
                 "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
        TYPES = {"Пароль": "Назови пароль одним словом. Ответ:", "Код": "Назови код одним словом. Ответ:",
                 "Пин-код": "Назови пин-код одним словом. Ответ:", "Секрет": "Назови секрет одним словом. Ответ:"}
        SECRETS = ["X7K9Q2", "M4N8Z1", "Q2P5R7", "T8W3V5", "R9L2X4", "Z6F3H8"]
        rng2 = random.Random(42)
        Xtr, Xva = [], []
        for d in train:
            sec = d["secret"]
            marker = d["type"]
            ctx = NOISE + f"{marker}: {sec}. " + NOISE
            ids = tok(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                Xtr.append(qwen.model.embed_tokens(ids.to("cuda"))[0].float())
        for d in val:
            marker, sec = d["type"], d["secret"]
            ctx = NOISE + f"{marker}: {sec}. " + NOISE
            ids = tok(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                Xva.append(qwen.model.embed_tokens(ids.to("cuda"))[0].float())
        Xtr = torch.cat(Xtr).cpu(); Xva = torch.cat(Xva).cpu()
    return Xtr, Xva


def metrics(X, M, name):
    with torch.no_grad():
        Xb = X.to("cuda")
        outs = []
        for i in range(0, len(Xb), 256):
            outs.append(M(Xb[i:i + 256]))
        Mo = torch.cat(outs)
        cos = F.cosine_similarity(Mo, Xb, dim=-1)
        rel = (Mo - Xb).norm(dim=-1) / Xb.norm(dim=-1)
    print(f"  {name}: cos(M(x), x) = {cos.mean().item():.4f} ± {cos.std().item():.4f} | "
          f"rel_err = {rel.mean().item():.4f} | ‖M(x)‖ = {Mo.norm(dim=-1).mean().item():.1f} "
          f"(‖x‖ = {Xb.norm(dim=-1).mean().item():.1f})", flush=True)


def distill(Xtr, Xva, steps, tag):
    torch.manual_seed(0)
    M = MemMLP().to("cuda")
    opt = torch.optim.Adam(M.parameters(), lr=LR)
    lossf = nn.MSELoss()
    for st in range(steps):
        opt.zero_grad()
        idx = torch.randint(0, len(Xtr), (BATCH_TOK,))
        x = Xtr[idx].to("cuda")
        loss = lossf(M(x), x)
        loss.backward()
        opt.step()
        if st % 400 == 399:
            print(f"  [{tag}] step {st}: loss {loss.item():.4f}", flush=True)
    print(f"=== {tag}: ПОСЛЕ дистилляции ===", flush=True)
    metrics(Xva, M, "val (невиданные)")
    return M


def mechanic_test(M, tag):
    """запись пары (k→v) в выровненный MLP, чтение по зашумлённому k' — работает ли"""
    torch.manual_seed(1)
    data = torch.load("dataset_yattn.pt")
    ctx = data[0]["ctx_hidden"].to("cuda")
    k = ctx[5]; v = ctx[10]
    eta, gam, iters = 0.3, 0.01, 8
    W1, W2 = M.net[0].weight, M.net[2].weight
    def fwd(x, w1, w2):
        return F.gelu(x @ w1.t()) @ w2.t()
    params = {n: p.detach().clone().requires_grad_(True) for n, p in M.named_parameters()}
    for _ in range(iters):
        pred = fwd(k, params["net.0.weight"], params["net.2.weight"])
        iloss = F.mse_loss(pred, v)
        g1, g2 = torch.autograd.grad(iloss, [params["net.0.weight"], params["net.2.weight"]])
        g1 = g1 / (g1.norm() + 1e-8); g2 = g2 / (g2.norm() + 1e-8)
        # БЕЗ no_grad: параметры остаются графовыми узлами (requires_grad=True)
        params["net.0.weight"] = params["net.0.weight"] * (1 - gam) - eta * g1
        params["net.2.weight"] = params["net.2.weight"] * (1 - gam) - eta * g2
    with torch.no_grad():
        for noise in (0.0, 0.05, 0.2):
            kq = k + torch.randn_like(k) * k.norm() * noise
            out = fwd(kq, params["net.0.weight"], params["net.2.weight"])
            cos = F.cosine_similarity(out, v, dim=0).item()
            print(f"  [{tag}] чтение по k' (шум {noise:.0%}): cos(выдача, v) = {cos:.4f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "hidden"
    Xtr, Xva = load_data(mode)
    print(f"режим {mode}: train {len(Xtr)} токенов | val {len(Xva)} токенов", flush=True)

    # 1) ablation: случайный MLP без дистилляции
    print("=== ablation: случайный MLP (БЕЗ дистилляции) ===", flush=True)
    torch.manual_seed(0)
    M0 = MemMLP().to("cuda")
    metrics(Xva, M0, "val (случайный)")

    # 2) дистилляция
    M = distill(Xtr, Xva, steps=2400, tag="MSE(M(x),x)")

    # 3) механика чтения в выровненном пространстве
    mechanic_test(M, "выровненный")
    mechanic_test(M0, "случайный")
    print("Готово.", flush=True)
