# Эксперимент 1c (PLAN_MEMORY_V3, Ф1-вариант C): «УЧИТСЯ ЛИ ЗАПИСЬ» — bilevel-минимум.
# Titans-механика: запись — АРИФМЕТИКА без detach (θ ← θ(1−γ) − η·∇ℓ_inner),
# граф идёт через ВСЕ шаги записи (TTT-LM; у нас — create_graph=True).
# v5 делает detach-клоны — градиент внешнего лосса к η/θ₀ не доходил (7 прогонов).
# Минимальная проверка: 1 слот (Linear→GELU→Linear, 896), синтетические пары (k_i→v_i),
# нормы как у hidden Qwen (~15):
#   запись: S шагов самовосстановления ||M(k)−k||², η (обучаемый lr), γ (decay)
#   чтение: внешний лосс ||M(k)−v||² → backward через ВСЕ шаги
# Режимы: bilevel (граф через шаги) vs detach (как v5) — сравнение:
#   1) доходит ли градиент до η/θ₀; 2) падает ли лосс чтения с обучением.
# Вывод: bilevel обучает (лосс падает заметно ниже detach) → запись обучаема, полный
# bilevel TTT жизнеспособен; нет → фиксируем причину.

import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda"
D = 896
N_PAIRS = 16
S = 4            # шагов записи на чанк
LR = 1e-2
STEPS = 400


class Slot(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, D, bias=False))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)


def slot_fwd(x, W1, W2):
    return F.gelu(x @ W1.t()) @ W2.t()


def run(mode):
    torch.manual_seed(0)
    slot = Slot().to(DEV)
    eta_raw = nn.Parameter(torch.tensor(0.1, device=DEV))   # обучаемый lr записи
    gam_raw = nn.Parameter(torch.tensor(-2.0, device=DEV))  # γ = sigmoid ≈ 0.12
    opt = torch.optim.Adam(list(slot.parameters()) + [eta_raw, gam_raw], lr=LR)
    lossf = nn.MSELoss()

    ks = torch.randn(N_PAIRS, D, device=DEV) * 15
    vs = torch.randn(N_PAIRS, D, device=DEV) * 15

    print(f"=== режим {mode} ===", flush=True)
    for step in range(STEPS):
        opt.zero_grad()
        i = step % N_PAIRS
        k, v = ks[i], vs[i]
        eta = F.softplus(eta_raw)
        gam = torch.sigmoid(gam_raw)

        if mode == "bilevel":
            # запись: арифметика БЕЗ detach — W1/W2 становятся графовыми узлами,
            # граф идёт через η_raw, gam_raw и исходные θ₀
            W1, W2 = slot.net[0].weight, slot.net[2].weight
            for _ in range(S):
                iloss = lossf(slot_fwd(k, W1, W2), k)
                g1, g2 = torch.autograd.grad(iloss, [W1, W2], create_graph=True)
                W1 = W1 * (1 - gam) - eta * g1
                W2 = W2 * (1 - gam) - eta * g2
        else:
            # detach: как v5 — механика записи ВНЕ графа, η/γ/θ₀ не обучаются
            W1, W2 = slot.net[0].weight.detach(), slot.net[2].weight.detach()
            for _ in range(S):
                iloss = lossf(slot_fwd(k, W1, W2), k)
                g1, g2 = torch.autograd.grad(iloss, [W1, W2])
                W1 = W1 * (1 - gam.detach()) - eta.detach() * g1
                W2 = W2 * (1 - gam.detach()) - eta.detach() * g2

        loss = lossf(slot_fwd(k, W1, W2), v)   # чтение пары
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == STEPS - 1:
            with torch.no_grad():
                j = (i + 3) % N_PAIRS          # свежая пара (не текущая)
                kk, vv = ks[j], vs[j]
                W1d, W2d = slot.net[0].weight, slot.net[2].weight
                read_loss = lossf(slot_fwd(kk, W1d, W2d), vv).item()
                grad_eta = eta_raw.grad.abs().item() if eta_raw.grad is not None else 0.0
                grad_w1 = slot.net[0].weight.grad.abs().sum().item() if slot.net[0].weight.grad is not None else 0.0
            print(f"step {step:4d}: read_loss(текущая) {loss.item():.4f} | "
                  f"read_loss(свежая пара) {read_loss:.4f} | η={eta.item():.3f} | "
                  f"γ={gam.item():.3f} | |grad_η|={grad_eta:.2e} | |grad_W1|={grad_w1:.2e}",
                  flush=True)


if __name__ == "__main__":
    run("detach")     # контроль: как v5 — запись вне графа, обучать нечего
    print()
    run("bilevel")    # Titans-стиль: граф через все шаги записи
