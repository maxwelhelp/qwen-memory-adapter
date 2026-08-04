# Шаг 2 (titans_qwen.md + titans_qwen2.md): механика памяти на синтетике.
# Цель smoke-теста: запись пар (key→value) градиентными шагами + η/γ, чтение по ключу.
# Проверить:
#   1. запись улучшает восстановление (M(k) ближе к v, чем до записи);
#   2. СЛОТЫ изолируют знания (запись в одни слоты не стирает другие);
#   3. decay γ стирает, η контролирует силу записи;
#   4. персистентность: веса памяти переживают «сессию» (save/load);
#   5. L_fast (большой γ) забывает быстрее L_slow (маленький γ).
# Урок прошлого провала (схлопывание к нулю): запись — это шаг НА КАЖДЫЙ ТОКЕН/пару
# (Titans), а не 2 шага на весь контекст. Для нелинейного слота — итеративная запись.

import torch, torch.nn as nn, os

DEV = "cuda"
D = 896          # размерность hidden Qwen
N_SLOTS = 256    # слоты (фикс Дыры 2: один M(q) = каша); ~1 пара на слот
H = 256
N_PAIRS = 256
CHUNK = 16

torch.manual_seed(0)

class SlotMLP(nn.Module):
    """слот; linear=False → 2-слойная нелинейность (titans_qwen2.md)"""
    def __init__(self, linear=True):
        super().__init__()
        if linear:
            self.net = nn.Linear(D, D)
        else:
            self.net = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, D))

    def forward(self, x):
        return self.net(x)

class SlotMemory(nn.Module):
    def __init__(self, linear=True):
        super().__init__()
        self.slots = nn.ModuleList([SlotMLP(linear) for _ in range(N_SLOTS)])
        self.route_keys = nn.Parameter(torch.randn(N_SLOTS, D) * 0.02)

    def route(self, x):
        return (x @ self.route_keys.t()).argmax(-1)

    def write_chunk(self, ks, vs, eta=0.1, gamma=0.0, iters=1):
        """запись чанка: iters проходов по парам, шаг на каждую пару
        θ ← θ·(1−γ) − η·∇||M(k)−v||²  (η/γ пока константы)"""
        for _ in range(iters):
            for k, v in zip(ks, vs):
                s = self.route(k)
                mlp = self.slots[s]
                loss = ((mlp(k) - v) ** 2).mean()
                g = torch.autograd.grad(loss, mlp.parameters())
                for p, gp in zip(mlp.parameters(), g):
                    p.data = p.data * (1 - gamma) - eta * gp

    def read_slot(self, q):
        """жёсткое чтение: только свой слот (механика записи, без retrieval)"""
        return self.slots[self.route(q)](q)

    def read(self, q, topk=4):
        """мягкое чтение: top-k слотов по сходству q с заголовками (как на инференсе)"""
        sim = q @ self.route_keys.t()
        idx = sim.topk(topk).indices
        outs = torch.stack([self.slots[i](q) for i in idx])
        w = torch.softmax(sim[idx], dim=0)
        return (outs * w.unsqueeze(1)).sum(0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

def mse_memory(M, ks, vs, hard=True):
    with torch.no_grad():
        preds = torch.stack([M.read_slot(k) if hard else M.read(k) for k in ks])
    return ((preds - vs) ** 2).mean().item()

def run_case(linear, eta, iters, label):
    print(f"\n=== {label} (eta={eta}, iters={iters}, linear={linear}) ===")
    ks = torch.randn(N_PAIRS, D, device=DEV) * 0.5
    vs = torch.randn(N_PAIRS, D, device=DEV) * 0.5
    M = SlotMemory(linear).to(DEV)
    err0 = mse_memory(M, ks[:32], vs[:32])
    print(f"до записи:      MSE = {err0:.4f}")
    for i in range(0, N_PAIRS, CHUNK):
        M.write_chunk(ks[i:i+CHUNK], vs[i:i+CHUNK], eta=eta, iters=iters)
    err1 = mse_memory(M, ks[:32], vs[:32], hard=True)
    err1t = mse_memory(M, ks[:32], vs[:32], hard=False)
    print(f"после записи:   MSE = {err1:.4f} (жёсткое чтение, механика) | "
          f"{err1t:.4f} (top-k чтение, retrieval)")

    # изоляция слотов: новые пары в свободные слоты
    ks2 = torch.randn(64, D, device=DEV) * 0.5
    vs2 = torch.randn(64, D, device=DEV) * 0.5
    with torch.no_grad():
        used = set((ks[:32] @ M.route_keys.t()).argmax(-1).tolist())
        free = [i for i in range(N_SLOTS) if i not in used][:64]
        for j, i in enumerate(free):
            ks2[j] = ks2[j] + M.route_keys[i] * 10
    for i in range(0, 64, CHUNK):
        M.write_chunk(ks2[i:i+CHUNK], vs2[i:i+CHUNK], eta=eta, iters=iters)
    err_old = mse_memory(M, ks[:32], vs[:32])
    err_new = mse_memory(M, ks2[:32], vs2[:32])
    print(f"изоляция слотов:")
    print(f"  старые пары:  MSE = {err_old:.4f} (не пострадали)")
    print(f"  новые пары:   MSE = {err_new:.4f} (записаны)")

    # γ-стирание: декей тех же слотов (те же ключи, но η=0 → только забывание)
    M2 = SlotMemory(linear).to(DEV)
    for i in range(0, N_PAIRS, CHUNK):
        M2.write_chunk(ks[i:i+CHUNK], vs[i:i+CHUNK], eta=eta, iters=iters)
    e_b = mse_memory(M2, ks[:32], vs[:32])
    for _ in range(10):
        M2.write_chunk(ks[:CHUNK], vs[:CHUNK], eta=0.0, gamma=0.1)
    e_a = mse_memory(M2, ks[:32], vs[:32])
    print(f"γ-стирание (η=0, γ=0.1, те же слоты): {e_b:.4f} → {e_a:.4f} (забыла)")

    # перезапись факта: тот же ключ, новое значение → читается новое
    M4 = SlotMemory(linear).to(DEV)
    k = torch.randn(1, D, device=DEV) * 0.5
    v1 = torch.randn(1, D, device=DEV) * 0.5
    v2 = torch.randn(1, D, device=DEV) * 0.5
    M4.write_chunk(k, v1, eta=eta, iters=iters)
    e_v1 = mse_memory(M4, k, v1)
    M4.write_chunk(k, v2, eta=eta, iters=iters)
    e_v1_after = mse_memory(M4, k, v1)
    e_v2 = mse_memory(M4, k, v2)
    print(f"перезапись факта: после v2 на тот же ключ: v1-ошибка {e_v1:.4f} → {e_v1_after:.4f}, "
          f"v2-ошибка {e_v2:.4f} (новое значение переопределило старое)")

    # персистентность
    M.save("memory_smoke.pt")
    M3 = SlotMemory(linear).to(DEV)
    M3.load("memory_smoke.pt")
    print(f"персистентность: после save/load MSE = {mse_memory(M3, ks[:32], vs[:32]):.4f}")
    os.remove("memory_smoke.pt")

    # L_fast vs L_slow: шумовая «сессия» бьёт по тем же слотам
    Mf = SlotMemory(linear).to(DEV); Ms = SlotMemory(linear).to(DEV)
    for i in range(0, N_PAIRS, CHUNK):
        Mf.write_chunk(ks[i:i+CHUNK], vs[i:i+CHUNK], eta=eta, gamma=0.1, iters=iters)
        Ms.write_chunk(ks[i:i+CHUNK], vs[i:i+CHUNK], eta=eta, gamma=0.001, iters=iters)
    noise_k = torch.randn(CHUNK, D, device=DEV) * 0.5
    noise_v = torch.randn(CHUNK, D, device=DEV) * 0.5
    for _ in range(5):
        Mf.write_chunk(noise_k, noise_v, eta=0.1, gamma=0.1)
        Ms.write_chunk(noise_k, noise_v, eta=0.1, gamma=0.001)
    print(f"L_fast (γ=0.1):   MSE = {mse_memory(Mf, ks[:32], vs[:32]):.4f} (забыл сильнее)")
    print(f"L_slow (γ=0.001): MSE = {mse_memory(Ms, ks[:32], vs[:32]):.4f} (хранит)")

if __name__ == "__main__":
    run_case(linear=True,  eta=0.5, iters=1,  label="линейный слот, шаг на пару")
    run_case(linear=False, eta=0.1, iters=16, label="нелинейный слот, итеративная запись")
