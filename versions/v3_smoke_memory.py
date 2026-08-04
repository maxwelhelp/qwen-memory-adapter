# Smoke-тест готового NeuralMemory из titans-pytorch (без изменений кода).
# Проверка: память записывает (k,v)-пары и читает по ключу — ошибка чтения
# ДО записи должна быть заметно хуже, чем ПОСЛЕ.

import torch
from titans_pytorch import NeuralMemory

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 64
N = 128          # число (k,v)-пар
CHUNK = 16

# --- данные: ключ -> значение (ассоциативная память, КОРРЕЛИРОВАННЫЕ пары) ---
seq = torch.randn(1, N, DIM, device=DEV)   # (B, N, D) — 3D обязательно
A = torch.randn(DIM, DIM, device=DEV) / (DIM ** 0.5)
vals = seq @ A.T                            # v = A·k — структура, которую можно выучить

# --- память: минимальная конфигурация Titans (готовый модуль) ---
mem = NeuralMemory(
    dim=DIM,
    chunk_size=CHUNK,          # запись по чанкам
    batch_size=32,
    model=None,                # дефолтный MemoryMLP (2 слоя)
).to(DEV)

# --- ДО записи: чтение из пустой (неинициализированной) памяти ---
with torch.no_grad():
    out_before = mem.retrieve_memories(seq, mem.init_weights(1))
mse_before = (out_before - vals).pow(2).mean().item()

# --- 5 проходов записи+чтения (state пробрасывается между проходами) ---
state = None
mse_curve = [mse_before]
for i in range(1, 6):
    retrieved, state, _ = mem(seq, state=state, return_surprises=True)
    mse_curve.append((retrieved - vals).pow(2).mean().item())

print(f"device={DEV}  dim={DIM}  pairs={N}  chunk={CHUNK}")
print(f"MSE чтения:  до записи   = {mse_curve[0]:.4f}")
for i, m in enumerate(mse_curve[1:], 1):
    print(f"             проход {i}    = {m:.4f}")
print(f"снижение MSE за 5 проходов: {(1 - mse_curve[-1]/mse_curve[0])*100:.1f}%")

ok = mse_curve[-1] < mse_curve[0] * 0.5
print(f"РЕЗУЛЬТАТ: {'OK — память усваивает (k,v)' if ok else 'FAIL — память не учится'}")
