# Шаг 3: абляция памяти на val + честная оценка.
# Три уровня:
#   1. baseline      — без подмешивания (||Δ||²);
#   2. пустая память — подмешивание из θ₀ (слоты НЕ писали контекст);
#   3. с записью      — память написала контекст и читает по вопросу.
# Если (3) ≈ (2) — запись не нужна, дистилляция выучила только отображение «вопрос→Δ».

import torch, torch.nn as nn

DEV = "cuda"
D = 896
N_SLOTS = 32
ETA = 0.3
GAMMA = 0.0
TOPK = 4
MAX_CTX = 70
CREATE_LAST = 8

from train_memory_distill import Memory, MAX_CTX as _mc  # переиспользуем класс

def main():
    data = torch.load("dataset_yattn.pt")
    ex = [d for d in data if d["n_ctx"] < MAX_CTX]
    import random
    rng = random.Random(0)
    rng.shuffle(ex)
    val = ex[: len(ex) // 10]

    model = Memory().to(DEV)
    model.load_state_dict(torch.load("memory_distill.pt"))
    lossf = nn.MSELoss()

    b_base, b_empty, b_full = 0.0, 0.0, 0.0
    for d in val:
        q = d["q_hidden"].to(DEV)
        y = d["target"].to(DEV)
        b_base += lossf(torch.zeros_like(y), y).item()
        # пустая память: слоты = θ₀ (запись не выполнялась)
        out_empty = model.mix(q, {})
        b_empty += lossf(out_empty, y).item()
        # с записью
        ctx = d["ctx_hidden"].to(DEV)
        work = model.write_work(ctx)
        out_full = model.mix(q, work)
        b_full += lossf(out_full, y).item()
    n = len(val)
    b_base, b_empty, b_full = b_base / n, b_empty / n, b_full / n
    print(f"baseline      (без памяти):       {b_base:.4f}  (norm {b_base ** 0.5:.3f})")
    print(f"пустая память (θ₀, без записи):   {b_empty:.4f}  (rel {b_empty / b_base:.3f})")
    print(f"с записью      (контекст записан): {b_full:.4f}  (rel {b_full / b_base:.3f})")
    print(f"выигрыш записи над пустой: {b_empty / b_full:.2f}× "
          f"({(1 - b_full / b_empty) * 100:.1f}% ошибки снято)")

if __name__ == "__main__":
    main()
