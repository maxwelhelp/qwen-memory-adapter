# Диагностика: содержит ли Δ = h_out_ctx − h_out_noctx правильную информацию?
# Контроль: первый токен из
#   (1) h_out_ctx (с контекстом) — эталон;
#   (2) h_out_noctx + ИСТИННЫЙ Δ — идеальное подмешивание;
#   (3) h_out_noctx — без контекста.
# Если (2) ≈ (1) — путь корректен, проблема в выученной памяти.
# Если (2) ≈ (3) — Δ не содержит знания в нужной форме — менять подход.

import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
PASSWORDS = ["X7K9Q2", "M4N8Z1", "Q2P5R7", "T8W3V5", "R9L2X4", "Z6F3H8"]
N = 20

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()

def run(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                  add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return ids, out

def first_token(h):
    logits = model.lm_head(model.model.norm(h.to(model.lm_head.weight.dtype)))
    return logits.argmax(-1).item()

def main():
    rng = random.Random(7)
    ok_ctx, ok_delta, ok_noctx = 0, 0, 0
    for i in range(N):
        pw = rng.choice(PASSWORDS)
        ctx = NOISE * 2 + f"Пароль: {pw}. " + NOISE * 2
        q = "Какой пароль указан в тексте?"
        target = tok.encode(pw)[0]

        _, outB = run([{"role": "user", "content": ctx}, {"role": "user", "content": q}])
        h_ctx = outB.hidden_states[-1][0, -1].float()
        _, outA = run([{"role": "user", "content": q}])
        h_noctx = outA.hidden_states[-1][0, -1].float()

        delta = h_ctx - h_noctx
        t1 = first_token(h_ctx)
        t2 = first_token(h_noctx + delta)
        t3 = first_token(h_noctx)
        ok_ctx += t1 == target
        ok_delta += t2 == target
        ok_noctx += t3 == target
        print(f"  {i}: {pw} | с контекстом: {tok.decode([t1])!r:>6} "
              f"| noctx+Δ: {tok.decode([t2])!r:>6} | без: {tok.decode([t3])!r:>6} "
              f"| ||Δ||={delta.norm():.2f}")
    print(f"\nс контекстом: {ok_ctx}/{N} | noctx+ИСТИННЫЙ Δ: {ok_delta}/{N} | без: {ok_noctx}/{N}")

if __name__ == "__main__":
    main()
