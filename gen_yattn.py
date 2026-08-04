# Шаг 3: датасет для дистилляции памяти.
# 4 типа задач (все dense-контроли 20/20): «Пароль/Код/Пин-код/Секрет: X» в мусоре.
# Вопросы РАЗНЫЕ — память обязана различать контексты.
# Два прогона: A (только вопрос) → h_inA, h_outA; B (контекст + вопрос) → h_outB, ctx_hidden.
# target Δ_norm = norm(h_outB) − norm(h_outA) — СДВИГ В НОРМАЛИЗОВАННОМ ПРОСТРАНСТВЕ.
# Применение (контроль 20/20): logits = lm_head(norm(h_outA) + mix), mix ≈ Δ_norm.
# (norm(h_no + mix) давил сигнал/градиент в 150 раз — масштаб Δ был ~150 при ||h||~15.)

import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
SECRETS = ["X7K9Q2", "M4N8Z1", "Q2P5R7", "T8W3V5", "R9L2X4", "Z6F3H8"]
TYPES = [  # (маркер в контексте, вопрос)
    ("Пароль", "Назови пароль одним словом. Ответ:"),
    ("Код", "Назови код одним словом. Ответ:"),
    ("Пин-код", "Назови пин-код одним словом. Ответ:"),
    ("Секрет", "Назови секрет одним словом. Ответ:"),
]
N_PER_TYPE = 100

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()

def ctx_token_pos(ids):
    """позиция конца ПЕРВОГО user-сообщения: после ВТОРОГО <|im_end|>
    (первый <|im_end|> закрывает system-сообщение шаблона)"""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ids = ids.tolist()
    first = ids.index(end)
    return ids.index(end, first + 1) + 1

def run(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                  add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return ids, out

def main():
    rng = random.Random(42)
    data = []
    for marker, q in TYPES:
        for _ in range(N_PER_TYPE):
            sec = rng.choice(SECRETS)
            ctx = NOISE + f"{marker}: {sec}. " + NOISE   # NOISE*1: dense-контроль 20/20

            idsA, outA = run([{"role": "user", "content": q}])
            h_inA_all = outA.hidden_states[-2][0].float()   # ВСЕ позиции вопроса (вход посл. слоя)
            h_inA = h_inA_all[-1]

            idsB, outB = run([{"role": "user", "content": ctx}, {"role": "user", "content": q}])
            h_outB = outB.hidden_states[-1][0, -1].float()
            h_inB_last = outB.hidden_states[-2][0, -1].float()   # вход посл. слоя, с контекстом
            tq = ctx_token_pos(idsB[0])
            ctx_hidden = outB.hidden_states[-2][0, :tq].float().cpu()

            # учитель η-головы: 1 для токенов секрета, 0 для мусора
            dec = tok.decode(idsB[0][:tq])
            needle = f"{marker}: {sec}"
            s0 = dec.find(needle)
            e0 = s0 + len(needle)
            mask = []
            acc = 0
            for i in range(tq):
                t = tok.decode([idsB[0][i].item()])
                mask.append(1.0 if acc < e0 and acc + len(t) > s0 else 0.0)
                acc += len(t)

            data.append({
                "ctx_hidden": ctx_hidden,
                "q_hidden": h_inA.cpu(),           # query чтения (последняя позиция)
                "h_inA_all": h_inA_all.cpu(),      # все позиции вопроса для прогона посл. слоя
                "target_in": (h_inB_last - h_inA).cpu(),  # Δ входа: mix должен его дать
                "h_outB": h_outB.cpu(),            # эталон выхода (для MSE/CE)
                "n_ctx": tq,
                "type": marker,
                "secret": sec,
                "mask_secret": torch.tensor(mask, dtype=torch.float32),  # для η-головы
            })
        print(f"собрано {marker}: {N_PER_TYPE}", flush=True)

    torch.save(data, "dataset_yattn.pt")
    n_tok = sum(d["n_ctx"] for d in data)
    print(f"Сохранено dataset_yattn.pt: {len(data)} примеров, {n_tok} токенов контекста, "
          f"||Δ_in||² mean = {torch.stack([d['target_in'] for d in data]).pow(2).mean().item():.5f}")

if __name__ == "__main__":
    main()
