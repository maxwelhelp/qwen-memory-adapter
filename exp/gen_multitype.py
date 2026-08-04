# Шаг 1 (AGENT_TASK): мульти-тайп датасет — 2-4 секрета ОДНОГО типа на контекст,
# каждый с referent'ом; вопрос указывает referent. Секреты — из 6 известных
# (совместимость с exp3a/exp5: acc6/CE валидны), привязка секрет→referent случайная.
# Формат полей как gen_yattn.py (+ поле referent). Сохраняет dataset_multitype.pt.
import torch, torch.nn.functional as F, random
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
SECRETS = ["X7K9Q2", "M4N8Z1", "Q2P5R7", "T8W3V5", "R9L2X4", "Z6F3H8"]
# (маркер типа, формат фразы «тип от referent: секрет», вопрос по referent'у)
TYPES = [
    ("Пароль", "Пароль от {ref}: {sec}.", "Какой пароль от {ref}? Ответ:"),
    ("Код", "Код доступа для {ref}: {sec}.", "Какой код для {ref}? Ответ:"),
    ("Пин-код", "Пин-код для {ref}: {sec}.", "Какой пин-код для {ref}? Ответ:"),
    ("Секрет", "Секрет для {ref}: {sec}.", "Какой секрет для {ref}? Ответ:"),
]
REFS = ["почта", "телефон", "ноутбук", "банк", "работа", "дом", "аккаунт", "сервер"]
N_PER_TYPE = 90          # 4 типа × 90 = 360 примеров
N_SEC_IN_CTX = 3         # секретов одного типа в контексте
MAX_CTX = 200

tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
SEC_TOK = {s: tok.encode(s, add_special_tokens=False)[0] for s in SECRETS}

def ctx_token_pos(ids):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    ids = ids.tolist()
    first = ids.index(end)
    return ids.index(end, first + 1) + 1

def run(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                  add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = qwen(ids, output_hidden_states=True)
    return ids, out

def main():
    rng = random.Random(55)
    data = []
    for marker, fmt, qfmt in TYPES:
        for _ in range(N_PER_TYPE):
            refs = rng.sample(REFS, N_SEC_IN_CTX)
            secs = rng.sample(SECRETS, N_SEC_IN_CTX)     # 3 разных секрета
            pairs = list(zip(refs, secs))
            rng.shuffle(pairs)                            # referent до/после секрета — случайно
            ph = [fmt.format(ref=r, sec=s) for r, s in pairs]
            rng.shuffle(ph)                               # порядок в контексте случайный
            ctx = NOISE + " ".join(ph) + "."
            q_ref, q_sec = pairs[0]                       # вопрос про ПЕРВУЮ пару (целевая)
            q = qfmt.format(ref=q_ref)

            idsA, outA = run([{"role": "user", "content": q}])
            h_inA_all = outA.hidden_states[-2][0].float()
            h_inA = h_inA_all[-1]

            idsB, outB = run([{"role": "user", "content": ctx}, {"role": "user", "content": q}])
            tq = ctx_token_pos(idsB[0])
            h_outB = outB.hidden_states[-1][0, -1].float()
            h_inB_last = outB.hidden_states[-2][0, -1].float()
            ctx_hidden = outB.hidden_states[-2][0, :tq].float().cpu()
            probs = F.softmax(outB.logits[0, :tq].float(), dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)

            # маска секретов (все секреты в контексте = «важное»)
            dec = tok.decode(idsB[0][:tq])
            mask = []
            acc = 0
            for i in range(tq):
                t = tok.decode([idsB[0][i].item()])
                hit = 0.0
                for r, s in pairs:
                    needle = fmt.format(ref=r, sec=s)
                    s0 = dec.find(needle)
                    if s0 >= 0 and acc < s0 + len(needle) and acc + len(t) > s0:
                        hit = 1.0
                        break
                mask.append(hit)
                acc += len(t)

            data.append({
                "ctx_hidden": ctx_hidden,
                "q_hidden": h_inA.cpu(),
                "h_inA_all": h_inA_all.cpu(),
                "target_in": (h_inB_last - h_inA).cpu(),
                "h_outB": h_outB.cpu(),
                "n_ctx": tq,
                "type": marker,
                "secret": q_sec,
                "referent": q_ref,
                "all_secrets": secs,          # все секреты этого примера (для confusion)
                "pairs": list(pairs),         # [(referent, secret)] — для confusion
                "mask_secret": torch.tensor(mask, dtype=torch.float32),
                "entropy": entropy.cpu(),
            })
            if len(data) % 60 == 0:
                print(f"собрано {len(data)} | пример: {marker}/{q_ref} → {q_sec}", flush=True)
    torch.save(data, "dataset_multitype.pt")
    n_ok = sum(1 for d in data if d["n_ctx"] < MAX_CTX)
    print(f"Сохранено dataset_multitype.pt: {len(data)} примеров | "
          f"n_ctx<{MAX_CTX}: {n_ok} | max n_ctx: {max(d['n_ctx'] for d in data)}", flush=True)

if __name__ == "__main__":
    main()
