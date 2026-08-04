# Генератор ОТКРЫТОГО датасета для теста векторной инъекции (exp3a).
# Секреты — СЛУЧАЙНЫЕ строки (не из 6 известных кандидатов!): память должна
# воспроизвести произвольный секрет, которого нет в закрытом списке.
# Поля как в dataset_yattn + entropy (удивление Qwen, считается сразу).
# dense-контроль: с контекстом модель должна воспроизводить секрет (голова генерации).
import torch, torch.nn.functional as F, random, string
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
TYPES = [("Пароль", "Назови пароль одним словом. Ответ:"),
         ("Код", "Назови код одним словом. Ответ:"),
         ("Пин-код", "Назови пин-код одним словом. Ответ:"),
         ("Секрет", "Назови секрет одним словом. Ответ:")]
N_PER_TYPE = 10
SEC_LEN = 6          # длина случайного секрета
ALPH = string.ascii_uppercase + string.digits

tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()

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

def gen_secret(rng):
    return "".join(rng.choice(ALPH) for _ in range(SEC_LEN))

def main():
    rng = random.Random(123)
    data = []
    dense_ok = 0
    for marker, q in TYPES:
        for _ in range(N_PER_TYPE):
            sec = gen_secret(rng)                       # СЛУЧАЙНЫЙ секрет (открытый словарь)
            ctx = NOISE + f"{marker}: {sec}. " + NOISE
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
            # dense-контроль: генерация с контекстом (greedy, 8 токенов)
            gen_ids = qwen.generate(idsB, max_new_tokens=8, do_sample=False)
            gen_text = tok.decode(gen_ids[0][idsB.shape[1]:], skip_special_tokens=True).strip()
            dense_ok += gen_text.replace(" ", "") == sec
            # маска секрета (для контроля)
            dec = tok.decode(idsB[0][:tq])
            needle = f"{marker}: {sec}"
            s0 = dec.find(needle); e0 = s0 + len(needle)
            mask, acc = [], 0
            for i in range(tq):
                t = tok.decode([idsB[0][i].item()])
                mask.append(1.0 if acc < e0 and acc + len(t) > s0 else 0.0)
                acc += len(t)
            data.append({
                "ctx_hidden": ctx_hidden,
                "q_hidden": h_inA.cpu(),
                "h_inA_all": h_inA_all.cpu(),
                "target_in": (h_inB_last - h_inA).cpu(),
                "h_outB": h_outB.cpu(),
                "n_ctx": tq,
                "type": marker,
                "secret": sec,
                "mask_secret": torch.tensor(mask, dtype=torch.float32),
                "entropy": entropy.cpu(),
            })
            if len(data) % 10 == 0:
                print(f"собрано {len(data)} | dense-контроль {dense_ok}/{len(data)} "
                      f"(сек: {sec}, gen: {gen_text[:12]!r})", flush=True)
    torch.save(data, "dataset_yattn_open.pt")
    print(f"Сохранено dataset_yattn_open.pt: {len(data)} примеров | "
          f"dense-контроль {dense_ok}/{len(data)} ({dense_ok/len(data)*100:.0f}%)", flush=True)

if __name__ == "__main__":
    main()
