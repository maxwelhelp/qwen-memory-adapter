# Межсессионный ОТКРЫТЫЙ тест exp3a: случайные секреты (не из 6 кандидатов!).
# Сессия 1: контекст → запись (η из энтропии Qwen).
# Сессия 2: вопрос → mix_vec → инъекция во вход последнего слоя → ГЕНЕРАЦИЯ (greedy).
# Метрики: exact match полного секрета, open1 (1-й токен), сравнение с dense-контролем.
import torch, random, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from exp3a_vector_injection import LinSlotsAssoc
from train_memory_distill import last_layer_forward, qwen, DEV

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW = 8

def run(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = qwen(ids, output_hidden_states=True)
    return ids, out

def ctx_pos(ids):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    idsl = ids[0].tolist()
    return idsl.index(end, idsl.index(end) + 1) + 1

def run_case(mem, d):
    ctx_text = d["ctx_text"]
    q = d["q_text"]
    idsB, outB = run([{"role": "user", "content": ctx_text}, {"role": "user", "content": q}])
    tq = ctx_pos(idsB)
    ctx_hidden = outB.hidden_states[-2][0, :tq].float()
    probs = F.softmax(outB.logits[0, :tq].float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    idsA, outA = run([{"role": "user", "content": q}])
    h_in_all = outA.hidden_states[-2][0].float()
    q_h = h_in_all[-1]
    work = mem.write_work(ctx_hidden, entropy)           # СЕССИЯ 1: запись
    with torch.no_grad():
        mv = mem.mix_vec(q_h, work)                      # СЕССИЯ 2: вектор
        # ГЕНЕРАЦИЯ с инъекцией на каждой позиции
        gen = []
        cur_ids = idsA.clone()
        for _ in range(MAX_NEW):
            with torch.no_grad():
                out = qwen(cur_ids, output_hidden_states=True)
                h_in = out.hidden_states[-2][0].float()
                h_in[-1] = h_in[-1] + mv                 # инъекция памяти
                h_out_p = last_layer_forward(h_in)
                lg = qwen.lm_head(h_out_p.unsqueeze(0).to(qwen.lm_head.weight.dtype))[0]
            nt = lg.argmax(-1).item()
            if nt == tok.eos_token_id:
                break
            gen.append(nt)
            cur_ids = torch.cat([cur_ids, torch.tensor([[nt]], device=DEV)], dim=1)
    gen_text = tok.decode(gen, skip_special_tokens=True).strip().replace(" ", "")
    return gen_text

def main():
    mem = LinSlotsAssoc().to(DEV)
    mem.load_state_dict(torch.load("exp3a_vector_injection.pt"))
    data = torch.load("dataset_yattn_open.pt")
    # реконструкция текста контекста/вопроса (как в генераторе)
    TYPES = {"Пароль": "Назови пароль одним словом. Ответ:", "Код": "Назови код одним словом. Ответ:",
             "Пин-код": "Назови пин-код одним словом. Ответ:", "Секрет": "Назови секрет одним словом. Ответ:"}
    NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
             "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
             "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
    tot = open1 = 0
    per_type = {}
    for d in data:
        d["ctx_text"] = NOISE + f"{d['type']}: {d['secret']}. " + NOISE
        d["q_text"] = TYPES[d["type"]]
        gen_text = run_case(mem, d)
        ok = gen_text == d["secret"]
        ok1 = gen_text[:1] == d["secret"][:1]
        tot += ok; open1 += ok1
        per_type.setdefault(d["type"], [0, 0])
        per_type[d["type"]][0] += ok
        per_type[d["type"]][1] += 1
        print(f"  {d['type']}: секрет {d['secret']} | память: {gen_text!r} | {'OK' if ok else 'нет'}")
    pt = " ".join(f"{k}:{c[0]}/{c[1]}" for k, c in per_type.items())
    print(f"\nИТОГО: exact {tot}/{len(data)} ({tot/len(data)*100:.0f}%) | open1 {open1}/{len(data)} | "
          f"dense-эталон 34/40 (85%) | {pt}")

if __name__ == "__main__":
    main()
