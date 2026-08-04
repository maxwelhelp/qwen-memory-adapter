# Межсессионный тест exp3a (векторная инъекция, η из энтропии Qwen).
# Сессия 1: контекст → запись (η из энтропии, считается из logits прогона B).
# Сессия 2: вопрос → mix_vec → инъекция во вход последнего слоя → логпробы.
# Метрики: acc6 (6 кандидатов), open1 (argmax ВСЕГО словаря == 1-й токен секрета).
import torch, random, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from exp3a_vector_injection import LinSlotsAssoc
from train_memory_distill import last_layer_forward, cand_logits, qwen, SECRETS, DEV

NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
TYPES = [("Пароль", "Назови пароль одним словом. Ответ:"),
         ("Код", "Назови код одним словом. Ответ:"),
         ("Пин-код", "Назови пин-код одним словом. Ответ:"),
         ("Секрет", "Назови секрет одним словом. Ответ:")]
N_PER_TYPE = 10

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def run(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = qwen(ids, output_hidden_states=True)
    return ids, out

def ctx_pos(ids):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    idsl = ids[0].tolist()
    return idsl.index(end, idsl.index(end) + 1) + 1

def run_case(mem, ctx, q):
    idsB, outB = run([{"role": "user", "content": ctx}, {"role": "user", "content": q}])
    tq = ctx_pos(idsB)
    h_outB = outB.hidden_states[-1][0, -1].float()
    ctx_hidden = outB.hidden_states[-2][0, :tq].float()
    probs = F.softmax(outB.logits[0, :tq].float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    idsA, outA = run([{"role": "user", "content": q}])
    h_in_all = outA.hidden_states[-2][0].float()
    q_h = h_in_all[-1]
    work = mem.write_work(ctx_hidden, entropy)          # СЕССИЯ 1: запись
    with torch.no_grad():
        mv = mem.mix_vec(q_h, work)                     # СЕССИЯ 2: чтение → вектор
        h_inj = h_in_all.clone()
        h_inj[-1] = h_inj[-1] + mv                      # ИНЪЕКЦИЯ во вход последнего слоя
        h_out_p = last_layer_forward(h_inj)
        final_cand = cand_logits(h_out_p)
        lg = qwen.lm_head(h_out_p.unsqueeze(0).to(qwen.lm_head.weight.dtype))[0]
        pred_open = tok.decode([lg.argmax(-1).item()]).strip()
    p_mem = SECRETS[final_cand.argmax(-1).item()]
    p_ctx = SECRETS[cand_logits(h_outB).argmax(-1).item()]
    p_no = SECRETS[cand_logits(last_layer_forward(h_in_all)).argmax(-1).item()]
    return p_ctx, p_no, p_mem, pred_open

def main():
    mem = LinSlotsAssoc().to(DEV)
    mem.load_state_dict(torch.load("exp3a_vector_injection.pt"))
    rng = random.Random(7)
    tot = tot_ctx = tot_no = tot_open = 0
    for marker, q in TYPES:
        m = c = n = o = 0
        for i in range(N_PER_TYPE):
            sec = rng.choice(SECRETS)
            ctx = NOISE + f"{marker}: {sec}. " + NOISE
            p_ctx, p_no, p_mem, pred_open = run_case(mem, ctx, q)
            c += p_ctx == sec; n += p_no == sec; m += p_mem == sec
            o += pred_open == sec[0]
        print(f"=== {marker}: с контекстом {c}/{N_PER_TYPE} | с памятью {m}/{N_PER_TYPE} | без {n}/{N_PER_TYPE} | open1 {o}/{N_PER_TYPE}")
        tot += m; tot_ctx += c; tot_no += n; tot_open += o
    print(f"\nИТОГО: с памятью {tot}/{N_PER_TYPE*4} | с контекстом {tot_ctx}/{N_PER_TYPE*4} | без {tot_no}/{N_PER_TYPE*4} (шанс {N_PER_TYPE*4//6}) | open1 {tot_open}/{N_PER_TYPE*4}")

if __name__ == "__main__":
    main()
