# Шаг 4 (memory_layer_v2.md 4c): МЕЖСЕССИОННЫЙ тест.
# Сессия 1: контекст (секрет в мусоре) → память пишет.
# Сессия 2: только вопрос → forward без контекста → последняя позиция:
#   h'_in[-1] = h_in[-1] + g·W_out(M(q) − M_θ₀(q));  h_out' = last_layer(h'_in)[-1]
# Метрика: кандидат с макс. логпробом первого токена (как eval_eviction).

import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_memory_distill import Memory, last_layer_forward, cand_logits

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
SECRETS = ["X7K9Q2", "M4N8Z1", "Q2P5R7", "T8W3V5", "R9L2X4", "Z6F3H8"]
TYPES = [("Пароль", "Назови пароль одним словом. Ответ:"),
         ("Код", "Назови код одним словом. Ответ:"),
         ("Пин-код", "Назови пин-код одним словом. Ответ:"),
         ("Секрет", "Назови секрет одним словом. Ответ:")]
N_PER_TYPE = 10

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
SEC_TOK = {p: tok.encode(p, add_special_tokens=False)[0] for p in SECRETS}

def run(msgs):
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                  add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return ids, out

def ctx_pos(ids):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    idsl = ids[0].tolist()
    return idsl.index(end, idsl.index(end) + 1) + 1

def pred_from_h(h_out):
    cl = cand_logits(h_out)   # [6] — логпробы кандидатов в порядке SECRETS
    return max(SECRETS, key=lambda p: cl[SECRETS.index(p)].item())

def run_case(mem, ctx, q):
    idsB, outB = run([{"role": "user", "content": ctx}, {"role": "user", "content": q}])
    h_outB = outB.hidden_states[-1][0, -1].float()
    ctx_hidden = outB.hidden_states[-2][0, :ctx_pos(idsB)].float()

    idsA, outA = run([{"role": "user", "content": q}])
    h_in_all = outA.hidden_states[-2][0].float()
    q_h = h_in_all[-1]

    p_ctx = pred_from_h(h_outB)
    p_no = pred_from_h(last_layer_forward(h_in_all))

    work = mem.write_work(ctx_hidden)
    with torch.no_grad():
        base_cand = cand_logits(last_layer_forward(h_in_all))
        mix = mem.mix(q_h, work)
    final_cand = base_cand + mix
    p_mem = max(SECRETS, key=lambda p: final_cand[SECRETS.index(p)].item())
    return p_ctx, p_no, p_mem

def main():
    mem = Memory().to(DEV)
    mem.load_state_dict(torch.load("memory_distill.pt"))
    mem.refresh_theta_stacks()   # кэш θ₀-стеков для батч-чтения
    rng = random.Random(7)

    for marker, q in TYPES:
        c, m, n = 0, 0, 0
        print(f"=== {marker.upper()} ===")
        for i in range(N_PER_TYPE):
            sec = rng.choice(SECRETS)
            ctx = NOISE + f"{marker}: {sec}. " + NOISE
            p_ctx, p_no, p_mem = run_case(mem, ctx, q)
            c += p_ctx == sec; m += p_mem == sec; n += p_no == sec
            print(f"  {i}: {sec} | с контекстом: {p_ctx:>7} | с памятью: {p_mem:>7} | без: {p_no:>7}")
        print(f"с контекстом {c}/{N_PER_TYPE} | с памятью {m}/{N_PER_TYPE} | без {n}/{N_PER_TYPE}\n")

if __name__ == "__main__":
    main()
