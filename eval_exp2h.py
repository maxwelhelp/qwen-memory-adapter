# Межсессионный тест для exp2h (линейные ассоциативные слоты).
# Сессия 1: контекст (секрет в мусоре) → память пишет.
# Сессия 2: только вопрос → forward без контекста → mix → argmax по 6 кандидатам.
# Свежие примеры (rng=7, 10 на тип) — НЕ из датасета (как eval_cross_session).
import torch, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from exp2h_linear_assoc_eta import LinSlotsAssoc
from train_memory_distill import last_layer_forward, cand_logits, SECRETS, SEC_TOK_VALUES, SECRET_TO_IDX, DEV

NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
TYPES = [("Пароль", "Назови пароль одним словом. Ответ:"),
         ("Код", "Назови код одним словом. Ответ:"),
         ("Пин-код", "Назови пин-код одним словом. Ответ:"),
         ("Секрет", "Назови секрет одним словом. Ответ:")]
N_PER_TYPE = 10

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16).to(DEV).eval()

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
    h_outB = outB.hidden_states[-1][0, -1].float()
    ctx_hidden = outB.hidden_states[-2][0, :ctx_pos(idsB)].float()
    idsA, outA = run([{"role": "user", "content": q}])
    h_in_all = outA.hidden_states[-2][0].float()
    q_h = h_in_all[-1]
    work = mem.write_work(ctx_hidden)          # СЕССИЯ 1: запись
    with torch.no_grad():
        base_cand = cand_logits(last_layer_forward(h_in_all))
        mv = mem.mix6(q_h, work)                # СЕССИЯ 2: чтение по вопросу
    final_cand = base_cand + mv.to(base_cand.dtype)
    p_mem = SECRETS[final_cand.argmax(-1).item()]
    p_ctx = SECRETS[cand_logits(h_outB).argmax(-1).item()]
    p_no = SECRETS[base_cand.argmax(-1).item()]
    return p_ctx, p_no, p_mem

def main():
    mem = LinSlotsAssoc().to(DEV)
    mem.load_state_dict(torch.load("exp2h_linear_assoc_eta.pt"))
    rng = random.Random(7)
    tot = tot_ctx = tot_no = 0
    for marker, q in TYPES:
        m = c = n = 0
        for i in range(N_PER_TYPE):
            sec = rng.choice(SECRETS)
            ctx = NOISE + f"{marker}: {sec}. " + NOISE
            p_ctx, p_no, p_mem = run_case(mem, ctx, q)
            c += p_ctx == sec; n += p_no == sec; m += p_mem == sec
            print(f"  {i}: {sec} | с контекстом: {p_ctx:>7} | с памятью: {p_mem:>7} | без: {p_no:>7}")
        print(f"=== {marker}: с контекстом {c}/{N_PER_TYPE} | с памятью {m}/{N_PER_TYPE} | без {n}/{N_PER_TYPE}")
        tot += m; tot_ctx += c; tot_no += n
    print(f"\nИТОГО: с памятью {tot}/{N_PER_TYPE*4} | с контекстом {tot_ctx}/{N_PER_TYPE*4} | без {tot_no}/{N_PER_TYPE*4} (шанс {N_PER_TYPE*4//6})")

if __name__ == "__main__":
    main()
