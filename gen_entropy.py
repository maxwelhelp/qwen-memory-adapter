# Добавляет в dataset_yattn.pt поле "entropy": энтропия предсказания Qwen на токенах
# контекста (удивление ОСНОВНОЙ СЕТИ). Контексты реконструируются из type/secret
# (тот же шаблон, что в gen_yattn).
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NOISE = ("Протокол сеанса подтверждён, шифрование установлено, сессия активна. "
         "Маршрут проверен, узлы синхронизированы, журнал обновлён. "
         "Аутентификация пройдена, доступ предоставлен, время ответа нормальное. ")
TYPES = {"Пароль": "Назови пароль одним словом. Ответ:", "Код": "Назови код одним словом. Ответ:",
         "Пин-код": "Назови пин-код одним словом. Ответ:", "Секрет": "Назови секрет одним словом. Ответ:"}

tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()

def ctx_token_pos(ids):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    idsl = ids[0].tolist()
    first = idsl.index(end)
    return idsl.index(end, first + 1) + 1

data = torch.load("dataset_yattn.pt")
print(f"примеров: {len(data)}", flush=True)
for i, d in enumerate(data):
    ctx = NOISE + f"{d['type']}: {d['secret']}. " + NOISE
    q = TYPES[d["type"]]
    msgs = [{"role": "user", "content": ctx}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                  add_generation_prompt=True).to(DEV)
    with torch.no_grad():
        out = qwen(ids)
    tq = ctx_token_pos(ids)
    logits = out.logits[0, :tq].float()
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    d["entropy"] = ent.cpu()
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(data)}", flush=True)
    del out, logits
    torch.cuda.empty_cache()

torch.save(data, "dataset_yattn.pt")
print("Сохранено (поле entropy добавлено)", flush=True)
