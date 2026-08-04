# Эксперимент 2m: УДИВЛЕНИЕ ОСНОВНОЙ СЕТИ (Qwen) — различает ли секрет/мусор, и как
# комбинируется с MLP-удивлением (‖M(h)−h‖², AUC 0.89).
# Сигналы Qwen (бесплатно при forward):
#   nll_t  = −log p(x_{t+1} | x_≤t)   — учитель-форсинг на true-токене
#   ent_t  = H(softmax(logits_t))     — энтропия предсказания
# Метрика: AUC различения секрет/мусор (mask_secret), сравнение с MLP-surprise,
# комбинация: max(norm_qwen, norm_mlp) и сумма.
import torch, torch.nn.functional as F, random
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

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
N = 40

tok = AutoTokenizer.from_pretrained(MODEL)
qwen = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()

def ctx_token_pos(ids):
    end = tok.convert_tokens_to_ids("<|im_end|>")
    idsl = ids[0].tolist()
    first = idsl.index(end)
    return idsl.index(end, first + 1) + 1

def mask_for(ids, tq, needle):
    dec = tok.decode(ids[0][:tq])
    s0 = dec.find(needle); e0 = s0 + len(needle)
    mask, acc = [], 0
    for i in range(tq):
        t = tok.decode([ids[0][i].item()])
        mask.append(1.0 if acc < e0 and acc + len(t) > s0 else 0.0)
        acc += len(t)
    return torch.tensor(mask)

rng = random.Random(42)
nlls, ents, masks = [], [], []
for marker, q in TYPES:
    for _ in range(N // 4):
        sec = rng.choice(SECRETS)
        ctx = NOISE + f"{marker}: {sec}. " + NOISE
        msgs = [{"role": "user", "content": ctx}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt",
                                      add_generation_prompt=True).to(DEV)
        with torch.no_grad():
            out = qwen(ids)
        tq = ctx_token_pos(ids)
        logits = out.logits[0, :tq]                     # [tq, V] — предсказания по контексту
        probs = F.softmax(logits.float(), dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)         # энтропия
        nll = F.cross_entropy(logits.float(), ids[0][1:tq + 1], reduction='none')  # NLL true
        m = mask_for(ids, tq, f"{marker}: {sec}")[:tq]
        nlls.append(nll.cpu()); ents.append(ent.cpu()); masks.append(m.cpu())
        del out, logits
        torch.cuda.empty_cache()

nll = torch.cat(nlls); ent = torch.cat(ents); m = torch.cat(masks)
print(f"токены: секрет {(m == 1).sum()} | мусор {(m == 0).sum()}")
for name, sc in (("nll(true-токен)", nll), ("энтропия", ent)):
    auc = roc_auc_score(m.numpy(), sc.numpy())
    print(f"  Qwen-удивление [{name}]: AUC = {auc:.4f} | секрет {sc[m == 1].mean():.4f} vs мусор {sc[m == 0].mean():.4f}")
# комбинация с MLP-удивлением: нужно MLP-surprise — нормируем и объединяем (макс/сумма)
print("(MLP-удивление ‖M(h)−h‖²: AUC 0.89 — exp2f, берётся как данность)")
# что даёт сумма двух Qwen-сигналов
comb = (nll - nll.mean()) / nll.std() + (ent - ent.mean()) / ent.std()
auc = roc_auc_score(m.numpy(), comb.numpy())
print(f"  комбинация nll+entropy: AUC = {auc:.4f}")
