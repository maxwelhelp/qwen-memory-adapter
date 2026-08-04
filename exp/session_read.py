# Сессия 2 (реально отдельный процесс ОС, запускается ПОСЛЕ session_write.py):
# НЕ имеет доступа ни к какому объекту из сессии 1 — только к файлу памяти на
# диске и к вопросу. Контекст (пароль/секрет) нигде в этом процессе не передаётся.
#
# python session_read.py --example_idx 3 --mem /tmp/memory_state.pt

import argparse, torch
from exp5_ctx_dynamic_gate import (
    LinSlotsAssocCtxGate, _load_work, qwen_lmhead_logits,
)
from train_memory_distill import DEV, last_layer_forward, cand_logits, SEC_TOK_VALUES, SECRET_TO_IDX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp5_ctx_gate.pt")
    ap.add_argument("--dataset", default="dataset_yattn.pt")
    ap.add_argument("--example_idx", type=int, required=True,
                     help="индекс только для того, чтобы взять готовый q_hidden/h_inA_all "
                          "из синтетического датасета для проверки; в реальном применении "
                          "сюда подставляется прогон РЕАЛЬНОГО вопроса через Qwen (без контекста)")
    ap.add_argument("--mem", default="memory_state.pt")
    args = ap.parse_args()

    model = LinSlotsAssocCtxGate().to(DEV)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    model.eval()
    work = _load_work(torch.load(args.mem, map_location="cpu"))

    data = torch.load(args.dataset)
    d = data[args.example_idx]        # контекст (d["ctx_hidden"]) НЕ используется ниже
    q_h = d["q_hidden"].to(DEV)
    h_all = d["h_inA_all"].to(DEV)

    with torch.no_grad():
        base_cand = cand_logits(last_layer_forward(h_all))
        mv = model.mix_vec(q_h, work, base_cand)
        h_inj = h_all.clone()
        h_inj[-1] = h_inj[-1] + mv
        h_out_p = last_layer_forward(h_inj)
        lg = qwen_lmhead_logits(h_out_p)
        top_id = lg.argmax(-1).item()

    correct_id = SEC_TOK_VALUES[SECRET_TO_IDX[d["secret"]]]
    print(f"Ответ памяти (top-1 token id): {top_id}")
    print(f"[только для проверки в тесте] правильный id: {correct_id} | "
          f"match={top_id == correct_id}")


if __name__ == "__main__":
    main()
