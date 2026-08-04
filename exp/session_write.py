# Сессия 1 (реально отдельный процесс ОС): читает контекст, пишет в память,
# сохраняет ТОЛЬКО файл на диск. После завершения процесса в питоне не остаётся
# вообще ничего — это самый строгий тест "забыла всё, кроме файла".
#
# python session_write.py --example_idx 3 --out /tmp/memory_state.pt
#
# Пример использует готовый dataset_yattn.pt для ctx_hidden/entropy — это позиция,
# куда в реальном применении нужно подставить прогон СВОЕГО текста через тот же
# кусок пайплайна, что использует gen_yattn.py (получить ctx_hidden + entropy
# для нового, не синтетического текста).

import argparse, torch
from exp5_ctx_dynamic_gate import LinSlotsAssocCtxGate, _detach_work
from train_memory_distill import DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp5_ctx_gate.pt")
    ap.add_argument("--dataset", default="dataset_yattn.pt")
    ap.add_argument("--example_idx", type=int, required=True)
    ap.add_argument("--out", default="memory_state.pt")
    args = ap.parse_args()

    model = LinSlotsAssocCtxGate().to(DEV)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    model.eval()

    data = torch.load(args.dataset)
    d = data[args.example_idx]
    ctx = d["ctx_hidden"].to(DEV)
    work = model.write_work(ctx, d["entropy"].to(DEV))
    torch.save(_detach_work(work), args.out)

    print(f"Записано в память: example #{args.example_idx} "
          f"(type={d['type']}, secret={d['secret']}) -> {args.out}")
    print("Процесс завершится сейчас — контекст нигде больше не хранится.")


if __name__ == "__main__":
    main()
