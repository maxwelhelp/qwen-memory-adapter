# Шаг 2: обучение МЕХАНИКИ памяти (дизайн: memory_design.md).
# Память: delta-rule, по головам. W_h (DH,DH) — СОДЕРЖИМОЕ (сессионное, пересоздаётся
# на каждом эпизоде записью). Обучаемые параметры МЕХАНИЗМА: η (скорость записи),
# o_out (линейная проекция выхода). Цель: чтение M(q_last) ≈ y_attn (то, что выдало
# бы полное внимание) — дистилляция.

import torch, math
from torch import nn

torch.manual_seed(0)
DEV = "cuda"
H, DH, D = 14, 64, 896
N_PROJ = 2          # проходов записи по контексту на эпизод
ETA = 0.05          # init скорости записи
LR = 1e-3
EPOCHS = 60         # по эпохе на каждую запись датасета (60 записей)

class DeltaMemory(nn.Module):
    """Линейная fast-weight память: W_h = Σ η·(v−Wk)⊗k; чтение y = o_out(W q)."""

    def __init__(self, H, DH, D):
        super().__init__()
        self.H, self.DH, self.D = H, DH, D
        # МЕХАНИЗМ (обучаемое): скорость записи + выходная проекция
        self.log_eta = nn.Parameter(torch.tensor(math.log(ETA)))
        self.o_out = nn.Linear(H * DH, D, bias=False)
        nn.init.zeros_(self.o_out.weight)   # старт: y=0 (память выключена)

    def write(self, k, v, steps=N_PROJ):
        """Блочная запись контекста в содержимое W (batch-delta, векторизовано по T).
        k,v: (1,H,T,DH). Возвращает W (H,DH,DH)."""
        k, v = k.float(), v.float()
        B, H, T, DH = k.shape
        W = torch.zeros(H, DH, DH, device=k.device)   # содержимое: старт с нуля (θ₀=0)
        eta = self.log_eta.exp()
        kk, vv = k[0], v[0]                                      # (H,T,DH)
        for _ in range(steps):
            pred = torch.einsum('htd,hde->hte', kk, W)          # Wk для всех токенов
            err = vv - pred                                     # (H,T,DH)
            grad = torch.einsum('hte,htd->hde', err, kk)        # Σ err⊗k (H,DH,DH)
            W = W + eta * grad / T                              # batch delta-rule
        return W

    def read(self, W, q):
        """Чтение по запросу q (1,H,1,DH) -> (1,1,D)."""
        qq = q.float()[:, :, 0, :]                              # (1,H,DH)
        yh = torch.einsum('hde,bhd->bhe', W, qq)                # (1,H,DH)
        return self.o_out(yh.reshape(1, -1))

    def forward(self, rec):
        W = self.write(rec["k"], rec["v"])
        return self.read(W, rec["q"])

def episode_loss(model, rec):
    y = model(rec)
    return (y - rec["y_attn"]).pow(2).mean(), y

ds = torch.load("dataset_attn.pt")
model = DeltaMemory(H, DH, D).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

print(f"память: delta-rule, H={H}, DH={DH}, η0={ETA}, проходов записи={N_PROJ}")
print(f"обучаемая механика: η, o_out ({sum(p.numel() for p in model.parameters())/1e6:.2f}M пар.)")
print(f"датасет: {len(ds)} записей, T до {max(r['T'] for r in ds)}")

for ep in range(1, EPOCHS + 1):
    losses, cors = [], []
    for rec in ds:
        rec_gpu = {kk: vv.to(DEV) for kk, vv in rec.items() if kk not in ("prompt", "T")}
        opt.zero_grad()
        loss, y = episode_loss(model, rec_gpu)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        yt = rec_gpu["y_attn"]
        cor = torch.cosine_similarity(y[0, 0], yt[0, 0], dim=0).item()
        cors.append(cor)
    sched.step()
    if ep % 10 == 0 or ep == 1:
        print(f"эпоха {ep:>3}: loss = {sum(losses)/len(losses):.4f}  "
              f"cosine(y, y_attn) = {sum(cors)/len(cors):.4f}  η = {model.log_eta.exp().item():.4f}")

torch.save({"model": model.state_dict()}, "memory_mech.pt")
print("сохранено memory_mech.pt")

# контроль: чтение на валидационных промптах
with torch.no_grad():
    print("\nконтроль на датасете (те же записи, другой порядок):")
    for i in [3, 17, 42]:
        rec = {kk: vv.to(DEV) for kk, vv in ds[i].items() if kk not in ("prompt", "T")}
        y = model(rec)
        cor = torch.cosine_similarity(y[0, 0], rec["y_attn"][0, 0], dim=0).item()
        print(f"  запись {i}: cosine = {cor:.4f}")
