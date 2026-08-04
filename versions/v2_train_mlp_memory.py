# Шаг 2b: единый MemoryMLP (Titans-стиль, dim=896) + дистилляция y_attn.
# Запись: градиентные шаги по ||M(k)−v||², ОДИН autograd.grad на эпизод (батч по токенам),
#         create_graph=True — граф записи дифференцируем (градиенты текут в механику θ0, η).
# Чтение: M_θ(q) -> y. Цель: y ≈ y_attn (то, что выдало бы полное внимание).

import torch, math
from torch import nn

torch.manual_seed(0)
DEV = "cuda"
D = 896
EXP = 4
N_PROJ = 2
ETA = 0.05
LR = 1e-3
EPOCHS = 60

class MemoryMLP(nn.Module):
    def __init__(self, d=D, exp=EXP):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d, d * exp) / (d ** 0.5))
        self.b1 = nn.Parameter(torch.zeros(d * exp))
        self.w2 = nn.Parameter(torch.randn(d * exp, d) / ((d * exp) ** 0.5))
        self.b2 = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return torch.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2


def mlp_forward(th, x):
    return torch.tanh(x @ th["w1"] + th["b1"]) @ th["w2"] + th["b2"]


def main():
    ds = torch.load("dataset_attn.pt")
    model = MLPMemory(D).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"MemoryMLP: {D}->{D*EXP}->{D}, механика {n_par/1e6:.2f}M пар. (θ0 + η)")
    print(f"датасет: {len(ds)} записей, T max {max(r['T'] for r in ds)}")

    for ep in range(1, EPOCHS + 1):
        losses, cors = [], []
        for rec in ds:
            r = {kk: vv.to(DEV) for kk, vv in rec.items() if kk not in ("prompt", "T")}
            opt.zero_grad()
            y = model(r)
            loss = (y - r["y_attn"]).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            cors.append(torch.cosine_similarity(y[0, 0], r["y_attn"][0, 0], dim=0).item())
        sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f"эпоха {ep:>3}: loss = {sum(losses)/len(losses):.4f}  "
                  f"cosine = {sum(cors)/len(cors):.4f}  η = {model.log_eta.exp().item():.4f}")

    torch.save({"model": model.state_dict()}, "mlp_memory_mech.pt")
    print("сохранено mlp_memory_mech.pt")

    # контроль (вне no_grad: write использует autograd.grad)
    print("\nконтроль:")
    for i in [3, 17, 42]:
        r = {kk: vv.to(DEV) for kk, vv in ds[i].items() if kk not in ("prompt", "T")}
        y = model(r)
        cor = torch.cosine_similarity(y[0, 0], r["y_attn"][0, 0], dim=0).item()
        print(f"  запись {i}: cosine = {cor:.4f}")


class MLPMemory(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D = D
        self.mlp = MemoryMLP(D)                    # θ0 — часть МЕХАНИЗМА (обучается)
        self.log_eta = nn.Parameter(torch.tensor(math.log(ETA)))

    @staticmethod
    def rmsnorm(x, eps=1e-6):
        """Нормализация входа (как pre-RMSNorm в Titans): нормы hidden ~900, без
        нормы MLP насыщается (tanh), градиенты умирают."""
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def write(self, k, v, steps=N_PROJ):
        """Градиентная запись контекста. k,v: (1,H,T,DH) -> (1,T,D). Возвращает θ."""
        k = self.rmsnorm(k.float().reshape(1, -1, self.D))
        v = self.rmsnorm(v.float().reshape(1, -1, self.D))
        eta = self.log_eta.exp()
        th = {n: p.detach().requires_grad_(True) for n, p in self.mlp.named_parameters()}
        for _ in range(steps):
            pred = mlp_forward(th, k[0])                       # (T,D) батч
            loss = (pred - v[0]).pow(2).mean()
            grads = torch.autograd.grad(loss, tuple(th.values()), create_graph=True)
            th = {n: (th[n] - eta * g) for n, g in zip(th.keys(), grads)}
        return th

    def read(self, th, q):
        """Чтение: q (1,H,1,DH) -> (1,D) -> M_θ(q)."""
        qq = self.rmsnorm(q.float().reshape(1, self.D))
        return mlp_forward(th, qq).reshape(1, 1, self.D)

    def forward(self, rec):
        th = self.write(rec["k"], rec["v"])
        return self.read(th, rec["q"])


if __name__ == "__main__":
    main()
