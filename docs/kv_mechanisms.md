# Механизмы KV-кэша: удаление, добавление, удивление, забывание
# Цель: референс для нашего MLP-фильтра (учится удалять/добавлять в KV-кэш).
# Источники кода: repos/KVCache-Factory, repos/kvpress (NVIDIA), repos/titans-pytorch.

---

## 1. УДАЛЕНИЕ из KV-кэша (внимание НЕ меняется — меняется набор KV)

### 1.1 По накопленной attention-массе (H2O, 2306.14048)
Идея: токен важен, если на него постоянно падает большое внимание.
```
score_t = Σ_s A[s, t]        # накопленная масса внимания на токен t
evict:  оставить top-k по score + последнее окно
```
Реализация (KVCache-Factory): `heavy_hitter`-индексы, top-k по кумулятивной массе,
выброс навсегда. Просто, но: «спящие токены» (ключи/числа с нулевым вниманием) теряются
(Transactional Attention, 2604.11288: все baseline-методы дают 0% на их retrieval).

### 1.2 По «голосованию» окна (SnapKV, 2404.14469)
Внимание последних N=16 токенов (observation window) «голосует» за важность
позиций префикса → top-k + окно. Работает в prefill.

### 1.3 По обученному скорингу + порог (KVzap/DMS, NVIDIA)
**Наш главный референс удаления.** MLP-предсказатель важности поверх hidden states;
пороговое удаление (Dynamic Memory Sparsification). Реальный код (kvpress/dms_press.py):

```python
# DMS: порог вместо фиксированного ratio — адаптивное сжатие
scores = press.score(module, hidden_states, keys, values, None, kwargs)  # MLP-скоринг
if scores_buffer.shape[-1] > sliding_window_size:           # 128 последних защищены
    n_to_evict = scores_buffer.shape[-1] - sliding_window_size
    scores_to_evict = scores_buffer[..., :n_to_evict]
    scores_buffer = scores_buffer[..., n_to_evict:]
    # ВЫБРОС: все токены со скором ниже порога
    new_masked = list(torch.where(scores_to_evict < threshold))
    # ... индексы мержатся в module.masked_key_indices → attention их не читает
```
MLP-скоринг (kvzap_press.py) — по голове на слой:
```python
self.layers = nn.ModuleList(                      # per-layer модули
    nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))
    for _ in range(n_modules))
def forward(self, x):                              # x: hidden states (T, d_in)
    return torch.stack([m(x[:, i, :]) for i, m in enumerate(self.layers)], dim=1)
```
Обучение KVzap (kvzap/train.py): target = log-importance из KVzip (дорогого точного
скоринга) → дистилляция в маленький MLP.

### 1.4 Не читать, а не удалять (Quest / TSA / DELTA)
Кэш остаётся, но внимание получает подмножество: TSA — per-head выбор → сжатие →
стандартное внимание → разжатие (совместимо с FlashAttention); DELTA — слои выбирают
токены, остальные слои читают только их. «Обратимость»: выброшенное можно вернуть.

### 1.5 Write-gate: решать ДО записи (WG-KV, 2512.17452)
MLP предсказывает полезность токена до записи в кэш → неважное вообще не пишется:
```
utility_t = WriteGate(h_t);  write if utility_t > θ_wg
```
Преимущество: кэш никогда не раздувается; совместимо с FlashAttention/PagedKV.

---

## 2. ДОБАВЛЕНИЕ в KV-кэш / повышение скора

### 2.1 Супер-токены-выжимка (CacheNotes 2510.10129, Cartridges 2606.04557)
Корпус офлайн **сжимается в компактный KV** (context distillation: обучается так, чтобы
attention-ответ с выжимкой ≈ ответ с полным текстом):
```
L = E[ KL(attn(q, cache_full) || attn(q, cache_full ∪ выжимка)) ]  # или MSE на выходах
выжимка (K векторов) кладётся в кэш как супер-токены
```
Cartridges: картридж на документ (3–4× меньше токенов, чем текст), ретрив картриджей
dense-retrieval'ом, загрузка в кэш. Наша MLP-память = «глубокий картридж» (веса MLP
вместо обучаемых токенов).

### 2.2 Persistent memory (Titans, eq.19)
Learnable токены в начале последовательности — всегда читаются вниманием:
```
x_new = [p_1, ..., p_{N_p}] || x        # p_i — обучаемые векторы (N_p ≈ 128)
```
Технически: входные-независимые ключи/значения = «умолчания», которые внимание всегда
учитывает (снимает перекос внимания на первые токены).

### 2.3 Link-токены (KVLINK)
Обучаемые токены между независимо закодированными сегментами — восстанавливают
кросс-сегментное внимание: `cache = [seg1, link, seg2, link, ...]`.

### 2.4 Восстановление выброшенного (ADORE, 2024.findings-acl.837)
GRU-контроллер предсказывает, какие ВЫБРОШЕННЫЕ токены станут нужны → их KV
пересобираются из запомненного представления и возвращаются в кэш.

### 2.5 Повышение скора соседям (GraphKV, EMNLP 2025)
Важность «растекается»: выбранные важные узлы передают decay-сигнал похожим соседям:
```
importance_j += decay · importance_i · sim(k_i, k_j)   для похожих (i,j)
```
→ разнообразие удержанных токенов (top-k по сходству вырождается в кучу одинаковых).

---

## 3. УДИВЛЕНИЕ и ЗАБЫВАНИЕ (Titans — код titans-pytorch/neural_memory.py)

### 3.1 Surprise (что записывать сильно)
Удивление = градиент ассоциативной потери по весам памяти:
```
S_t = ∇_θ ℓ(M_θ(k_t), v_t),  ℓ = ||M(k) − v||²        # per-sample градиент
```
Код: `per_sample_grad_fn = vmap(grad(forward_and_loss), in_dims=(0,0,0,0))` — градиент
для каждого токена сразу. Surprise = насколько запись неожиданна → приоритет записи.

### 3.2 Momentum (удержание во времени, eq.10 Titans)
```
S̄_t = η_t · S_t + θ_t · S̄_{t−1}      # η — data-dependent decay, θ — вес momentary
```
Код: `assoc_scan(adaptive_momentum, surprise, prev=last_momentum)` — параллельный скан.

### 3.3 Forgetting (забывание, эквивалент weight decay)
```
θ ← θ · (1 − γ_t) − S̄_t             # γ_t = σ(W_γ x_t) — обучаемое забывание
```
Код: `to_decay_factor = Linear(dim, heads) → sigmoid; update = assoc_scan(1−decay, update)`.

### 3.4 Adaptive learning rate (как сильно писать)
```
η_t = σ(W_η x_t) · η_max
```
Код: `to_adaptive_step → sigmoid → * η_max`.

### 3.5 Приоритеты записи через маску (готовый механизм!)
```python
# store_memories(mask=...): если mask=False → скорость записи = 0
adaptive_lr = torch.where(mask, adaptive_lr, 0.)   # ТОЧНО так boost/приоритеты
```
→ НАША критика-буст = маска/множитель η для важных пар: `η_eff = η · (1 + boost·crit)`.

### 3.6 EpiKV-удивление БЕЗ обучения (2606.26472)
```
g_l(t)  = ||h_l(t) − h_l(t−1)||₂          # L2-изменение hidden между decode-шагами
z_l(t)  = (g_l(t) − μ_l(t)) / σ_l(t)      # z-score по скользящему окну 64
score(t)= z₁₀(t) − z₂₁(t)                # слои 7–13 коррелируют «+», 18–25 «−»
```
72% MATH-500 при кэше 4096 (vs H2O 67%), до 2.8× быстрее, без обучения.

---

## 4. СИНТЕЗ: наш MLP-фильтр (удаляет + добавляет + удивление + забывание)

### 4.1 Архитектура (всё поверх НЕизменного attention)

```
hidden h_t ──► [фичи] ──► MLP-скоринг ──► score_t ──► решения:
                    │
                    ├── EpiKV z-score (удивление, бесплатно) ──► +α·z_t
                    ├── attention-масса (H2O-сигнал)        ──► +β·a_t
                    └── критика-буст (наши приоритеты)      ──► +γ·boost_t

решения:
  УДАЛИТЬ:      score_t < threshold  (DMS-механика, sliding window 128)
  НЕ ПИСАТЬ:    score_t < θ_wg       (write-gate, до записи)
  ДОБАВИТЬ:     выжимка-супер-токены (CacheNotes/Cartridges-стиль)
  ПОВЫСИТЬ:     score_j += decay·score_i·sim(k_i,k_j)  (GraphKV-стиль)
  ЗАБЫТЬ:       decay по времени (Titans γ_t) или выброс по score
```

### 4.2 Формулы итогового скора
```
score_t = MLP(h_t)  +  α·z_t(удивление)  +  β·A_t(масса)  +  γ·crit_t(критика)
evict   = (score_t < τ) & (t < T − window)
write   = score_t ≥ θ_wg
выжимка: L = ||attn(q, cache) − attn(q, cache ∪ sup)||²   (обучение добавления)
```

### 4.3 Что брать из готового кода
| Механизм | Код-база | Что берём |
|---|---|---|
| Пороговое удаление | kvpress/dms_press.py | `torch.where(scores < threshold)` + sliding window |
| MLP-скоринг | kvpress/kvzap_press.py | per-layer MLP, вход = hidden states |
| Surprise-градиент | titans-pytorch/neural_memory.py | `per_sample_grad_fn` (vmap grad) |
| Momentum/decay | titans-pytorch (assoc_scan) | `assoc_scan(1−γ, update)` |
| Приоритеты записи | titans-pytorch (store_mask) | `adaptive_lr = where(mask, lr, 0)` |
| Эвристика удивления | EpiKV (формула §3.6) | z-score L2-разницы hidden, 15 строк |
| Выжимка-добавление | CacheNotes/Cartridges (дист.) | context distillation loss |

### 4.4 Порядок внедрения (минимальный путь)
1. **Без обучения**: EpiKV-скоринг (удивление) + DMS-порог + sliding window → сравнить с H2O
   на dormant-задачах (где H2O даёт 0%) — цель: не падать в 0. (1–2 дня, kvpress-база)
2. **Обученный скоринг**: KVzap-стиль MLP (hidden → score), target = контрафактика/важность
   → сравнить с KVzap. (2–3 дня)
3. **Добавление**: выжимка-супер-токены (дистилляция §4.2) → сравнить с CacheNotes.
4. **Критика-буст**: наши приоритеты из критики (η-маска Titans) поверх скоринга.
