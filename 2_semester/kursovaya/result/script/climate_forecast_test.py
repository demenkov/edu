#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест воспроизводимой методики прогноза климатического временного ряда.

Что проверяется:
  • разделение ряда СТРОГО по времени (train / val / test, без перемешивания);
  • нормализация ТОЛЬКО по обучающей выборке;
  • формирование временных окон заданной длины;
  • обучение LSTM и GRU (Adam, MSE, Dropout, ранняя остановка);
  • ОБЯЗАТЕЛЬНОЕ сравнение с базовыми методами:
        - наивный прогноз ("как в прошлом месяце"),
        - линейная регрессия,
        - ARIMA;
  • метрики MAE, RMSE, R² считаются ТОЛЬКО на тестовой выборке.

Данные:
  По умолчанию генерируется синтетический ряд (тренд + сезонность + шум),
  чтобы скрипт запускался без интернета. Для реальных данных:
      python climate_forecast_test.py --csv путь.csv --col value
  Для NASA GISTEMP (годы×месяцы): --csv GLB.Ts+dSST.csv --gistemp
      файл: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv

Зависимости: numpy, pandas, scikit-learn, torch, (опционально) statsmodels.

Как запустить:

pip install numpy pandas scikit-learn torch statsmodels
python climate_forecast_test.py                 # синтетика, без интернета
python climate_forecast_test.py --csv data.csv --col value   # свой CSV
"""

import argparse
import numpy as np
import pandas as pd
 
# --- опциональные зависимости: скрипт работает и без них ------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
 
try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
 
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
 
 
# ==========================================================================
# 1. ДАННЫЕ
# ==========================================================================
GISTEMP_URL = ("https://data.giss.nasa.gov/gistemp/tabledata_v4/"
               "GLB.Ts+dSST.csv")
 
 
def download_gistemp(dest="GLB.Ts+dSST.csv"):
    """Скачать месячные аномалии температуры NASA GISTEMP. Возвращает путь или None."""
    import os
    import urllib.request
    if os.path.exists(dest):
        print(f"[данные] файл уже скачан: {dest}")
        return dest
    try:
        print(f"[данные] скачиваю NASA GISTEMP -> {dest}")
        req = urllib.request.Request(GISTEMP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest
    except Exception as e:
        print(f"[!] скачать не удалось ({e}).")
        return None
 
 
def parse_gistemp(path):
    """Разобрать таблицу GISTEMP (годы×месяцы) в один месячный ряд."""
    df = pd.read_csv(path, skiprows=1)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = (df[months]
            .replace("***", np.nan)
            .apply(pd.to_numeric, errors="coerce")
            .values.flatten())
    s = pd.Series(vals).dropna().astype(np.float32).values
    print(f"[данные] GISTEMP: {len(s)} месячных значений.")
    return s
 
 
def make_synthetic(n_months=600, seed=42):
    """Синтетический месячный ряд: линейный тренд + сезон + шум."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    trend = 0.0015 * t                      # медленное потепление
    season = 0.5 * np.sin(2 * np.pi * t / 12)   # годовая сезонность
    noise = rng.normal(0, 0.15, n_months)
    return (trend + season + noise).astype(np.float32)
 
 
def load_series(args):
    """Загрузить ряд: автоскачивание GISTEMP, локальный CSV или синтетика."""
    if args.download:
        path = download_gistemp()
        if path:
            return parse_gistemp(path)
        print("[данные] откат на синтетический ряд.")
        return make_synthetic(args.n_months)
 
    if not args.csv:
        print("[данные] CSV не задан — используется синтетический ряд.")
        return make_synthetic(args.n_months)
 
    if args.gistemp:
        return parse_gistemp(args.csv)
 
    df = pd.read_csv(args.csv)
    s = pd.to_numeric(df[args.col], errors="coerce").dropna().astype(np.float32).values
    print(f"[данные] {args.csv}: {len(s)} значений из столбца '{args.col}'.")
    return s
 
 
# ==========================================================================
# 2. РАЗДЕЛЕНИЕ, НОРМАЛИЗАЦИЯ, ОКНА
# ==========================================================================
def split_indices(n, train=0.70, val=0.15):
    """Границы train / val / test строго по времени."""
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return i_tr, i_va
 
 
def make_windows(feat_scaled, tgt_scaled, raw, lo, hi, W, H, floor=0):
    """
    Окна для прогноза на H шагов вперёд. Цель попадает в [lo, hi).
      feat_scaled — то, что видит модель на входе (уровни или 1-шаговые приросты);
      tgt_scaled  — то, что модель предсказывает (уровень или H-шаговый прирост);
      raw         — исходный ряд уровней (для y_raw и anchor);
      H           — горизонт прогноза (на сколько шагов вперёд);
      floor       — минимальный индекс входа (в режиме --diff = 1).
    Для цели j: последний наблюдаемый индекс = j-H, окно входа — W точек до него.
    anchor = raw[j-H] — последнее известное значение (для наивного и для --diff).
    """
    X, y, y_raw, anchor = [], [], [], []
    start = max(lo, H + W - 1 + floor)
    for j in range(start, hi):
        obs_end = j - H                      # последний наблюдаемый индекс
        X.append(feat_scaled[obs_end - W + 1:obs_end + 1])
        y.append(tgt_scaled[j])
        y_raw.append(raw[j])
        anchor.append(raw[obs_end])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(y_raw, dtype=np.float32),
            np.array(anchor, dtype=np.float32))
 
 
# ==========================================================================
# 3. МЕТРИКИ
# ==========================================================================
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2
 
 
# ==========================================================================
# 4. НЕЙРОСЕТЬ (LSTM / GRU)
# ==========================================================================
if HAS_TORCH:
    class RNNForecaster(nn.Module):
        def __init__(self, kind="lstm", hidden=64, dropout=0.2):
            super().__init__()
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn(input_size=1, hidden_size=hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden, hidden // 2)
            self.act = nn.ReLU()
            self.out = nn.Linear(hidden // 2, 1)
 
        def forward(self, x):                 # x: (B, W) -> (B, W, 1)
            x = x.unsqueeze(-1)
            o, _ = self.rnn(x)
            h = o[:, -1, :]                   # последнее состояние
            h = self.drop(h)
            h = self.act(self.fc1(h))
            return self.out(h).squeeze(-1)
 
 
def train_rnn(kind, Xtr, ytr, Xva, yva, hidden, dropout,
              epochs, batch, lr, patience, seed=0):
    torch.manual_seed(seed)
    model = RNNForecaster(kind, hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
 
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
 
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva_t), yva_t).item()
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:               # ранняя остановка
                break
    if best_state:
        model.load_state_dict(best_state)
    return model
 
 
def predict_rnn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy()
 
 
# ==========================================================================
# 5. ARIMA (walk-forward, прогноз на H шагов)
# ==========================================================================
def run_arima(raw, i_va, H, order=(5, 1, 0)):
    """Для каждой цели j из [i_va, n) прогноз на H шагов от момента j-H."""
    n = len(raw)
    origin0 = i_va - H                       # первый момент наблюдения
    res = ARIMA(list(raw[:origin0 + 1]), order=order).fit()
    preds = []
    for k, j in enumerate(range(i_va, n)):
        preds.append(float(res.forecast(H)[-1]))   # берём H-й шаг вперёд
        nxt = origin0 + k + 1                       # добавляем следующий факт
        if nxt < n:
            res = res.append([raw[nxt]], refit=False)
    return np.array(preds, dtype=np.float32)
 
 
# ==========================================================================
# MAIN
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Тест прогноза климатического ряда")
    ap.add_argument("--csv", default=None, help="путь к CSV (если нет — синтетика)")
    ap.add_argument("--col", default="value", help="столбец со значениями")
    ap.add_argument("--gistemp", action="store_true", help="формат NASA GISTEMP")
    ap.add_argument("--download", action="store_true",
                    help="скачать данные NASA GISTEMP автоматически")
    ap.add_argument("--n-months", type=int, default=600, help="длина синтетики")
    ap.add_argument("--window", type=int, default=24, help="длина окна (мес.)")
    ap.add_argument("--diff", action="store_true",
                    help="режим дифференцирования: предсказывать прирост, а не уровень")
    ap.add_argument("--horizon", type=int, default=1,
                    help="горизонт прогноза в шагах (месяцах) вперёд")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()
 
    # --- данные ---
    series = load_series(args)
    n = len(series)
    W = args.window
    if n < W + args.horizon + 30:
        raise SystemExit("Ряд слишком короткий для такого окна и горизонта.")
 
    # --- разделение строго по времени ---
    i_tr, i_va = split_indices(n)
    print(f"[split] train=[0:{i_tr}]  val=[{i_tr}:{i_va}]  test=[{i_va}:{n}]")
 
    # --- режим: уровни или приросты (--diff) ---
    H = args.horizon
    if args.diff:
        # вход: 1-шаговые приросты; цель: H-шаговый прирост y[j] - y[j-H]
        onestep = np.zeros_like(series)
        onestep[1:] = series[1:] - series[:-1]
        hstep = np.zeros_like(series)
        hstep[H:] = series[H:] - series[:-H]
        mu_f, sd_f = float(onestep[1:i_tr].mean()), float(onestep[1:i_tr].std())
        mu_t, sd_t = float(hstep[H:i_tr].mean()), float(hstep[H:i_tr].std())
        feat_scaled = (onestep - mu_f) / sd_f
        tgt_scaled = (hstep - mu_t) / sd_t
        floor = 1
        print(f"[mode] дифференцирование: цель = прирост за {H} шаг(ов)")
    else:
        mu_t, sd_t = float(series[:i_tr].mean()), float(series[:i_tr].std())
        feat_scaled = (series - mu_t) / sd_t
        tgt_scaled = feat_scaled
        floor = 0
        print("[mode] обычный режим: цель = уровень")
 
    print(f"[horizon] прогноз на {H} шаг(ов) вперёд")
    print(f"[norm] target mean={mu_t:.4f}, std={sd_t:.4f} (рассчитаны на train)")
 
    # --- окна по сегментам ---
    Xtr, ytr, _, _ = make_windows(feat_scaled, tgt_scaled, series, 0, i_tr, W, H, floor)
    Xva, yva, _, _ = make_windows(feat_scaled, tgt_scaled, series, i_tr, i_va, W, H, floor)
    Xte, yte, yte_raw, anchor_te = make_windows(feat_scaled, tgt_scaled, series, i_va, n, W, H, floor)
    print(f"[windows] train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
 
    def denorm(z):
        return z * sd_t + mu_t
 
    def to_level(model_out, anchor):
        """Перевести выход модели в прогноз уровня."""
        base = denorm(model_out)
        return anchor + base if args.diff else base   # ŷ = y_{j-H} + Δ̂
 
    results = {}
 
    # --- базовый 1: наивный прогноз (значение H шагов назад) ---
    results["Наивный"] = metrics(yte_raw, anchor_te)
 
    # --- базовый 2: линейная регрессия по окну ---
    if HAS_SKLEARN:
        lr = LinearRegression().fit(Xtr, ytr)
        pred = to_level(lr.predict(Xte), anchor_te)
        results["Лин. регрессия"] = metrics(yte_raw, pred)
    else:
        print("[!] sklearn не установлен — линейная регрессия пропущена.")
 
    # --- базовый 3: ARIMA ---
    if HAS_STATSMODELS:
        try:
            arima_pred_full = run_arima(series, i_va, H)
            arima_pred = arima_pred_full[len(arima_pred_full) - len(yte_raw):]
            results["ARIMA(5,1,0)"] = metrics(yte_raw, arima_pred)
        except Exception as e:
            print(f"[!] ARIMA не удалась: {e}")
    else:
        print("[!] statsmodels не установлен — ARIMA пропущена.")
 
    # --- нейросети LSTM и GRU ---
    if HAS_TORCH:
        for kind in ["lstm", "gru"]:
            model = train_rnn(kind, Xtr, ytr, Xva, yva,
                              args.hidden, args.dropout, args.epochs,
                              args.batch, args.lr, args.patience)
            pred = to_level(predict_rnn(model, Xte), anchor_te)
            results[kind.upper()] = metrics(yte_raw, pred)
    else:
        print("[!] torch не установлен — LSTM/GRU пропущены.")
 
    # --- таблица результатов (только на тесте) ---
    print("\n" + "=" * 56)
    print(f"РЕЗУЛЬТАТЫ НА ТЕСТЕ — горизонт {H} шаг(ов) (исходные единицы)")
    print("=" * 56)
    print(f"{'Метод':<18}{'MAE':>10}{'RMSE':>12}{'R²':>12}")
    print("-" * 56)
    for name, (mae, rmse, r2) in results.items():
        print(f"{name:<18}{mae:>10.4f}{rmse:>12.4f}{r2:>12.4f}")
    print("=" * 56)
    print("Чем меньше MAE и RMSE — тем лучше. R² ближе к 1 — тем лучше.")
    print("Польза нейросети есть, только если она ОБГОНЯЕТ базовые методы.")
 
 
if __name__ == "__main__":
    main()