import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import sklearn.metrics

# ── NOTE ──────────────────────────────────────────────────────────────────────
# This script re-runs the full pipeline to ensure the scalers are identical
# to what the model was trained on, then forecasts N_DAYS ahead recursively.
# ─────────────────────────────────────────────────────────────────────────────

tf.random.set_seed(99)
np.random.seed(99)

N_DAYS = 5   # <── change this to forecast further (accuracy degrades past ~5)

# ──────────────────────────────────────────────
# 1. REBUILD DATA + SCALERS (must match v4 exactly)
# ──────────────────────────────────────────────
url = 'https://raw.githubusercontent.com/SusmitSekharBhakta/Stock-market-price-prediction/main/final_data_adj.csv'
df_raw = pd.read_csv(url)
dates  = pd.to_datetime(df_raw['Date'])
df_raw.drop(columns=['Date'], inplace=True)

imputer = SimpleImputer(missing_values=np.nan)
df = pd.DataFrame(imputer.fit_transform(df_raw), columns=df_raw.columns).reset_index(drop=True)

raw_prices = df[['Open', 'Close']].copy()

for col in ['Open', 'Close']:
    df[col] = np.log(df[col] / df[col].shift(1))

df.dropna(inplace=True)
dates      = dates.iloc[1:].reset_index(drop=True)
raw_prices = raw_prices.iloc[1:].reset_index(drop=True)
df         = df.reset_index(drop=True)

WINDOW    = 40
SPLIT_IDX = int(0.85 * len(df))

train_df = df.iloc[:SPLIT_IDX]

feature_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_df.values)
target_scaler  = MinMaxScaler(feature_range=(-1, 1)).fit(train_df[['Open', 'Close']].values)

def scale_df(raw):
    s = pd.DataFrame(feature_scaler.transform(raw.values), columns=raw.columns)
    s[['Open', 'Close']] = target_scaler.transform(raw[['Open', 'Close']].values)
    return s.astype(float)

full_scaled = scale_df(df)

# ──────────────────────────────────────────────
# 2. LOAD / REBUILD TRAINED MODEL
#    If you saved the model with model.save('gru_v4.keras'),
#    replace the build+fit block with:
#      model = tf.keras.models.load_model('gru_v4.keras', ...)
#    Otherwise we retrain here so this file is self-contained.
# ──────────────────────────────────────────────
def huber_directional_loss(delta=0.05, direction_weight=0.25):
    def loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)
        huber = tf.where(abs_err <= delta,
                         0.5 * tf.square(err),
                         delta * (abs_err - 0.5 * delta))
        dir_wrong = tf.maximum(0.0, -tf.sign(y_true) * tf.sign(y_pred))
        return tf.reduce_mean(huber) + direction_weight * tf.reduce_mean(dir_wrong)
    loss.__name__ = 'huber_directional'
    return loss

def build_sequences(scaled, offset):
    vals    = scaled.values
    targets = scaled[['Open', 'Close']].values
    X, y, anchors = [], [], []
    for i in range(len(scaled) - WINDOW):
        X.append(vals[i : i + WINDOW])
        y.append(targets[i + WINDOW])
        anchors.append(raw_prices.iloc[offset + i + WINDOW - 1].values)
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(anchors, dtype=np.float64))

train_scaled = scale_df(train_df)
test_scaled  = scale_df(df.iloc[SPLIT_IDX:])

X_train, y_train, _ = build_sequences(train_scaled, 0)
X_test,  y_test,  test_anchors  = build_sequences(test_scaled,  offset=SPLIT_IDX) # Added to define X_test, y_test, test_anchors

def build_model(input_shape, hidden=64, l2=5e-5):
    reg = tf.keras.regularizers.l2(l2)
    inp = tf.keras.Input(shape=input_shape)
    x1  = tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=reg)(inp)
    x1  = tf.keras.layers.BatchNormalization()(x1)
    x1  = tf.keras.layers.Dropout(0.2)(x1)
    x1p = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden, use_bias=False))(x1)
    x2  = tf.keras.layers.GRU(hidden, return_sequences=True, kernel_regularizer=reg)(x1)
    x2  = tf.keras.layers.BatchNormalization()(x2)
    x2  = tf.keras.layers.Dropout(0.3)(x2)
    x2  = tf.keras.layers.Add()([x2, x1p])
    x3  = tf.keras.layers.GRU(hidden, return_sequences=True, kernel_regularizer=reg)(x2)
    x3  = tf.keras.layers.BatchNormalization()(x3)
    x3  = tf.keras.layers.Dropout(0.3)(x3)
    x3  = tf.keras.layers.Add()([x3, x2])
    x4  = tf.keras.layers.GRU(32, return_sequences=False, kernel_regularizer=reg)(x3)
    x4  = tf.keras.layers.Dropout(0.4)(x4)
    out = tf.keras.layers.Dense(2, activation='linear')(x4)
    return tf.keras.Model(inp, out)

model = build_model((X_train.shape[1], X_train.shape[2]))
EPOCHS, WARMUP, LR_MAX, LR_MIN = 60, 5, 3e-4, 1e-6

def cosine_warmup(epoch):
    if epoch < WARMUP:
        return LR_MIN + (LR_MAX - LR_MIN) * (epoch / WARMUP)
    p = (epoch - WARMUP) / max(EPOCHS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * p))

model.compile(loss=huber_directional_loss(), metrics=['MAE'],
              optimizer=tf.keras.optimizers.Adam(LR_MAX))

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, # 'history' is defined here
          validation_split=0.1, verbose=1, callbacks=[
              tf.keras.callbacks.LearningRateScheduler(cosine_warmup, verbose=0),
              tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                               restore_best_weights=True, verbose=1),
          ])

# ──────────────────────────────────────────────
# Prediction for performance metrics
# ──────────────────────────────────────────────
pred_scaled  = model.predict(X_test)
# Inverse-transform back to actual log-returns
pred_returns = target_scaler.inverse_transform(pred_scaled)
true_returns = target_scaler.inverse_transform(y_test)
# ONE-STEP price reconstruction per sample — no cumsum
pred_prices = test_anchors * np.exp(pred_returns)
true_prices = test_anchors * np.exp(true_returns)

# ──────────────────────────────────────────────
# 3. RECURSIVE FORECASTING
#
#    Starting from the LAST WINDOW of known data,
#    we predict one step ahead, then:
#      - reconstruct the scaled feature row for that step
#        (all non-target features held at their last known value;
#         only Open/Close returns are updated from the prediction)
#      - slide the window forward by one, drop the oldest row
#      - repeat N_DAYS times
#
#    This is the "recursive" or "iterated one-step" strategy.
#    Uncertainty grows each step because errors in step t
#    feed into the input for step t+1.
# ──────────────────────────────────────────────

# Seed window: last WINDOW rows of the full scaled dataset
seed_window = full_scaled.values[-WINDOW:].copy()   # shape (WINDOW, n_features)
last_known_price = raw_prices.iloc[-1].values        # anchor for first forecast step
last_known_date  = dates.iloc[-1]

open_col  = list(df.columns).index('Open')
close_col = list(df.columns).index('Close')

forecast_returns = []
forecast_prices  = []
forecast_dates   = []

current_window = seed_window.copy()
current_price  = last_known_price.copy()

for step in range(N_DAYS):
    x_input = current_window[np.newaxis, :, :].astype(np.float32)

    # Predict scaled log-return
    pred_scaled = model.predict(x_input, verbose=0)[0]          # shape (2,)
    pred_return = target_scaler.inverse_transform([pred_scaled])[0]  # actual log-return

    # Reconstruct price from the current anchor
    next_price  = current_price * np.exp(pred_return)

    forecast_returns.append(pred_return)
    forecast_prices.append(next_price)

    # Advance date (skip weekends naively)
    next_date = last_known_date + pd.Timedelta(days=step + 1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)
    forecast_dates.append(next_date)

    # Build the next scaled feature row:
    # - copy last row of window (carries forward volume, RSI, MACD etc.)
    # - overwrite Open/Close with the freshly predicted scaled return
    new_row = current_window[-1].copy()
    new_row[open_col]  = pred_scaled[0]
    new_row[close_col] = pred_scaled[1]

    # Slide window forward
    current_window = np.vstack([current_window[1:], new_row])
    current_price  = next_price

# ──────────────────────────────────────────────
# 4. PLOT
# ──────────────────────────────────────────────
forecast_prices  = np.array(forecast_prices)   # (N_DAYS, 2)
forecast_returns = np.array(forecast_returns)  # (N_DAYS, 2)

# Show last 60 days of history + forecast
history_n    = 60
history_dates  = dates.iloc[-history_n:].values
history_open   = raw_prices['Open'].iloc[-history_n:].values
history_close  = raw_prices['Close'].iloc[-history_n:].values

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

for ax, hist, fc, title in zip(
    axes,
    [history_open,  history_close],
    [forecast_prices[:, 0], forecast_prices[:, 1]],
    ['Open Price',  'Close Price'],
):
    ax.plot(history_dates, hist, color='steelblue', linewidth=1.5, label='Historical')

    # Connect history to forecast with a dotted bridge
    bridge_x = [history_dates[-1], forecast_dates[0]]
    bridge_y = [hist[-1],          fc[0]]
    ax.plot(bridge_x, bridge_y, color='orange', linewidth=1, linestyle=':')

    ax.plot(forecast_dates, fc, color='orange', linewidth=2,
            marker='o', markersize=5, label=f'{N_DAYS}-day forecast')

    # Naive uncertainty bands: ±1 std of historical daily returns, widening per step
    hist_returns = np.diff(np.log(hist))
    daily_std    = hist_returns.std()
    for i, (fd, fp) in enumerate(zip(forecast_dates, fc)):
        sigma = daily_std * np.sqrt(i + 1) * fp   # scales with price level
        ax.fill_between([fd], [fp - sigma], [fp + sigma],
                        color='orange', alpha=0.15)

    ax.set_title(f'{title} — last {history_n} days + {N_DAYS}-day forecast')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('forecast_v4.png', dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 5. FORECAST TABLE
# ──────────────────────────────────────────────
print(f"\n{'Date':<14} {'Open':>10} {'Close':>10} {'Open ret%':>10} {'Close ret%':>10}")
print("─" * 58)
for i, (d, p, r) in enumerate(zip(forecast_dates, forecast_prices, forecast_returns)):
    print(f"{str(d.date()):<14} {p[0]:>10.2f} {p[1]:>10.2f} "
          f"{r[0]*100:>+9.3f}% {r[1]*100:>+9.3f}%")

print(f"\nLast known prices  →  Open: {last_known_price[0]:.2f}  Close: {last_known_price[1]:.2f}")
print(f"Forecast end       →  Open: {forecast_prices[-1,0]:.2f}  Close: {forecast_prices[-1,1]:.2f}")
print(f"Total move         →  Open: {((forecast_prices[-1,0]/last_known_price[0])-1)*100:+.2f}%  "
      f"Close: {((forecast_prices[-1,1]/last_known_price[1])-1)*100:+.2f}%")

# ═══════════════════════════════════════════════════════════════
# PERFORMANCE METRICS + VISUALISATION
# Paste this at the end of stacked_GRU_v4.py in your Colab cell
# Requires: pred_returns, true_returns, pred_prices, true_prices
#           history (from model.fit), y_train, X_train
# ═══════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────
# A. COMPUTE ALL METRICS
# ──────────────────────────────────────────────

def compute_metrics(true, pred, label):
    r2       = sklearn.metrics.r2_score(true, pred)
    mse      = sklearn.metrics.mean_squared_error(true, pred)
    mae      = sklearn.metrics.mean_absolute_error(true, pred)
    rmse     = np.sqrt(mse)
    dir_acc  = np.mean(np.sign(true) == np.sign(pred))
    # Mean Absolute Percentage Error (on returns, avoid div-by-zero)
    nonzero  = true != 0
    mape     = np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100
    # Hit rate per quartile — is the model better on large moves?
    q75      = np.percentile(np.abs(true), 75)
    big_mask = np.abs(true) >= q75
    big_dir  = np.mean(np.sign(true[big_mask]) == np.sign(pred[big_mask])) if big_mask.any() else np.nan
    return {
        'label':    label,
        'R²':       r2,
        'MSE':      mse,
        'RMSE':     rmse,
        'MAE':      mae,
        'MAPE %':   mape,
        'Dir Acc':  dir_acc,
        'Dir Acc (large moves)': big_dir,
    }

m_open_ret   = compute_metrics(true_returns[:, 0], pred_returns[:, 0], 'Open returns')
m_close_ret  = compute_metrics(true_returns[:, 1], pred_returns[:, 1], 'Close returns')
m_open_px    = compute_metrics(true_prices[:, 0],  pred_prices[:, 0],  'Open price')
m_close_px   = compute_metrics(true_prices[:, 1],  pred_prices[:, 1],  'Close price')

all_metrics  = [m_open_ret, m_close_ret, m_open_px, m_close_px]

# Print table
print(f"\n{'Metric':<28} {'Open ret':>11} {'Close ret':>11} {'Open px':>11} {'Close px':>11}")
print("─" * 76)
for key in ['R²', 'MSE', 'RMSE', 'MAE', 'MAPE %', 'Dir Acc', 'Dir Acc (large moves)']:
    row = f"{key:<28}"
    for m in all_metrics:
        v = m[key]
        row += f" {v:>11.4f}" if not np.isnan(v) else f" {'n/a':>11}"
    print(row)

# ──────────────────────────────────────────────
# B. VISUALISATION — 3-panel figure
#    Panel 1: metric bar chart (returns only — the honest ones)
#    Panel 2: actual vs predicted returns scatter
#    Panel 3: training & validation loss curve
# ──────────────────────────────────────────────

GRAY   = '#6b6b6b'
TEAL   = '#1D9E75'
CORAL  = '#D85A30'
PURPLE = '#7F77DD'
AMBER  = '#EF9F27'
BG     = '#f8f8f6'

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle('Residual Stacked GRU — model performance', fontsize=15, fontweight='500', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                       left=0.07, right=0.97, top=0.93, bottom=0.08)

ax_bar    = fig.add_subplot(gs[0, :2])   # top-left wide: bar chart
ax_loss   = fig.add_subplot(gs[0, 2])    # top-right:     loss curve
ax_sc_op  = fig.add_subplot(gs[1, 0])   # bottom-left:   Open scatter
ax_sc_cl  = fig.add_subplot(gs[1, 1])   # bottom-mid:    Close scatter
ax_resid  = fig.add_subplot(gs[1, 2])   # bottom-right:  residual dist

for ax in [ax_bar, ax_loss, ax_sc_op, ax_sc_cl, ax_resid]:
    ax.set_facecolor(BG)

# ── Panel 1: Metric bars (returns metrics only) ─────────────────
metrics_to_plot = ['Dir Acc', 'Dir Acc (large moves)']
# Normalise R² to 0–1 for display (clamp negative to 0)
labels  = ['Dir Acc\n(overall)', 'Dir Acc\n(large moves)']
open_v  = [max(m_open_ret[k],  0) for k in metrics_to_plot]
close_v = [max(m_close_ret[k], 0) for k in metrics_to_plot]

x      = np.arange(len(labels))
width  = 0.3
bars_o = ax_bar.bar(x - width/2, open_v,  width, color=TEAL,   label='Open',  zorder=3)
bars_c = ax_bar.bar(x + width/2, close_v, width, color=CORAL,  label='Close', zorder=3)

# Reference line at 0.5 (coin flip)
ax_bar.axhline(0.5, color=GRAY, linewidth=1, linestyle='--', zorder=2)
ax_bar.text(len(labels) - 0.05, 0.515, 'coin flip (0.50)', color=GRAY,
            fontsize=9, ha='right')

ax_bar.set_ylim(0, 1.0)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, fontsize=10)
ax_bar.set_ylabel('Accuracy', fontsize=10)
ax_bar.set_title('Directional accuracy on log-returns  (primary metric)', fontsize=11)
ax_bar.legend(fontsize=9)
ax_bar.grid(axis='y', alpha=0.3, zorder=1)
ax_bar.spines[['top','right']].set_visible(False)

# Value annotations
for bar in bars_o:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, color=TEAL, fontweight='500')
for bar in bars_c:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9, color=CORAL, fontweight='500')

# Add R² and MAE as text annotations in the plot
info = (f"Open returns   →  R²: {m_open_ret['R²']:.3f}   MAE: {m_open_ret['MAE']:.5f}\n"
        f"Close returns  →  R²: {m_close_ret['R²']:.3f}   MAE: {m_close_ret['MAE']:.5f}")
ax_bar.text(0.01, 0.06, info, transform=ax_bar.transAxes,
            fontsize=8.5, color=GRAY, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='#ccc'))

# ── Panel 2: Training & validation loss ─────────────────────────
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
epochs_ran = range(1, len(train_loss) + 1)

ax_loss.plot(epochs_ran, train_loss, color=PURPLE, linewidth=1.5, label='Train loss')
ax_loss.plot(epochs_ran, val_loss,   color=AMBER,  linewidth=1.5, label='Val loss', linestyle='--')

best_epoch = int(np.argmin(val_loss)) + 1
best_val   = min(val_loss)
ax_loss.axvline(best_epoch, color=GRAY, linewidth=0.8, linestyle=':')
ax_loss.annotate(f'best: ep {best_epoch}\n({best_val:.4f})',
                 xy=(best_epoch, best_val),
                 xytext=(best_epoch + max(1, len(train_loss)//8), best_val * 1.15),
                 fontsize=8, color=GRAY,
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8))

ax_loss.set_title('Training vs validation loss', fontsize=11)
ax_loss.set_xlabel('Epoch', fontsize=9)
ax_loss.set_ylabel('Huber + directional loss', fontsize=9)
ax_loss.legend(fontsize=9)
ax_loss.grid(alpha=0.3)
ax_loss.spines[['top','right']].set_visible(False)

# ── Panel 3 & 4: Actual vs predicted scatter (returns) ──────────
for ax, true_r, pred_r, color, title in [
    (ax_sc_op, true_returns[:, 0], pred_returns[:, 0], TEAL,  'Open returns'),
    (ax_sc_cl, true_returns[:, 1], pred_returns[:, 1], CORAL, 'Close returns'),
]:
    ax.scatter(true_r, pred_r, alpha=0.25, s=8, color=color, zorder=3)

    # Perfect-prediction diagonal
    lim = max(np.abs(true_r).max(), np.abs(pred_r).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, linewidth=1,
            linestyle='--', zorder=2, label='Perfect prediction')

    # Quadrant shading: top-left & bottom-right = wrong direction
    ax.fill_between([-lim, 0], [0, 0], [lim, lim],
                    color='#E24B4A', alpha=0.06, zorder=1)
    ax.fill_between([0, lim], [-lim, -lim], [0, 0],
                    color='#E24B4A', alpha=0.06, zorder=1)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color=GRAY, linewidth=0.5)
    ax.axvline(0, color=GRAY, linewidth=0.5)
    ax.set_xlabel('Actual return', fontsize=9)
    ax.set_ylabel('Predicted return', fontsize=9)
    ax.set_title(f'Actual vs predicted — {title}', fontsize=10)
    ax.grid(alpha=0.2, zorder=0)
    ax.spines[['top','right']].set_visible(False)

    # Annotate wrong-direction quadrants
    ax.text(-lim * 0.9,  lim * 0.85, 'wrong\ndirection', fontsize=7,
            color='#E24B4A', ha='left', alpha=0.7)
    ax.text( lim * 0.05, -lim * 0.9, 'wrong\ndirection', fontsize=7,
            color='#E24B4A', ha='left', alpha=0.7)

    r2_val = sklearn.metrics.r2_score(true_r, pred_r)
    da_val = np.mean(np.sign(true_r) == np.sign(pred_r))
    ax.text(0.97, 0.05,
            f'R²: {r2_val:.3f}\nDir: {da_val:.3f}',
            transform=ax.transAxes, fontsize=8.5, ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='#ccc'))

# ── Panel 5: Residual distribution ──────────────────────────────
resid_open  = pred_returns[:, 0] - true_returns[:, 0]
resid_close = pred_returns[:, 1] - true_returns[:, 1]

ax_resid.hist(resid_open,  bins=40, color=TEAL,  alpha=0.55, label='Open',  density=True)
ax_resid.hist(resid_close, bins=40, color=CORAL, alpha=0.55, label='Close', density=True)
ax_resid.axvline(0, color=GRAY, linewidth=1, linestyle='--')
ax_resid.axvline(resid_open.mean(),  color=TEAL,  linewidth=1.2, linestyle=':')
ax_resid.axvline(resid_close.mean(), color=CORAL, linewidth=1.2, linestyle=':')

ax_resid.set_title('Prediction error distribution\n(pred − actual return)', fontsize=10)
ax_resid.set_xlabel('Residual', fontsize=9)
ax_resid.set_ylabel('Density', fontsize=9)
ax_resid.legend(fontsize=9)
ax_resid.grid(alpha=0.3)
ax_resid.spines[['top','right']].set_visible(False)

ax_resid.text(0.97, 0.95,
              f'Open  bias: {resid_open.mean():+.5f}\nClose bias: {resid_close.mean():+.5f}',
              transform=ax_resid.transAxes, fontsize=8, ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75, edgecolor='#ccc'))

plt.savefig('gru_v4_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved → gru_v4_performance.png")


# 1. Save the trained Keras model
model.save('gru_v4.keras')
print("Model saved as 'gru_v4.keras'")

# 2. Save the feature_scaler
joblib.dump(feature_scaler, 'feature_scaler.joblib')
print("Feature scaler saved as 'feature_scaler.joblib'")

# 3. Save the target_scaler
joblib.dump(target_scaler, 'target_scaler.joblib')
print("Target scaler saved as 'target_scaler.joblib'")