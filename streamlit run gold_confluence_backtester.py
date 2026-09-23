import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# ============================================================
# XAUUSD MASTER CONFLUENCE BACKTESTER
# S&D + Liquidity Sweep + Supertrend + Volume + MACD
# ============================================================

st.set_page_config(
    page_title="XAUUSD Master Confluence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helpers
# -----------------------------
def fmt_money(x):
    return f"${x:,.2f}"


def calculate_supertrend(df, atr_length=20, factor=2.0):
    """
    Standard Supertrend using Wilder-style ATR.
    Returns:
      Supertrend, Direction
      Direction = 1 bullish, -1 bearish
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / atr_length,
        adjust=False,
        min_periods=atr_length,
    ).mean()

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + factor * atr
    basic_lower = hl2 - factor * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    direction = pd.Series(index=df.index, dtype=float)
    supertrend = pd.Series(index=df.index, dtype=float)

    direction.iloc[0] = 1
    supertrend.iloc[0] = np.nan

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            direction.iloc[i] = direction.iloc[i - 1]
            final_upper.iloc[i] = basic_upper.iloc[i]
            final_lower.iloc[i] = basic_lower.iloc[i]
            continue

        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or close.iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        elif supertrend.iloc[i - 1] == final_lower.iloc[i - 1]:
            if close.iloc[i] >= final_lower.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
        else:
            direction.iloc[i] = 1 if close.iloc[i] >= hl2.iloc[i] else -1

        supertrend.iloc[i] = (
            final_lower.iloc[i]
            if direction.iloc[i] == 1
            else final_upper.iloc[i]
        )

    # A cleaner implementation of the first valid Supertrend state.
    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            continue

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = final_lower.iloc[i]
        else:
            supertrend.iloc[i] = final_upper.iloc[i]

    return supertrend, direction, atr


def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - macd_signal

    return macd, macd_signal, histogram


# ============================================================
# HEADER
# ============================================================

st.title("XAUUSD Master Confluence Backtester")
st.caption(
    "Supply & Demand • Liquidity Sweep • Supertrend (ATR 20 / Factor 2) • Volume • MACD"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Backtest Settings")

    symbol = st.selectbox(
        "Instrument",
        ["XAUUSD=X", "GC=F"],
        index=0,
        help="XAUUSD=X = Yahoo spot gold. GC=F = COMEX gold futures.",
    )

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=100.0,
        value=1000.0,
        step=100.0,
    )

    position_size = st.number_input(
        "Position Size (oz)",
        min_value=0.01,
        value=1.0,
        step=0.01,
        help="P&L = gold price movement × position size in ounces.",
    )

    st.subheader("S&D + Liquidity")

    momentum_mult = st.slider(
        "S&D Expansion Threshold (ATR)",
        1.0,
        4.0,
        1.5,
        0.1,
    )

    pivot_len = st.slider(
        "Liquidity Swing Length",
        2,
        10,
        2,
    )

    st.subheader("Supertrend")

    # Requested settings: ATR length 20 and factor 2.
    supertrend_atr = st.number_input(
        "Supertrend ATR Length",
        min_value=2,
        max_value=100,
        value=20,
        step=1,
    )

    supertrend_factor = st.number_input(
        "Supertrend Factor",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    st.subheader("Volume")

    volume_length = st.number_input(
        "Volume Average Length",
        min_value=2,
        max_value=100,
        value=20,
        step=1,
    )

    volume_multiplier = st.number_input(
        "Minimum Volume × Average",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="1.0 means current volume must be at least the 20-bar average.",
    )

    st.subheader("MACD")

    macd_fast = st.number_input(
        "MACD Fast",
        min_value=2,
        max_value=50,
        value=12,
        step=1,
    )

    macd_slow = st.number_input(
        "MACD Slow",
        min_value=3,
        max_value=100,
        value=26,
        step=1,
    )

    macd_signal_len = st.number_input(
        "MACD Signal",
        min_value=2,
        max_value=50,
        value=9,
        step=1,
    )

    st.subheader("Risk")

    rr_ratio = st.slider(
        "Risk / Reward",
        0.5,
        5.0,
        2.0,
        0.5,
    )

    sl_atr = st.slider(
        "SL ATR Buffer",
        0.0,
        1.0,
        0.20,
        0.05,
    )

    mode = st.radio(
        "S&D Entry Mode",
        ["Strict Confluence", "Relaxed"],
        index=0,
        help=(
            "Strict = liquidity sweep must occur inside an active S&D zone. "
            "Relaxed = sweep OR zone tap can create the base signal."
        ),
    )

    st.subheader("Data")

    days_history = st.slider(
        "Days of 15m Data",
        5,
        59,
        59,
    )

    st.subheader("Chart")

    show_zones = st.checkbox("Show S&D Zones", value=True)
    show_liquidity = st.checkbox("Show Liquidity", value=True)
    show_supertrend = st.checkbox("Show Supertrend", value=True)
    show_trade_levels = st.checkbox("Show Entry / SL / TP", value=True)

    st.button("🚀 Run Backtest", use_container_width=True)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=900)
def get_data(ticker, days):
    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="15m",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df.columns = df.columns.droplevel(-1)
        else:
            df.columns = df.columns.droplevel(1)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]

    return df


with st.spinner("Downloading data and calculating confluence..."):
    df = get_data(symbol, days_history)

if df.empty:
    st.error("No data was returned. Try another instrument or data period.")
    st.stop()


# ============================================================
# INDICATORS
# ============================================================

# ----- ATR for S&D / risk -----
prev_close = df["Close"].shift(1)

tr = pd.concat(
    [
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ],
    axis=1,
).max(axis=1)

df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
df["Body"] = (df["Close"] - df["Open"]).abs()


# ----- Supertrend: ATR 20 / Factor 2 -----
(
    df["Supertrend"],
    df["ST_Direction"],
    df["ST_ATR"],
) = calculate_supertrend(
    df,
    atr_length=int(supertrend_atr),
    factor=float(supertrend_factor),
)

df["ST_Bullish"] = df["ST_Direction"] == 1
df["ST_Bearish"] = df["ST_Direction"] == -1


# ----- Volume filter -----
df["Volume_Avg"] = (
    df["Volume"]
    .rolling(int(volume_length))
    .mean()
)

df["Volume_OK"] = (
    df["Volume"]
    >= df["Volume_Avg"] * float(volume_multiplier)
)


# ----- MACD -----
(
    df["MACD"],
    df["MACD_Signal"],
    df["MACD_Hist"],
) = calculate_macd(
    df["Close"],
    fast=int(macd_fast),
    slow=int(macd_slow),
    signal=int(macd_signal_len),
)

df["MACD_Bullish"] = (
    (df["MACD"] > df["MACD_Signal"])
    & (df["MACD_Hist"] > 0)
)

df["MACD_Bearish"] = (
    (df["MACD"] < df["MACD_Signal"])
    & (df["MACD_Hist"] < 0)
)


# ============================================================
# LIQUIDITY ENGINE
# ============================================================

# Confirmed pivots only.
# The shift prevents future candles from being used before the
# pivot is actually known.

raw_pivot_high = (
    df["High"]
    .rolling(
        window=pivot_len * 2 + 1,
        center=True,
    )
    .max()
)

raw_pivot_low = (
    df["Low"]
    .rolling(
        window=pivot_len * 2 + 1,
        center=True,
    )
    .min()
)

df["Confirmed_Pivot_High"] = raw_pivot_high.shift(pivot_len)
df["Confirmed_Pivot_Low"] = raw_pivot_low.shift(pivot_len)

df["Liquidity_High"] = (
    df["Confirmed_Pivot_High"].ffill()
)

df["Liquidity_Low"] = (
    df["Confirmed_Pivot_Low"].ffill()
)

df["Bull_Sweep"] = (
    df["Liquidity_Low"].notna()
    & (df["Low"] < df["Liquidity_Low"])
    & (df["Close"] > df["Liquidity_Low"])
)

df["Bear_Sweep"] = (
    df["Liquidity_High"].notna()
    & (df["High"] > df["Liquidity_High"])
    & (df["Close"] < df["Liquidity_High"])
)


# ============================================================
# BACKTEST ENGINE
# ============================================================

balance = float(initial_capital)

trade_log = []
equity_by_time = {}

in_pos = False
pos_type = None

entry_price = 0.0
sl = 0.0
tp = 0.0

entry_date = None
entry_reason = ""
entry_risk = 0.0

active_demand_zones = []
active_supply_zones = []
zone_history = []

for i in range(1, len(df)):

    row = df.iloc[i]
    date = df.index[i]
    prev_row = df.iloc[i - 1]

    # ========================================================
    # 1. MANAGE OPEN POSITION
    # ========================================================

    if in_pos:

        exit_price = None
        result = None
        exit_reason = None

        if pos_type == "LONG":

            hit_sl = row["Low"] <= sl
            hit_tp = row["High"] >= tp

            # Conservative candle assumption:
            # if both levels are touched, SL is considered first.
            if hit_sl:
                exit_price = sl
                pnl = (exit_price - entry_price) * position_size
                result = "Loss"
                exit_reason = "Stop Loss"

            elif hit_tp:
                exit_price = tp
                pnl = (exit_price - entry_price) * position_size
                result = "Win"
                exit_reason = "Take Profit"

        else:

            hit_sl = row["High"] >= sl
            hit_tp = row["Low"] <= tp

            if hit_sl:
                exit_price = sl
                pnl = (entry_price - exit_price) * position_size
                result = "Loss"
                exit_reason = "Stop Loss"

            elif hit_tp:
                exit_price = tp
                pnl = (entry_price - exit_price) * position_size
                result = "Win"
                exit_reason = "Take Profit"

        if exit_price is not None:

            balance += pnl

            risk_dollars = entry_risk * position_size

            r_multiple = (
                pnl / risk_dollars
                if risk_dollars > 0
                else np.nan
            )

            trade_log.append(
                {
                    "Entry Date": entry_date,
                    "Exit Date": date,
                    "Type": pos_type,
                    "Entry": entry_price,
                    "Exit": exit_price,
                    "Target": tp,
                    "StopLoss": sl,
                    "Result": result,
                    "Exit Reason": exit_reason,
                    "PnL": pnl,
                    "R": r_multiple,
                    "Reason": entry_reason,
                }
            )

            in_pos = False
            pos_type = None
            entry_date = None
            entry_reason = ""
            entry_risk = 0.0

    equity_by_time[date] = balance

    # ========================================================
    # 2. BUILD BASE S&D / LIQUIDITY SIGNAL
    # ========================================================

    if not in_pos:

        zone_tapped_long = any(
            row["Low"] <= z["top"]
            and row["Low"] >= z["bottom"]
            for z in active_demand_zones
        )

        zone_tapped_short = any(
            row["High"] >= z["bottom"]
            and row["High"] <= z["top"]
            for z in active_supply_zones
        )

        if mode == "Relaxed":

            long_base = bool(
                row["Bull_Sweep"]
                or zone_tapped_long
            )

            short_base = bool(
                row["Bear_Sweep"]
                or zone_tapped_short
            )

        else:

            long_base = bool(
                row["Bull_Sweep"]
                and zone_tapped_long
            )

            short_base = bool(
                row["Bear_Sweep"]
                and zone_tapped_short
            )

        # ====================================================
        # 3. INDICATOR CONFLUENCE
        # ====================================================

        long_filters = {
            "Supertrend": bool(row["ST_Bullish"]),
            "Volume": bool(row["Volume_OK"]),
            "MACD": bool(row["MACD_Bullish"]),
        }

        short_filters = {
            "Supertrend": bool(row["ST_Bearish"]),
            "Volume": bool(row["Volume_OK"]),
            "MACD": bool(row["MACD_Bearish"]),
        }

        long_confluence = all(long_filters.values())
        short_confluence = all(short_filters.values())

        long_signal = (
            long_base
            and long_confluence
        )

        short_signal = (
            short_base
            and short_confluence
        )

        # ====================================================
        # 4. ENTER LONG
        # ====================================================

        if long_signal and not short_signal:

            entry_price = float(row["Close"])

            sl = float(
                row["Low"]
                - row["ATR"] * sl_atr
            )

            risk = entry_price - sl

            if risk > 0 and np.isfinite(risk):

                tp = (
                    entry_price
                    + risk * rr_ratio
                )

                pos_type = "LONG"
                in_pos = True

                entry_date = date
                entry_risk = risk

                entry_reason = (
                    "LONG | "
                    f"S&D/Liquidity ✓ | "
                    f"Supertrend Bull ✓ | "
                    f"Volume ✓ | "
                    f"MACD Bull ✓"
                )

        # ====================================================
        # 5. ENTER SHORT
        # ====================================================

        elif short_signal and not long_signal:

            entry_price = float(row["Close"])

            sl = float(
                row["High"]
                + row["ATR"] * sl_atr
            )

            risk = sl - entry_price

            if risk > 0 and np.isfinite(risk):

                tp = (
                    entry_price
                    - risk * rr_ratio
                )

                pos_type = "SHORT"
                in_pos = True

                entry_date = date
                entry_risk = risk

                entry_reason = (
                    "SHORT | "
                    f"S&D/Liquidity ✓ | "
                    f"Supertrend Bear ✓ | "
                    f"Volume ✓ | "
                    f"MACD Bear ✓"
                )

    # ========================================================
    # 6. REMOVE BROKEN ZONES
    # ========================================================

    active_demand_zones = [
        z
        for z in active_demand_zones
        if row["Close"] > z["bottom"]
    ]

    active_supply_zones = [
        z
        for z in active_supply_zones
        if row["Close"] < z["top"]
    ]

    # ========================================================
    # 7. CREATE NEW S&D ZONES
    # ========================================================

    is_expansion = (
        np.isfinite(row["ATR"])
        and row["Body"]
        > row["ATR"] * momentum_mult
    )

    if is_expansion:

        zone = {
            "top": float(prev_row["High"]),
            "bottom": float(prev_row["Low"]),
            "start": date,
            "type": (
                "DEMAND"
                if row["Close"] > row["Open"]
                else "SUPPLY"
            ),
        }

        if zone["type"] == "DEMAND":
            active_demand_zones.append(zone)
        else:
            active_supply_zones.append(zone)

        zone_history.append(zone)


# ============================================================
# RESULTS
# ============================================================

trade_df = pd.DataFrame(trade_log)

equity_series = pd.Series(
    equity_by_time,
    dtype=float,
)

if not equity_series.empty:

    equity_series = equity_series.sort_index()

    equity_series = pd.concat(
        [
            pd.Series(
                [initial_capital],
                index=[df.index[0]],
                dtype=float,
            ),
            equity_series,
        ]
    )

    equity_series = (
        equity_series[
            ~equity_series.index.duplicated(
                keep="last"
            )
        ]
    )

    df["Equity"] = (
        equity_series
        .reindex(df.index)
        .ffill()
        .fillna(initial_capital)
    )

else:

    df["Equity"] = initial_capital


# ============================================================
# PERFORMANCE METRICS
# ============================================================

total_trades = len(trade_df)

wins = (
    int((trade_df["Result"] == "Win").sum())
    if total_trades
    else 0
)

losses = (
    int((trade_df["Result"] == "Loss").sum())
    if total_trades
    else 0
)

win_rate = (
    wins / total_trades * 100
    if total_trades
    else 0.0
)

net_profit = balance - initial_capital

return_pct = (
    net_profit / initial_capital * 100
)

gross_profit = (
    trade_df.loc[
        trade_df["PnL"] > 0,
        "PnL",
    ].sum()
    if total_trades
    else 0.0
)

gross_loss = (
    abs(
        trade_df.loc[
            trade_df["PnL"] < 0,
            "PnL",
        ].sum()
    )
    if total_trades
    else 0.0
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else np.inf
)

avg_win = (
    trade_df.loc[
        trade_df["PnL"] > 0,
        "PnL",
    ].mean()
    if wins
    else 0.0
)

avg_loss = (
    trade_df.loc[
        trade_df["PnL"] < 0,
        "PnL",
    ].mean()
    if losses
    else 0.0
)

expectancy = (
    trade_df["PnL"].mean()
    if total_trades
    else 0.0
)

avg_r = (
    trade_df["R"].mean()
    if total_trades
    else 0.0
)

peak = df["Equity"].cummax()

drawdown = (
    df["Equity"] - peak
)

drawdown_pct = (
    df["Equity"] / peak - 1.0
) * 100

max_drawdown = (
    float(drawdown.min())
    if len(drawdown)
    else 0.0
)

max_drawdown_pct = (
    float(drawdown_pct.min())
    if len(drawdown_pct)
    else 0.0
)


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Balance",
    fmt_money(balance),
    f"{net_profit:+,.2f}",
)

c2.metric(
    "Net Return",
    f"{return_pct:+.2f}%",
)

c3.metric(
    "Win Rate",
    f"{win_rate:.1f}%",
)

c4.metric(
    "Profit Factor",
    (
        "∞"
        if np.isinf(profit_factor)
        else f"{profit_factor:.2f}"
    ),
)

c5.metric(
    "Max Drawdown",
    fmt_money(max_drawdown),
    f"{max_drawdown_pct:.2f}%",
)

st.divider()

tab_dashboard, tab_trades, tab_analysis = st.tabs(
    [
        "📊 Dashboard",
        "📋 Trade Log",
        "🔍 Confluence Analysis",
    ]
)


# ============================================================
# DASHBOARD TAB
# ============================================================

with tab_dashboard:

    a1, a2, a3, a4, a5, a6 = st.columns(6)

    a1.metric("Trades", total_trades)
    a2.metric("Wins", wins)
    a3.metric("Losses", losses)
    a4.metric("Avg Win", fmt_money(avg_win))
    a5.metric("Avg Loss", fmt_money(avg_loss))
    a6.metric("Expectancy", fmt_money(expectancy))

    st.subheader("📈 Equity Curve")

    equity_fig = go.Figure()

    equity_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Equity"],
            mode="lines",
            name="Equity",
            line=dict(width=2),
        )
    )

    equity_fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        annotation_text="Starting Balance",
    )

    equity_fig.update_layout(
        height=320,
        template="plotly_dark",
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
        yaxis_title="Balance ($)",
    )

    st.plotly_chart(
        equity_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.subheader("🕯️ Price / Confluence Map")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        )
    )

    if show_liquidity:

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Liquidity_High"],
                mode="lines",
                line=dict(
                    width=1,
                    dash="dot",
                ),
                name="Liquidity High",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Liquidity_Low"],
                mode="lines",
                line=dict(
                    width=1,
                    dash="dot",
                ),
                name="Liquidity Low",
            )
        )

    if show_supertrend:

        bullish_st = df["ST_Direction"] == 1
        bearish_st = df["ST_Direction"] == -1

        st_bull = df["Supertrend"].where(bullish_st)
        st_bear = df["Supertrend"].where(bearish_st)

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=st_bull,
                mode="lines",
                name="Supertrend Bull",
                line=dict(width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=st_bear,
                mode="lines",
                name="Supertrend Bear",
                line=dict(width=2),
            )
        )

    if show_zones:

        for z in zone_history[-40:]:

            fill = (
                "rgba(0, 200, 120, 0.12)"
                if z["type"] == "DEMAND"
                else "rgba(255, 80, 80, 0.12)"
            )

            fig.add_shape(
                type="rect",
                x0=z["start"],
                x1=df.index[-1],
                y0=z["bottom"],
                y1=z["top"],
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            )

    if show_trade_levels and not trade_df.empty:

        longs = trade_df[
            trade_df["Type"] == "LONG"
        ]

        shorts = trade_df[
            trade_df["Type"] == "SHORT"
        ]

        if not longs.empty:

            fig.add_trace(
                go.Scatter(
                    x=longs["Entry Date"],
                    y=longs["Entry"],
                    mode="markers",
                    marker=dict(
                        size=13,
                        symbol="triangle-up",
                    ),
                    name="Long Entry",
                    text=longs["Reason"],
                    hovertemplate=(
                        "%{text}"
                        "<br>Entry: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        if not shorts.empty:

            fig.add_trace(
                go.Scatter(
                    x=shorts["Entry Date"],
                    y=shorts["Entry"],
                    mode="markers",
                    marker=dict(
                        size=13,
                        symbol="triangle-down",
                    ),
                    name="Short Entry",
                    text=shorts["Reason"],
                    hovertemplate=(
                        "%{text}"
                        "<br>Entry: %{y:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10,
        ),
        title=(
            f"{symbol} | "
            f"S&D + Liquidity + "
            f"Supertrend {int(supertrend_atr)}/"
            f"{supertrend_factor:g} + "
            f"Volume + MACD"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
        },
    )


# ============================================================
# TRADE LOG
# ============================================================

with tab_trades:

    if trade_df.empty:

        st.info(
            "No completed trades were generated. "
            "The full confluence filter may be too restrictive."
        )

    else:

        st.subheader("Completed Trades")

        display_df = trade_df.copy()

        for col in [
            "Entry",
            "Exit",
            "Target",
            "StopLoss",
            "PnL",
            "R",
        ]:
            display_df[col] = display_df[col].round(2)

        st.dataframe(
            display_df[
                [
                    "Entry Date",
                    "Exit Date",
                    "Type",
                    "Entry",
                    "Exit",
                    "Target",
                    "StopLoss",
                    "Result",
                    "Exit Reason",
                    "PnL",
                    "R",
                    "Reason",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        csv = trade_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Trade Log CSV",
            csv,
            file_name="xauusd_master_confluence_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# CONFLUENCE ANALYSIS
# ============================================================

with tab_analysis:

    st.subheader("🔎 Strategy Components")

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Supertrend",
        f"ATR {int(supertrend_atr)} / "
        f"Factor {supertrend_factor:g}",
    )

    b2.metric(
        "Volume Filter",
        f"{int(volume_length)} bars × "
        f"{volume_multiplier:g}",
    )

    b3.metric(
        "MACD",
        f"{int(macd_fast)}/"
        f"{int(macd_slow)}/"
        f"{int(macd_signal_len)}",
    )

    b4.metric(
        "Target R:R",
        f"1 : {rr_ratio:g}",
    )

    st.markdown(
        """
### Long setup

A long trade requires the S&D/liquidity base condition **and all three indicator confirmations**:

1. 🟢 **Supertrend bullish**
2. 📊 **Volume ≥ average volume × volume multiplier**
3. 📈 **MACD > signal and MACD histogram > 0**

### Short setup

A short trade requires the S&D/liquidity base condition **and all three bearish confirmations**:

1. 🔴 **Supertrend bearish**
2. 📊 **Volume ≥ average volume × volume multiplier**
3. 📉 **MACD < signal and MACD histogram < 0**

This makes the strategy a genuine multi-factor confluence model rather than allowing a single liquidity sweep to trigger a trade.
"""
    )

    if not trade_df.empty:

        long_count = int(
            (trade_df["Type"] == "LONG").sum()
        )

        short_count = int(
            (trade_df["Type"] == "SHORT").sum()
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric("Long Trades", long_count)
        q2.metric("Short Trades", short_count)
        q3.metric("Average R", f"{avg_r:.2f}R")
        q4.metric("Total P&L", fmt_money(net_profit))

        st.subheader("Trade P&L")

        pnl_fig = go.Figure()

        pnl_fig.add_trace(
            go.Bar(
                x=list(
                    range(
                        1,
                        len(trade_df) + 1,
                    )
                ),
                y=trade_df["PnL"],
                name="Trade P&L",
            )
        )

        pnl_fig.update_layout(
            height=350,
            template="plotly_dark",
            xaxis_title="Trade Number",
            yaxis_title="P&L ($)",
        )

        st.plotly_chart(
            pnl_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.subheader("Entry Reasons")

        reason_counts = (
            trade_df["Reason"]
            .value_counts()
            .rename_axis("Reason")
            .reset_index(name="Trades")
        )

        st.dataframe(
            reason_counts,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Backtest Assumptions")

    st.markdown(
        """
- **Timeframe:** 15 minutes.
- **Supertrend:** ATR length 20, factor 2 by default.
- **MACD:** 12 / 26 / 9 by default.
- **Volume:** current volume compared with a 20-bar average by default.
- **Liquidity:** confirmed pivots are used to reduce look-ahead bias.
- **Same-candle SL/TP:** if both are touched, the backtest assumes SL was hit first.
- **Spread/slippage:** not included.
- **Commission:** not included.
- **Position sizing:** price movement × position size in ounces.
- **Open position at the end:** not counted as a completed trade.
"""
    )

st.caption(
    "Educational backtesting tool. Historical results do not establish future trading performance."
)
