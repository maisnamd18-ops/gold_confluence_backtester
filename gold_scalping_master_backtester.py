import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(
    page_title="Gold Scalping Master",
    page_icon="🥇",
    layout="wide",
)

# ============================================================
# INDICATORS
# ============================================================

def calc_atr(df, length=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False,
                  min_periods=length).mean()


def calc_supertrend(df, length=20, factor=2.0):
    a = calc_atr(df, length)
    hl2 = (df["High"] + df["Low"]) / 2

    bu = hl2 + factor * a
    bl = hl2 - factor * a
    fu = bu.copy()
    fl = bl.copy()

    direction = pd.Series(np.nan, index=df.index)
    st = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        if pd.isna(a.iloc[i]):
            continue

        fu.iloc[i] = (
            bu.iloc[i]
            if bu.iloc[i] < fu.iloc[i-1]
            or df["Close"].iloc[i-1] > fu.iloc[i-1]
            else fu.iloc[i-1]
        )

        fl.iloc[i] = (
            bl.iloc[i]
            if bl.iloc[i] > fl.iloc[i-1]
            or df["Close"].iloc[i-1] < fl.iloc[i-1]
            else fl.iloc[i-1]
        )

        if pd.isna(direction.iloc[i-1]):
            direction.iloc[i] = 1 if df["Close"].iloc[i] >= hl2.iloc[i] else -1
        elif direction.iloc[i-1] == -1:
            direction.iloc[i] = 1 if df["Close"].iloc[i] > fu.iloc[i] else -1
        else:
            direction.iloc[i] = -1 if df["Close"].iloc[i] < fl.iloc[i] else 1

        st.iloc[i] = fl.iloc[i] if direction.iloc[i] == 1 else fu.iloc[i]

    return st, direction, a


def add_indicators(df):
    df = df.copy()
    df["ATR"] = calc_atr(df, 14)
    df["Body"] = (df["Close"] - df["Open"]).abs()

    df["Supertrend"], df["ST_Direction"], df["ST_ATR"] = calc_supertrend(
        df, 20, 2.0
    )

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["Volume_Avg"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_Avg"]

    return df


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=900)
def get_gold(interval, days):
    df = yf.download(
        "GC=F",
        period=f"{days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols if c in df.columns]].dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ============================================================
# 15M SETUP ENGINE
# ============================================================

def create_15m_setups(df, pivot_len=3, expansion_atr=1.5):
    df = df.copy()

    # Confirmed pivots. The shift delays availability so future
    # candles are not used as if they were already known.
    raw_high = df["High"].rolling(
        pivot_len * 2 + 1, center=True
    ).max()

    raw_low = df["Low"].rolling(
        pivot_len * 2 + 1, center=True
    ).min()

    df["Liquidity_High"] = raw_high.shift(pivot_len).ffill()
    df["Liquidity_Low"] = raw_low.shift(pivot_len).ffill()

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

    active_demand = []
    active_supply = []
    zones = []

    bull_zone = []
    bear_zone = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        active_demand = [
            z for z in active_demand
            if row["Close"] > z["bottom"]
        ]
        active_supply = [
            z for z in active_supply
            if row["Close"] < z["top"]
        ]

        expansion = (
            pd.notna(row["ATR"])
            and row["Body"] > row["ATR"] * expansion_atr
        )

        if expansion:
            z = {
                "top": float(prev["High"]),
                "bottom": float(prev["Low"]),
                "start": df.index[i],
            }

            if row["Close"] > row["Open"]:
                z["type"] = "DEMAND"
                active_demand.append(z)
                zones.append(z.copy())
            elif row["Close"] < row["Open"]:
                z["type"] = "SUPPLY"
                active_supply.append(z)
                zones.append(z.copy())

        long_zone = any(
            row["Low"] <= z["top"] and row["Low"] >= z["bottom"]
            for z in active_demand
        )

        short_zone = any(
            row["High"] >= z["bottom"] and row["High"] <= z["top"]
            for z in active_supply
        )

        bull_zone.append(long_zone)
        bear_zone.append(short_zone)

    # align the first bar
    df["Demand_Tap"] = [False] + bull_zone
    df["Supply_Tap"] = [False] + bear_zone

    # Core 15m setup: sweep + corresponding zone
    df["Long_Setup"] = df["Bull_Sweep"] & df["Demand_Tap"]
    df["Short_Setup"] = df["Bear_Sweep"] & df["Supply_Tap"]

    return df, zones


# ============================================================
# 5M CONFIRMATION
# ============================================================

def confirm_5m(df5, timestamp, side, volume_min):
    """
    Confirmation uses the first completed 5m candle at or after
    the 15m setup timestamp. It does not use a future 5m candle
    to create a signal before that candle closes.
    """
    if df5.empty:
        return None

    candidates = df5[df5.index >= timestamp]
    if candidates.empty:
        return None

    row = candidates.iloc[0]
    t = candidates.index[0]

    # We require the 5m candle itself to close with confirmation.
    if pd.isna(row["ST_Direction"]) or pd.isna(row["MACD_Signal"]):
        return None

    vol_ok = (
        pd.notna(row["Volume_Ratio"])
        and row["Volume_Ratio"] >= volume_min
    )

    if side == "LONG":
        st_ok = row["ST_Direction"] == 1
        macd_ok = (
            row["MACD"] > row["MACD_Signal"]
            and row["MACD_Hist"] > 0
        )
        candle_ok = row["Close"] > row["Open"]
        confirmed = st_ok and macd_ok and vol_ok and candle_ok

    else:
        st_ok = row["ST_Direction"] == -1
        macd_ok = (
            row["MACD"] < row["MACD_Signal"]
            and row["MACD_Hist"] < 0
        )
        candle_ok = row["Close"] < row["Open"]
        confirmed = st_ok and macd_ok and vol_ok and candle_ok

    score = int(st_ok) + int(macd_ok) + int(vol_ok) + int(candle_ok)

    return {
        "time": t,
        "confirmed": confirmed,
        "score": score,
        "st_ok": st_ok,
        "macd_ok": macd_ok,
        "vol_ok": vol_ok,
        "candle_ok": candle_ok,
        "close": float(row["Close"]),
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df15, df5, initial_capital, size_oz,
                 rr, sl_buffer, min_score, volume_min,
                 require_zone=True):

    df15, zones = create_15m_setups(
        df15,
        pivot_len=3,
        expansion_atr=1.5,
    )

    balance = float(initial_capital)
    equity = []
    trades = []

    in_trade = False
    side = None
    entry = sl = tp = 0.0
    entry_time = None
    confirm_time = None
    reason = ""
    risk = 0.0

    # setup lock prevents repeatedly using the same 15m setup
    last_setup_time = None

    for i in range(len(df15)):
        t = df15.index[i]
        row = df15.iloc[i]

        # -------------------------
        # Manage existing trade
        # -------------------------
        if in_trade:
            if side == "LONG":
                hit_sl = row["Low"] <= sl
                hit_tp = row["High"] >= tp

                if hit_sl:
                    exit_price = sl
                    pnl = (exit_price - entry) * size_oz
                    result = "Loss"
                elif hit_tp:
                    exit_price = tp
                    pnl = (exit_price - entry) * size_oz
                    result = "Win"
                else:
                    equity.append(balance)
                    continue

            else:
                hit_sl = row["High"] >= sl
                hit_tp = row["Low"] <= tp

                if hit_sl:
                    exit_price = sl
                    pnl = (entry - exit_price) * size_oz
                    result = "Loss"
                elif hit_tp:
                    exit_price = tp
                    pnl = (entry - exit_price) * size_oz
                    result = "Win"
                else:
                    equity.append(balance)
                    continue

            balance += pnl
            r_mult = pnl / (risk * size_oz) if risk > 0 else np.nan

            trades.append({
                "Entry Time": entry_time,
                "Confirmation": confirm_time,
                "Exit Time": t,
                "Side": side,
                "Entry": entry,
                "Exit": exit_price,
                "SL": sl,
                "TP": tp,
                "Result": result,
                "PnL": pnl,
                "R": r_mult,
                "Reason": reason,
            })

            in_trade = False
            side = None

        equity.append(balance)

        # -------------------------
        # Find new setup
        # -------------------------
        if in_trade:
            continue

        setup_long = bool(row["Long_Setup"])
        setup_short = bool(row["Short_Setup"])

        if not setup_long and not setup_short:
            continue

        if last_setup_time == t:
            continue

        candidates = []

        if setup_long:
            c = confirm_5m(
                df5, t, "LONG", volume_min
            )
            if c and c["score"] >= min_score and c["confirmed"]:
                candidates.append(("LONG", c))

        if setup_short:
            c = confirm_5m(
                df5, t, "SHORT", volume_min
            )
            if c and c["score"] >= min_score and c["confirmed"]:
                candidates.append(("SHORT", c))

        if not candidates:
            continue

        # Normally only one side exists. If both appear, skip rather
        # than arbitrarily choosing a direction.
        if len(candidates) > 1:
            continue

        side, conf = candidates[0]

        entry = conf["close"]
        confirm_time = conf["time"]
        entry_time = t

        if side == "LONG":
            sl = float(row["Low"] - row["ATR"] * sl_buffer)
            risk = entry - sl
            if risk <= 0 or not np.isfinite(risk):
                continue
            tp = entry + risk * rr

        else:
            sl = float(row["High"] + row["ATR"] * sl_buffer)
            risk = sl - entry
            if risk <= 0 or not np.isfinite(risk):
                continue
            tp = entry - risk * rr

        reason = (
            f"{side} | 15m S&D + Liquidity Sweep | "
            f"5m ST20/2 + MACD + Volume | "
            f"Score {conf['score']}/4"
        )

        in_trade = True
        last_setup_time = t

    # Do not manufacture an exit for an open position.
    trade_df = pd.DataFrame(trades)

    eq = pd.Series(
        equity,
        index=df15.index[:len(equity)],
        dtype=float,
    )

    if eq.empty:
        eq = pd.Series(
            [initial_capital],
            index=[df15.index[0]],
        )

    return df15, zones, trade_df, eq


# ============================================================
# UI
# ============================================================

st.title("🥇 Gold Scalping Master")
st.caption(
    "15M S&D + Liquidity Sweep → 5M Supertrend 20/2 + MACD + Volume confirmation"
)

with st.sidebar:
    st.header("Strategy")

    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=100.0,
        value=1000.0,
        step=100.0,
    )

    size_oz = st.number_input(
        "Position Size (oz)",
        min_value=0.01,
        value=0.01,
        step=0.01,
    )

    rr = st.slider(
        "Risk : Reward",
        1.0, 5.0, 2.0, 0.5
    )

    sl_buffer = st.slider(
        "SL ATR Buffer",
        0.0, 1.0, 0.20, 0.05
    )

    min_score = st.slider(
        "5M Confirmation Score",
        2, 4, 3,
        help="Score components: Supertrend, MACD, Volume, candle direction.",
    )

    volume_min = st.slider(
        "5M Volume Ratio",
        0.5, 3.0, 1.0, 0.1
    )

    days = st.slider(
        "15M History (days)",
        5, 59, 30
    )

    st.divider()
    st.subheader("Fixed Indicators")
    st.write("Supertrend: **ATR 20 / Factor 2**")
    st.write("MACD: **12 / 26 / 9**")
    st.write("Volume: **20-bar average**")
    st.write("Setup TF: **15 minutes**")
    st.write("Confirmation TF: **5 minutes**")

    run = st.button(
        "🚀 RUN BACKTEST",
        use_container_width=True,
    )

if "run" not in st.session_state:
    st.session_state.run = True

if run:
    st.session_state.run = True

if st.session_state.run:

    with st.spinner("Downloading Gold 15m + 5m data..."):
        df15_raw = get_gold("15m", days)
        df5_raw = get_gold("5m", min(days, 59))

    if df15_raw.empty or df5_raw.empty:
        st.error(
            "Yahoo Finance did not return enough intraday data."
        )
        st.stop()

    df15_raw = add_indicators(df15_raw)
    df5_raw = add_indicators(df5_raw)

    with st.spinner("Running non-lookahead backtest..."):
        df15, zones, trades, equity = run_backtest(
            df15_raw,
            df5_raw,
            initial_capital,
            size_oz,
            rr,
            sl_buffer,
            min_score,
            volume_min,
        )

    # -------------------------
    # Metrics
    # -------------------------
    total = len(trades)
    wins = int((trades["Result"] == "Win").sum()) if total else 0
    losses = int((trades["Result"] == "Loss").sum()) if total else 0

    win_rate = wins / total * 100 if total else 0
    net = trades["PnL"].sum() if total else 0
    final_balance = initial_capital + net

    gp = trades.loc[trades["PnL"] > 0, "PnL"].sum() if total else 0
    gl = abs(trades.loc[trades["PnL"] < 0, "PnL"].sum()) if total else 0
    pf = gp / gl if gl > 0 else np.inf

    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min() if len(dd) else 0
    max_dd_pct = (
        ((equity / peak) - 1).min() * 100
        if len(equity)
        else 0
    )

    # -------------------------
    # Dashboard
    # -------------------------
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Final Balance",
        f"${final_balance:,.2f}",
        f"{net:+,.2f}",
    )
    c2.metric("Trades", total)
    c3.metric("Win Rate", f"{win_rate:.1f}%")
    c4.metric(
        "Profit Factor",
        "∞" if np.isinf(pf) else f"{pf:.2f}",
    )
    c5.metric(
        "Max Drawdown",
        f"${max_dd:,.2f}",
        f"{max_dd_pct:.2f}%",
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Chart", "📋 Trades", "🔬 Diagnostics"]
    )

    # -------------------------
    # Chart
    # -------------------------
    with tab1:
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df15.index,
                open=df15["Open"],
                high=df15["High"],
                low=df15["Low"],
                close=df15["Close"],
                name="Gold 15M",
            )
        )

        bull_st = df15["Supertrend"].where(
            df15["ST_Direction"] == 1
        )
        bear_st = df15["Supertrend"].where(
            df15["ST_Direction"] == -1
        )

        fig.add_trace(
            go.Scatter(
                x=df15.index,
                y=bull_st,
                mode="lines",
                name="Supertrend Bull",
                line=dict(width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df15.index,
                y=bear_st,
                mode="lines",
                name="Supertrend Bear",
                line=dict(width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df15.index,
                y=df15["Liquidity_High"],
                mode="lines",
                name="Buy-side Liquidity",
                line=dict(dash="dot", width=1),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df15.index,
                y=df15["Liquidity_Low"],
                mode="lines",
                name="Sell-side Liquidity",
                line=dict(dash="dot", width=1),
            )
        )

        for z in zones[-30:]:
            fill = (
                "rgba(0,200,120,0.10)"
                if z["type"] == "DEMAND"
                else "rgba(255,70,70,0.10)"
            )

            fig.add_shape(
                type="rect",
                x0=z["start"],
                x1=df15.index[-1],
                y0=z["bottom"],
                y1=z["top"],
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            )

        if not trades.empty:
            longs = trades[trades["Side"] == "LONG"]
            shorts = trades[trades["Side"] == "SHORT"]

            if not longs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=longs["Confirmation"],
                        y=longs["Entry"],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=13,
                        ),
                        name="LONG Entry",
                        text=longs["Reason"],
                        hovertemplate=(
                            "%{text}<br>"
                            "Entry %{y:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

            if not shorts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=shorts["Confirmation"],
                        y=shorts["Entry"],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=13,
                        ),
                        name="SHORT Entry",
                        text=shorts["Reason"],
                        hovertemplate=(
                            "%{text}<br>"
                            "Entry %{y:.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            template="plotly_dark",
            height=720,
            xaxis_rangeslider_visible=False,
            title=(
                "XAUUSD/GC=F — 15M Setup → "
                "5M Confirmation"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )

        st.subheader("Equity Curve")

        eqfig = go.Figure()
        eqfig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Equity",
            )
        )
        eqfig.update_layout(
            template="plotly_dark",
            height=300,
            yaxis_title="Balance ($)",
        )
        st.plotly_chart(
            eqfig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    # -------------------------
    # Trades
    # -------------------------
    with tab2:
        if trades.empty:
            st.warning(
                "No completed trades. Try confirmation score 2/4 "
                "to diagnose whether the filter is too strict."
            )
        else:
            display = trades.copy()
            for col in ["Entry", "Exit", "SL", "TP", "PnL", "R"]:
                display[col] = display[col].round(2)

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download CSV",
                trades.to_csv(index=False).encode("utf-8"),
                "gold_scalping_trades.csv",
                "text/csv",
                use_container_width=True,
            )

    # -------------------------
    # Diagnostics
    # -------------------------
    with tab3:
        st.subheader("Why this setup is different")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Supertrend", "20 / 2")
        d2.metric("MACD", "12 / 26 / 9")
        d3.metric("Volume", "20-bar average")
        d4.metric("Minimum Score", f"{min_score}/4")

        st.markdown(
            """
**15-minute timeframe = setup**

- Confirmed liquidity sweep
- Price interaction with S&D zone

**5-minute timeframe = confirmation**

- Supertrend direction
- MACD direction
- Relative volume
- Confirmation candle direction

The entry is generated only after the 5-minute confirmation candle closes.

The backtester does not use future 15-minute pivot information as if it were available at the time of the trade. When a candle touches both SL and TP, the conservative assumption is that SL is hit first.
"""
        )

        if not trades.empty:
            st.subheader("Performance by Side")

            side_stats = (
                trades.groupby("Side")
                .agg(
                    Trades=("Side", "size"),
                    Wins=("Result", lambda x: (x == "Win").sum()),
                    PnL=("PnL", "sum"),
                    Avg_R=("R", "mean"),
                )
                .reset_index()
            )

            side_stats["Win Rate %"] = (
                side_stats["Wins"]
                / side_stats["Trades"]
                * 100
            )

            st.dataframe(
                side_stats.round(2),
                use_container_width=True,
                hide_index=True,
            )

st.caption(
    "Backtesting is historical analysis. Spread, slippage, commissions, "
    "broker-specific contract specifications and execution latency are not modeled."
)
