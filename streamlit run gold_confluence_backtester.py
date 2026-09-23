import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# ============================================================
# XAUUSD CONFLUENCE BACKTESTER
# Supply & Demand + Liquidity Sweep
# ============================================================

st.set_page_config(
    page_title="XAUUSD Confluence Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Helpers
# -----------------------------
def fmt_money(x):
    return f"${x:,.2f}"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# -----------------------------
# Header
# -----------------------------
st.title("XAUUSD Confluence Backtester")
st.caption("Supply & Demand • Liquidity Sweep • ATR Risk Management")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Backtest Settings")

    symbol = st.selectbox(
        "Instrument",
        ["XAUUSD=X", "GC=F"],
        index=0,
        help="XAUUSD=X is Yahoo Finance's spot-gold series. GC=F is COMEX Gold futures."
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
        help="For many MT4/MT5 gold brokers, 0.01 lot is commonly 1 oz, but verify your broker's contract specification."
    )

    st.subheader("Strategy")

    momentum_mult = st.slider(
        "S&D Expansion Threshold (ATR)",
        1.0, 4.0, 1.5, 0.1
    )

    pivot_len = st.slider(
        "Liquidity Swing Length",
        2, 10, 2
    )

    rr_ratio = st.slider(
        "Risk / Reward",
        0.5, 5.0, 2.0, 0.5
    )

    mode = st.radio(
        "Entry Mode",
        ["Strict Confluence", "Relaxed"],
        index=0,
        help="Strict = sweep must occur inside an active S&D zone. Relaxed = sweep OR zone tap can trigger a trade."
    )

    sl_atr = st.slider(
        "SL ATR Buffer",
        0.0, 1.0, 0.20, 0.05
    )

    st.subheader("Data")

    days_history = st.slider(
        "Days of 15m Data",
        5, 59, 59
    )

    show_zones = st.checkbox("Show S&D Zones", value=True)
    show_liquidity = st.checkbox("Show Liquidity", value=True)
    show_trade_levels = st.checkbox("Show Entry / SL / TP", value=True)

    run = st.button("🚀 Run Backtest", use_container_width=True)

# -----------------------------
# Data
# -----------------------------
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
        # Yahoo can return (field, ticker)
        if ticker in df.columns.get_level_values(-1):
            df.columns = df.columns.droplevel(-1)
        else:
            df.columns = df.columns.droplevel(1)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()

    # Remove duplicate timestamps.
    df = df[~df.index.duplicated(keep="last")]

    return df


# Run automatically on first load, or when button is pressed.
if "has_run" not in st.session_state:
    st.session_state.has_run = True

with st.spinner("Downloading data and running backtest..."):
    df = get_data(symbol, days_history)

if df.empty:
    st.error(
        "No data was returned. Try another instrument or reduce the requested history."
    )
    st.stop()


# ============================================================
# INDICATORS
# ============================================================

# True Range / ATR
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

# ------------------------------------------------------------
# IMPORTANT:
# Do NOT use a centered rolling pivot directly as liquidity.
# That introduces look-ahead bias because future candles are
# used to define the current pivot.
#
# A pivot at i is only known after pivot_len candles to the
# right have completed.
# ------------------------------------------------------------
raw_pivot_high = (
    df["High"]
    .rolling(window=pivot_len * 2 + 1, center=True)
    .max()
)

raw_pivot_low = (
    df["Low"]
    .rolling(window=pivot_len * 2 + 1, center=True)
    .min()
)

# Shift until the pivot becomes known in real time.
df["Confirmed_Pivot_High"] = raw_pivot_high.shift(pivot_len)
df["Confirmed_Pivot_Low"] = raw_pivot_low.shift(pivot_len)

# Keep the last confirmed liquidity level.
df["Liquidity_High"] = df["Confirmed_Pivot_High"].ffill()
df["Liquidity_Low"] = df["Confirmed_Pivot_Low"].ffill()

# Sweep logic.
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
equity_curve = []
trade_log = []

in_pos = False
pos_type = None
entry_price = sl = tp = 0.0
entry_date = None
entry_reason = ""
entry_risk = 0.0

active_demand_zones = []
active_supply_zones = []
zone_history = []

# We store the balance after each candle.
equity_by_time = {}

for i in range(1, len(df)):
    row = df.iloc[i]
    date = df.index[i]
    prev_row = df.iloc[i - 1]

    # --------------------------------------------------------
    # 1. MANAGE OPEN POSITION
    # --------------------------------------------------------
    if in_pos:
        exit_reason = None
        exit_price = None
        result = None

        if pos_type == "LONG":
            hit_sl = row["Low"] <= sl
            hit_tp = row["High"] >= tp

            # Conservative assumption:
            # If both are touched in the same candle, SL is
            # assumed to have happened first.
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

            r_multiple = (
                pnl / (entry_risk * position_size)
                if entry_risk > 0
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

    # Record equity at current candle.
    equity_by_time[date] = balance

    # --------------------------------------------------------
    # 2. ENTRY LOGIC
    # --------------------------------------------------------
    if not in_pos:
        zone_tapped_long = any(
            row["Low"] <= z["top"] and row["Low"] >= z["bottom"]
            for z in active_demand_zones
        )

        zone_tapped_short = any(
            row["High"] >= z["bottom"] and row["High"] <= z["top"]
            for z in active_supply_zones
        )

        long_signal = False
        short_signal = False
        reason = ""

        if mode == "Relaxed":
            long_signal = bool(df["Bull_Sweep"].iloc[i] or zone_tapped_long)
            short_signal = bool(df["Bear_Sweep"].iloc[i] or zone_tapped_short)

            if df["Bull_Sweep"].iloc[i] and zone_tapped_long:
                reason = "Bullish liquidity sweep + demand tap"
            elif df["Bull_Sweep"].iloc[i]:
                reason = "Bullish liquidity sweep"
            elif zone_tapped_long:
                reason = "Demand zone tap"

            if df["Bear_Sweep"].iloc[i] and zone_tapped_short:
                reason = "Bearish liquidity sweep + supply tap"
            elif df["Bear_Sweep"].iloc[i]:
                reason = "Bearish liquidity sweep"
            elif zone_tapped_short:
                reason = "Supply zone tap"

        else:
            # Strict confluence:
            # sweep MUST happen inside corresponding zone.
            long_signal = bool(
                df["Bull_Sweep"].iloc[i] and zone_tapped_long
            )
            short_signal = bool(
                df["Bear_Sweep"].iloc[i] and zone_tapped_short
            )

            if long_signal:
                reason = "Bullish sweep inside demand zone"

            if short_signal:
                reason = "Bearish sweep inside supply zone"

        # Avoid taking both directions on the same candle.
        if long_signal and not short_signal:
            entry_price = float(row["Close"])
            sl = float(row["Low"] - row["ATR"] * sl_atr)
            risk = entry_price - sl

            if risk > 0 and np.isfinite(risk):
                tp = entry_price + risk * rr_ratio
                pos_type = "LONG"
                in_pos = True
                entry_date = date
                entry_reason = reason
                entry_risk = risk

        elif short_signal and not long_signal:
            entry_price = float(row["Close"])
            sl = float(row["High"] + row["ATR"] * sl_atr)
            risk = sl - entry_price

            if risk > 0 and np.isfinite(risk):
                tp = entry_price - risk * rr_ratio
                pos_type = "SHORT"
                in_pos = True
                entry_date = date
                entry_reason = reason
                entry_risk = risk

    # --------------------------------------------------------
    # 3. REMOVE BROKEN ZONES
    # --------------------------------------------------------
    active_demand_zones = [
        z for z in active_demand_zones
        if row["Close"] > z["bottom"]
    ]

    active_supply_zones = [
        z for z in active_supply_zones
        if row["Close"] < z["top"]
    ]

    # --------------------------------------------------------
    # 4. IDENTIFY NEW SUPPLY / DEMAND ZONES
    # --------------------------------------------------------
    is_expansion = (
        np.isfinite(row["ATR"])
        and row["Body"] > row["ATR"] * momentum_mult
    )

    if is_expansion:
        zone = {
            "top": float(prev_row["High"]),
            "bottom": float(prev_row["Low"]),
            "start": date,
            "type": "DEMAND" if row["Close"] > row["Open"] else "SUPPLY",
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

equity_series = pd.Series(equity_by_time, dtype=float)

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
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")]
    df["Equity"] = equity_series.reindex(df.index).ffill().fillna(initial_capital)
else:
    df["Equity"] = initial_capital


# -----------------------------
# Metrics
# -----------------------------
total_trades = len(trade_df)

wins = int((trade_df["Result"] == "Win").sum()) if total_trades else 0
losses = int((trade_df["Result"] == "Loss").sum()) if total_trades else 0

win_rate = wins / total_trades * 100 if total_trades else 0.0
net_profit = balance - initial_capital
return_pct = net_profit / initial_capital * 100

gross_profit = (
    trade_df.loc[trade_df["PnL"] > 0, "PnL"].sum()
    if total_trades else 0.0
)

gross_loss = (
    abs(trade_df.loc[trade_df["PnL"] < 0, "PnL"].sum())
    if total_trades else 0.0
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0 else np.inf
)

avg_win = (
    trade_df.loc[trade_df["PnL"] > 0, "PnL"].mean()
    if wins else 0.0
)

avg_loss = (
    trade_df.loc[trade_df["PnL"] < 0, "PnL"].mean()
    if losses else 0.0
)

expectancy = trade_df["PnL"].mean() if total_trades else 0.0
avg_r = trade_df["R"].mean() if total_trades else 0.0

peak = df["Equity"].cummax()
drawdown = df["Equity"] - peak
drawdown_pct = (df["Equity"] / peak - 1.0) * 100

max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
max_drawdown_pct = float(drawdown_pct.min()) if len(drawdown_pct) else 0.0


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

# Main metric row
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
    "∞" if np.isinf(profit_factor) else f"{profit_factor:.2f}",
)

c5.metric(
    "Max Drawdown",
    fmt_money(max_drawdown),
    f"{max_drawdown_pct:.2f}%",
)

st.divider()

# Tabs
tab_dashboard, tab_trades, tab_analysis = st.tabs(
    ["📊 Dashboard", "📋 Trade Log", "🔍 Analysis"]
)


# ============================================================
# DASHBOARD TAB
# ============================================================

with tab_dashboard:

    # Secondary metrics
    a1, a2, a3, a4, a5, a6 = st.columns(6)

    a1.metric("Trades", total_trades)
    a2.metric("Wins", wins)
    a3.metric("Losses", losses)
    a4.metric("Avg Win", fmt_money(avg_win))
    a5.metric("Avg Loss", fmt_money(avg_loss))
    a6.metric("Expectancy", fmt_money(expectancy))

    # Equity curve
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
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title=None,
        yaxis_title="Balance ($)",
    )

    st.plotly_chart(
        equity_fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    # Price chart
    st.subheader("🕯️ Price / Liquidity / Trade Map")

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
                line=dict(width=1, dash="dot"),
                name="Liquidity High",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Liquidity_Low"],
                mode="lines",
                line=dict(width=1, dash="dot"),
                name="Liquidity Low",
            )
        )

    # Zones
    if show_zones:
        # Only display the most recent 40 zones so the chart remains readable.
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

    # Trade markers
    if show_trade_levels and not trade_df.empty:
        longs = trade_df[trade_df["Type"] == "LONG"]
        shorts = trade_df[trade_df["Type"] == "SHORT"]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["Entry Date"],
                    y=longs["Entry"],
                    mode="markers",
                    marker=dict(size=12, symbol="triangle-up"),
                    name="Long Entry",
                    text=longs["Reason"],
                    hovertemplate="%{text}<br>Entry: %{y:.2f}<extra></extra>",
                )
            )

        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts["Entry Date"],
                    y=shorts["Entry"],
                    mode="markers",
                    marker=dict(size=12, symbol="triangle-down"),
                    name="Short Entry",
                    text=shorts["Reason"],
                    hovertemplate="%{text}<br>Entry: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{symbol} — {mode}",
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
# TRADE LOG TAB
# ============================================================

with tab_trades:

    if trade_df.empty:
        st.info("No completed trades were generated.")
    else:
        st.subheader("Completed Trades")

        display_df = trade_df.copy()

        for col in ["Entry", "Exit", "Target", "StopLoss", "PnL", "R"]:
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

        csv = trade_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Trade Log CSV",
            csv,
            file_name="xauusd_trade_log.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# ANALYSIS TAB
# ============================================================

with tab_analysis:

    if trade_df.empty:
        st.info("Analysis will appear after completed trades are generated.")
    else:

        st.subheader("Trade Distribution")

        long_count = int((trade_df["Type"] == "LONG").sum())
        short_count = int((trade_df["Type"] == "SHORT").sum())

        q1, q2, q3, q4 = st.columns(4)

        q1.metric("Long Trades", long_count)
        q2.metric("Short Trades", short_count)
        q3.metric("Average R", f"{avg_r:.2f}R")
        q4.metric("Total P&L", fmt_money(net_profit))

        # P&L by trade
        pnl_fig = go.Figure()

        pnl_fig.add_trace(
            go.Bar(
                x=list(range(1, len(trade_df) + 1)),
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

        # Entry reasons
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

        # Important backtest assumptions
        st.subheader("Backtest Assumptions")

        st.markdown(
            """
            - **Timeframe:** 15 minutes.
            - **Pivot confirmation:** a swing is only used after the required right-side candles have completed, reducing look-ahead bias.
            - **Same-candle SL/TP:** if both are touched, the backtest assumes the stop was hit first.
            - **Spread/slippage:** not included.
            - **Commission:** not included.
            - **Position sizing:** P&L is calculated as price movement × position size in ounces.
            - **Open position at the end:** not counted as a completed trade.
            """
        )

st.caption(
    "Educational backtesting tool. Historical results do not establish future trading performance."
)
