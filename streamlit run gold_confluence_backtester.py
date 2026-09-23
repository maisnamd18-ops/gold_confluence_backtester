import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# =====================================================================
# UI SETUP
# =====================================================================
st.set_page_config(page_title="Master Confluence Backtester", layout="wide")
st.title("XAUT/USD Master Strategy: S&D + Liquidity Sweep")

st.sidebar.header("Data")
symbol = st.sidebar.selectbox(
    "Symbol",
    ["GC=F", "XAUUSD=X"],
    help="GC=F is COMEX gold futures. XAUUSD=X is the spot-style quote, "
         "closer to what TradingView/Binance show for XAUUSD — pick whichever "
         "matches what you're paper trading.",
)
timeframe = st.sidebar.selectbox(
    "Timeframe", ["5m", "15m"],
    help="Yahoo Finance serves up to 60 days of history for both 5m and 15m bars.",
)
days_history = st.sidebar.slider("Days of data", 5, 59, 59)

st.sidebar.header("Confluence Parameters")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000.0, step=100.0)
sizing_mode = st.sidebar.radio("Position sizing", ["Fixed lot size", "Risk % of equity"])
if sizing_mode == "Fixed lot size":
    lot_size = st.sidebar.number_input("Lot Size (oz)", value=1.0, min_value=0.01)
    risk_pct = None
else:
    risk_pct = st.sidebar.number_input("Risk per trade (%)", value=0.5, min_value=0.05, step=0.05)
    lot_size = None

momentum_mult = st.sidebar.slider("S&D Momentum (x ATR)", 1.0, 4.0, 1.5, 0.1)
pivot_len = st.sidebar.number_input("Liquidity Swing Length", min_value=2, value=2)
rr_ratio = st.sidebar.slider("Risk / Reward", 1.0, 5.0, 2.0, 0.5)
relax_filter = st.sidebar.checkbox("Relax confluence (take sweeps OR zone taps, not just both)", value=True)
max_trades_day = st.sidebar.number_input("Max trades per day (0 = no limit)", min_value=0, value=0)

# =====================================================================
# DATA FETCHING
# =====================================================================
@st.cache_data(ttl=900)
def get_data(sym: str, days: int, interval: str) -> pd.DataFrame:
    df = yf.download(sym, period=f"{days}d", interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

try:
    with st.spinner("Fetching data and calculating confluence zones..."):
        df = get_data(symbol, days_history, timeframe)
except Exception as e:
    st.error(f"Failed to fetch data for {symbol}: {e}")
    st.stop()

if df.empty:
    st.error(f"No data returned for {symbol}. Try a different symbol or day range.")
    st.stop()

# =====================================================================
# INDICATORS
# =====================================================================
high_low = df["High"] - df["Low"]
high_close = (df["High"] - df["Close"].shift()).abs()
low_close = (df["Low"] - df["Close"].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
df["Body"] = (df["Close"] - df["Open"]).abs()

df["Pivot_High"] = df["High"].rolling(window=pivot_len * 2 + 1, center=True).max()
df["Pivot_Low"] = df["Low"].rolling(window=pivot_len * 2 + 1, center=True).min()
df["Liquidity_High"] = df["Pivot_High"].ffill()
df["Liquidity_Low"] = df["Pivot_Low"].ffill()

df["Bull_Sweep"] = (df["Low"] < df["Liquidity_Low"]) & (df["Close"] > df["Liquidity_Low"])
df["Bear_Sweep"] = (df["High"] > df["Liquidity_High"]) & (df["Close"] < df["Liquidity_High"])

# =====================================================================
# BACKTEST ENGINE
# =====================================================================
balance = initial_capital
equity_curve = [initial_capital]
trade_log = []

in_pos = False
pos_type = None
entry_price = sl = tp = qty = 0.0

active_demand_zones = []   # each: {'top','bottom','start'}
active_supply_zones = []
zone_shapes = []            # finalized rectangles for plotting


def position_size(risk_per_unit: float) -> float:
    """Fixed lot, or equity-risk-based sizing."""
    if sizing_mode == "Fixed lot size":
        return lot_size
    if risk_per_unit <= 0:
        return 0.0
    risk_money = balance * risk_pct / 100.0
    return risk_money / risk_per_unit


def close_position(exit_price: float, date, reason: str):
    """Close the open position, update balance, log the trade."""
    global balance, in_pos
    if pos_type == "LONG":
        pnl = (exit_price - entry_price) * qty
    else:
        pnl = (entry_price - exit_price) * qty
    balance += pnl
    trade_log.append({
        "Date": date, "Type": pos_type, "Entry": entry_price,
        "Exit": exit_price, "Target": tp, "StopLoss": sl,
        "Reason": reason, "Result": "Win" if pnl > 0 else "Loss",
        "PnL": pnl, "R": pnl / (abs(entry_price - sl) * qty) if qty else 0,
    })
    in_pos = False


trades_today = 0
cur_day = df.index[0].date()

for i in range(1, len(df)):
    row = df.iloc[i]
    date = df.index[i]
    prev_row = df.iloc[i - 1]

    if date.date() != cur_day:
        cur_day = date.date()
        trades_today = 0

    # 1. Manage exits (stop checked first if both levels are hit in-bar)
    if in_pos:
        if pos_type == "LONG":
            if row["Low"] <= sl:
                close_position(sl, date, "stop")
            elif row["High"] >= tp:
                close_position(tp, date, "target")
        elif pos_type == "SHORT":
            if row["High"] >= sl:
                close_position(sl, date, "stop")
            elif row["Low"] <= tp:
                close_position(tp, date, "target")

    equity_curve.append(balance)

    can_trade = (not in_pos) and (max_trades_day == 0 or trades_today < max_trades_day)

    # 2. Entries
    if can_trade:
        if relax_filter:
            zone_tapped_long = next(
                (z for z in active_demand_zones if z["bottom"] <= row["Low"] <= z["top"]), None)
            zone_tapped_short = next(
                (z for z in active_supply_zones if z["bottom"] <= row["High"] <= z["top"]), None)

            if row["Bull_Sweep"] or zone_tapped_long:
                entry_price = row["Close"]
                sl = (zone_tapped_long["bottom"] if zone_tapped_long else row["Low"]) - row["ATR"] * 0.5
                risk = entry_price - sl
                if risk > 0:
                    qty = position_size(risk)
                    if qty > 0:
                        tp = entry_price + risk * rr_ratio
                        pos_type = "LONG"
                        in_pos = True
                        trades_today += 1
            elif row["Bear_Sweep"] or zone_tapped_short:
                entry_price = row["Close"]
                sl = (zone_tapped_short["top"] if zone_tapped_short else row["High"]) + row["ATR"] * 0.5
                risk = sl - entry_price
                if risk > 0:
                    qty = position_size(risk)
                    if qty > 0:
                        tp = entry_price - risk * rr_ratio
                        pos_type = "SHORT"
                        in_pos = True
                        trades_today += 1
        else:
            # Strict confluence: sweep must land inside an active zone
            if row["Bull_Sweep"]:
                for z in active_demand_zones[:]:
                    if z["bottom"] <= row["Low"] <= z["top"]:
                        entry_price = row["Close"]
                        sl = min(row["Low"], z["bottom"]) - row["ATR"] * 0.2
                        risk = entry_price - sl
                        if risk > 0:
                            qty = position_size(risk)
                            if qty > 0:
                                tp = entry_price + risk * rr_ratio
                                pos_type = "LONG"
                                in_pos = True
                                trades_today += 1
                                active_demand_zones.remove(z)
                                break
            if not in_pos and row["Bear_Sweep"]:
                for z in active_supply_zones[:]:
                    if z["bottom"] <= row["High"] <= z["top"]:
                        entry_price = row["Close"]
                        sl = max(row["High"], z["top"]) + row["ATR"] * 0.2
                        risk = sl - entry_price
                        if risk > 0:
                            qty = position_size(risk)
                            if qty > 0:
                                tp = entry_price - risk * rr_ratio
                                pos_type = "SHORT"
                                in_pos = True
                                trades_today += 1
                                active_supply_zones.remove(z)
                                break

    # 3. Invalidate broken zones (finalize their rectangle end date for plotting)
    still_demand = []
    for z in active_demand_zones:
        if row["Close"] > z["bottom"]:
            still_demand.append(z)
        else:
            zone_shapes.append(dict(type="rect", x0=z["start"], y0=z["bottom"], x1=date,
                                     y1=z["top"], fillcolor="rgba(0,255,0,0.12)", line=dict(width=0)))
    active_demand_zones = still_demand

    still_supply = []
    for z in active_supply_zones:
        if row["Close"] < z["top"]:
            still_supply.append(z)
        else:
            zone_shapes.append(dict(type="rect", x0=z["start"], y0=z["bottom"], x1=date,
                                     y1=z["top"], fillcolor="rgba(255,0,0,0.12)", line=dict(width=0)))
    active_supply_zones = still_supply

    # 4. Identify new S&D zones from displacement candles
    is_expansion = row["Body"] > (row["ATR"] * momentum_mult)
    if is_expansion:
        if row["Close"] > row["Open"]:
            active_demand_zones.append({"top": prev_row["High"], "bottom": prev_row["Low"], "start": date})
        elif row["Close"] < row["Open"]:
            active_supply_zones.append({"top": prev_row["High"], "bottom": prev_row["Low"], "start": date})

# finalize any zones still open at the end of the data
for z in active_demand_zones:
    zone_shapes.append(dict(type="rect", x0=z["start"], y0=z["bottom"], x1=df.index[-1],
                             y1=z["top"], fillcolor="rgba(0,255,0,0.12)", line=dict(width=0)))
for z in active_supply_zones:
    zone_shapes.append(dict(type="rect", x0=z["start"], y0=z["bottom"], x1=df.index[-1],
                             y1=z["top"], fillcolor="rgba(255,0,0,0.12)", line=dict(width=0)))

df["Equity"] = equity_curve
trade_df = pd.DataFrame(trade_log)

# =====================================================================
# METRICS
# =====================================================================
total_trades = len(trade_df)
if total_trades > 0:
    wins = trade_df[trade_df["Result"] == "Win"]
    losses = trade_df[trade_df["Result"] == "Loss"]
    win_rate = len(wins) / total_trades * 100
    gross_profit = wins["PnL"].sum()
    gross_loss = abs(losses["PnL"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    avg_r = trade_df["R"].mean()
    running_max = df["Equity"].cummax()
    max_dd_pct = ((df["Equity"] - running_max) / running_max).min() * 100
else:
    win_rate = profit_factor = avg_r = max_dd_pct = 0.0

net_profit = balance - initial_capital

# =====================================================================
# UI DISPLAY
# =====================================================================
tab1, tab2 = st.tabs(["Dashboard & Chart", "Trade Log"])

with tab1:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Final Balance", f"${balance:,.2f}", f"{net_profit:,.2f}")
    c2.metric("Total Trades", total_trades)
    c3.metric("Win Rate", f"{win_rate:.1f}%")
    c4.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")
    c5.metric("Avg R / Trade", f"{avg_r:.2f}R")
    c6.metric("Max Drawdown", f"{max_dd_pct:.1f}%")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price")])
    fig.add_trace(go.Scatter(x=df.index, y=df["Liquidity_High"], mode="lines",
                              line=dict(color="red", width=1, dash="dot"), name="Liq High"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Liquidity_Low"], mode="lines",
                              line=dict(color="green", width=1, dash="dot"), name="Liq Low"))

    if not trade_df.empty:
        longs = trade_df[trade_df["Type"] == "LONG"]
        shorts = trade_df[trade_df["Type"] == "SHORT"]
        fig.add_trace(go.Scatter(x=longs["Date"], y=longs["Entry"], mode="markers",
                                  marker=dict(color="blue", size=11, symbol="triangle-up"), name="Long Entry"))
        fig.add_trace(go.Scatter(x=longs["Date"], y=longs["Exit"], mode="markers",
                                  marker=dict(color="green", size=7, symbol="circle"), name="Long Exit"))
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Entry"], mode="markers",
                                  marker=dict(color="magenta", size=11, symbol="triangle-down"), name="Short Entry"))
        fig.add_trace(go.Scatter(x=shorts["Date"], y=shorts["Exit"], mode="markers",
                                  marker=dict(color="green", size=7, symbol="circle"), name="Short Exit"))

    fig.update_layout(height=700, template="plotly_dark", title="Master Confluence Strategy",
                       xaxis_rangeslider_visible=False, shapes=zone_shapes[-40:])
    st.plotly_chart(fig, use_container_width=True)

    st.line_chart(df.set_index(df.index)["Equity"])

with tab2:
    if not trade_df.empty:
        st.dataframe(
            trade_df[["Date", "Type", "Entry", "Exit", "Target", "StopLoss", "Reason", "Result", "PnL", "R"]]
            .style.map(lambda x: "color: #5fa876" if x == "Win" else ("color: #c1594a" if x == "Loss" else ""),
                       subset=["Result"])
        )
    else:
        st.info("No trades executed with the current settings.")
