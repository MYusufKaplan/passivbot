import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import threading
import time
import numpy as np
import matplotlib.dates as mdates
import json

# === CONFIGURATION ===
CSV_PATH = "/home/ubuntu/passivbot/balance.csv"
OUTPUT_PLOT = "/tmp/passivbot_report.png"
POSITIONS_CSV_PATH = "/home/ubuntu/passivbot-plotter/futures_positions.csv"


# === GLOBAL DATA STORAGE ===
df_balance = None
df_positions = None

# === TELEGRAM CONFIGURATION FROM FILE ===
def load_telegram_keys():
    with open('telegram.key', 'r') as file:
        lines = file.readlines()
        bot_token = lines[0].strip().split('=')[1]
        chat_id = lines[1].strip().split('=')[1]
    return bot_token, chat_id

BOT_TOKEN, CHAT_ID = load_telegram_keys()

# === INITIAL DATA LOADING ===
def load_csv_data():
    global df_balance
    if os.path.exists(CSV_PATH):
        df_balance = pd.read_csv(CSV_PATH)
        df_balance["timestamp"] = pd.to_datetime(df_balance["timestamp"], unit="s")
        df_balance["timestamp"] = df_balance["timestamp"] + pd.Timedelta(hours=3)
        print("✅ Balance CSV loaded into memory.")
    else:
        print("⚠️ Balance CSV file not found.")

def load_positions_data():
    global df_positions
    if os.path.exists(POSITIONS_CSV_PATH):
        df_positions = pd.read_csv(POSITIONS_CSV_PATH)
        df_positions["timestamp"] = pd.to_datetime(df_positions["timestamp"], unit="s")
        print("✅ Positions CSV loaded into memory.")
    else:
        print("⚠️ Positions CSV file not found.")

# === DOWNSAMPLING HELPER ===
def downsample(df, max_points=500):
    if len(df) > max_points:
        return df.iloc[::len(df)//max_points]
    return df

# === TELEGRAM SENDER ===
def send_telegram_image(file_path, caption="📊 Daily Balance Report"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(file_path, 'rb') as image:
        files = {'photo': image}
        data = {'chat_id': CHAT_ID, 'caption': caption}
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("✅ Sent to Telegram.")
        else:
            print(f"❌ Telegram error: {response.text}")

# === AUTO-RELOAD THREAD ===
def auto_reload_data(interval=3600):  # 3600 seconds = 60 minutes
    while True:
        print("🔄 Auto-reloading CSV data...")
        load_csv_data()
        load_positions_data()
        print("✅ Data reloaded.")
        time.sleep(interval)

# === TIME FILTER PARSER ===
def parse_time_arg(arg):
    if not arg:
        return None
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    try:
        val = int(arg[:-1])
        unit = arg[-1]
        if unit not in units:
            raise ValueError("Invalid unit")
        delta = timedelta(**{units[unit]: val})
        return datetime.utcnow() - delta
    except:
        print("⚠️ Invalid time argument. Use formats like 2h, 1d, 15m")
        sys.exit(1)

# === TEXT SUMMARY ===
def generate_summary(df, label):
    if df.empty or len(df) < 2:
        return f"📊 {label} Summary\nNo data available."

    start = df.iloc[0]
    end = df.iloc[-1]

    profit_total = end["profit"] - start["profit"]
    balance_change = end["balance"] - start["balance"]
    equity_change = end["equity"] - start["equity"]

    def fmt(val, TRY=False):
        if not TRY:
            return f"{val:+.2f} USDT"
        return f"{val:+.2f} TRY"

    def fmt_raw(val, TRY=False):
        if not TRY:
            return f"{val:.2f} USDT"
        return f"{val:.2f} TRY"

    summary = (
        f"📊 {label} Summary\n"
        f"💰 Profit: {fmt(profit_total, TRY=True)}\n"
        f"🏦 Balance Change: {fmt(balance_change)}\n"
        f"📉 Equity Change: {fmt(equity_change)}\n"
        f"\n"
        f"🔢 Latest Values\n"
        f"💰 Profit Now: {fmt_raw(end['profit'], TRY=True)}\n"
        f"🏦 Balance Now: {fmt_raw(end['balance'])}\n"
        f"📉 Equity Now: {fmt_raw(end['equity'])}"
    )
    return summary

def plot_balance_equity(df):

    df.rename(columns={df.columns[0]: 'minutes'}, inplace=True)

    df['datetime'] = df['timestamp']
    df_pos = df.copy()
    df_neg = df.copy()
    df_pos["profit"] = df["profit"].where(df["profit"] > 0, None)
    df_neg["profit"] = df["profit"].where(df["profit"] < 0, None)


    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    STARTING_BALANCE = 1000
    df['balance'] = df['balance'] / STARTING_BALANCE
    df['equity'] = df['equity'] / STARTING_BALANCE

    # Linear scale plot

    axes[0].plot(df["datetime"], df_pos["profit"], label="Profit (+)", color="green")
    axes[0].plot(df["datetime"], df_neg["profit"], label="Profit (-)", color="red")
    axes[0].set_ylabel("Profit")
    axes[0].grid()

    ax1 = axes[0].twinx()
    ax1.plot(df['datetime'], df['balance'], label='Balance', color='blue')
    ax1.plot(df['datetime'], df['equity'], label='Equity', color='green')
    ax1.set_ylabel("Value")
    ax1.set_title(f"Balance & Equity (Linear Scale)")
    ax1.legend()
    ax1.grid(True)


    axes[1].plot(df["datetime"], df_pos["profit"], label="Profit (+)", color="green")
    axes[1].plot(df["datetime"], df_neg["profit"], label="Profit (-)", color="red")
    axes[1].set_ylabel("Profit")
    axes[1].grid()

    ax2 = axes[1].twinx()
    # Log scale plot
    ax2.plot(df['datetime'], df['balance'], label='Balance', color='blue')
    ax2.plot(df['datetime'], df['equity'], label='Equity', color='green')
    ax2.set_yscale("log")
    ax2.set_ylabel("Value (Log Scale)")
    ax2.set_title("Balance & Equity (Logarithmic Scale)")

    # Draw trend line for balance (straight line from (x0, y0) to (xn, yn))
    x_vals = df['datetime'].values
    balance_vals = df['balance'].values
    equity_vals = df['equity'].values
    neg_profit_vals = df_neg['profit'].values
    pos_profit_vals = df_pos['profit'].values

    # Trend line for log values
    log_balance = np.log(balance_vals)
    log_equity = np.log(equity_vals)
    log_neg_profits = np.log(neg_profit_vals)
    log_pos_profits = np.log(pos_profit_vals)

    # Generate straight line between first and last log values
    def get_trend_line(y_values, x_values):
        y0, yn = y_values[0], y_values[-1]
        x0, xn = x_values[0], x_values[-1]
        slope = (yn - y0) / (xn - x0)
        return y0 + slope * (x_values - x0)

    balance_trend = get_trend_line(log_balance, x_vals)
    equity_trend = get_trend_line(log_equity, x_vals)
    pos_profit_trend = get_trend_line(log_pos_profits, x_vals)
    neg_profit_trend = get_trend_line(log_neg_profits, x_vals)

    # Plot log-based trend lines on the linear scale chart
    ax1.plot(df['datetime'], np.exp(pos_profit_trend), color='purple', linestyle='--', linewidth=1, label='Profit Log Trend')
    ax1.plot(df['datetime'], np.exp(neg_profit_trend), color='purple', linestyle='--', linewidth=1, label='Profit Log Trend')
    axes[0].plot(df['datetime'], np.exp(equity_trend), color='red', linestyle='--', linewidth=1, label='Log Trend')

    # Plot trend line for balance on log scale (exponentiate back)
    ax2.plot(df['datetime'], np.exp(equity_trend), color='red', linestyle='--', linewidth=1, label='Trend Line')
    axes[1].plot(df['datetime'], np.exp(equity_trend), color='red', linestyle='--', linewidth=1, label='Trend Line')

    axes[1].legend()
    axes[1].grid(True, which="both", ls="--")

    # Date formatting
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xlabel("Date")

    plt.tight_layout()
    return plt

def get_closed_positions_summary(json_path="closed_positions.json"):
    if not os.path.exists(json_path):
        return "⚠️ No closed positions data available."

    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        return "📭 No closed positions found."

    message = "🔒 *Closed Positions Summary*\n"
    winners = []
    losers = []
    total_pnl = 0

    for symbol, pnl in sorted(data.items(), key=lambda x: x[1], reverse=True):
        emoji = "🟢" if pnl > 0 else "🔴"
        line = f"{emoji} `{symbol}`: {pnl:+.2f} USDT"
        (winners if pnl > 0 else losers).append(line)
        total_pnl += pnl

    if winners:
        message += "\n💰 *Profitable Trades:*\n" + "\n".join(winners)
    if losers:
        message += "\n\n💸 *Losing Trades:*\n" + "\n".join(losers)

    net_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
    message += f"\n\n📊 *Total Realized PnL:* {net_emoji} `{total_pnl:+.2f} USDT`"

    return message

async def closed_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    summary = get_closed_positions_summary()
    await update.message.reply_text(summary)

# === COMMAND HANDLERS ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Use the following commands:\n"
                                    "/report - Get the latest report\n"
                                    "/summary [time] - Get a summary of the data (e.g., /summary 1d)\n"
                                    "/alltime - Get data from the start until now")

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if df_balance is None or df_balance.empty:
        await update.message.reply_text("❌ No balance data loaded!")
        return

    start_time = datetime.utcnow() - timedelta(hours=24)
    df = df_balance[df_balance["timestamp"] >= start_time]

    if df.empty:
        await update.message.reply_text("⚠️ No data available for the last 24 hours.")
        return

    label = "Last 24 Hours"
    df = downsample(df)

    df_pos = df.copy()
    df_neg = df.copy()
    df_pos["profit"] = df["profit"].where(df["profit"] > 0, None)
    df_neg["profit"] = df["profit"].where(df["profit"] < 0, None)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["timestamp"], df_pos["profit"], label="Profit (+)", color="green")
    ax1.plot(df["timestamp"], df_neg["profit"], label="Profit (-)", color="red")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Profit")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["balance"], label="Balance", color="blue")
    ax2.plot(df["timestamp"], df["equity"], label="Equity", color="purple", linestyle="--")
    ax2.set_ylabel("Balance / Equity")
    
    fig.suptitle("📈 Profit, Balance & Equity Report (Last 24 Hours)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    fig.tight_layout()

    plt.savefig(OUTPUT_PLOT)
    plt.close()

    caption = generate_summary(df, label)
    send_telegram_image(OUTPUT_PLOT, caption)

async def all_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if df_balance is None or df_balance.empty:
        await update.message.reply_text("❌ No balance data loaded!")
        return

    label = "All Time"
    df = downsample(df_balance)

    df_pos = df.copy()
    df_neg = df.copy()
    df_pos["profit"] = df["profit"].where(df["profit"] > 0, None)
    df_neg["profit"] = df["profit"].where(df["profit"] < 0, None)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["timestamp"], df_pos["profit"], label="Profit (+)", color="green")
    ax1.plot(df["timestamp"], df_neg["profit"], label="Profit (-)", color="red")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Profit")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["balance"], label="Balance", color="blue")
    ax2.plot(df["timestamp"], df["equity"], label="Equity", color="purple", linestyle="--")
    ax2.set_ylabel("Balance / Equity")

    fig.suptitle("📈 Profit, Balance & Equity Report (All Time)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    fig.tight_layout()

    plot_path = "/tmp/passivbot_all_time_report.png"
    plt.savefig(plot_path)
    plt.close()

    caption = generate_summary(df, label)
    send_telegram_image(plot_path, caption)

async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    time_arg = context.args[0] if len(context.args) > 0 else None

    if df_balance is None or df_balance.empty:
        await update.message.reply_text("❌ No balance data loaded!")
        return

    start_time = parse_time_arg(time_arg)
    label = f"Last {time_arg}" if time_arg else "All Time"

    df = df_balance.copy()
    if start_time:
        df = df[df["timestamp"] >= start_time]

    summary_text = generate_summary(df, label)
    await update.message.reply_text(summary_text)



async def refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    load_csv_data()
    load_positions_data()
    await update.message.reply_text("🔄 Data refreshed manually!")


# === MAIN BOT SETUP ===
def main():
    # Load CSVs once at startup
    load_csv_data()
    load_positions_data()

    # Start auto-reload in background thread
    threading.Thread(target=auto_reload_data, daemon=True).start()

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("report", report))
    application.add_handler(CommandHandler("alltime", all_time))
    application.add_handler(CommandHandler("summary", summary))
    application.add_handler(CommandHandler("positions", closed_positions))
    application.add_handler(CommandHandler("refresh", refresh))
    application.run_polling()

if __name__ == '__main__':
    main()
