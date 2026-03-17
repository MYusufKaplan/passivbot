#!/usr/bin/env python3

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from datetime import timedelta, datetime
import numpy as np
import os
import sys
import json
import webbrowser


def calculate_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def colorful_log(msg, emoji="✨"):
    print(f"{emoji} {msg}")


def get_dates_from_config(csv_file_path):
    """Extract start date, end date, and starting balance from config.json."""
    try:
        csv_dir = os.path.dirname(csv_file_path)
        config_path = os.path.join(csv_dir, "config.json")

        if not os.path.isfile(config_path):
            colorful_log(f"⚠️  Config file not found: {config_path}", emoji="🚨")
            return None, None, None

        colorful_log(f"📖 Reading config from: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        backtest = config.get('backtest', {})
        start_date_str = backtest.get('start_date')
        end_date_str = backtest.get('end_date')
        starting_balance = backtest.get('starting_balance', 10000)

        colorful_log(f"📅 Config start_date: {start_date_str}")
        colorful_log(f"📅 Config end_date: {end_date_str}")
        colorful_log(f"💰 Config starting_balance: {starting_balance}")

        if not start_date_str:
            colorful_log("⚠️  start_date not found in config.json", emoji="🚨")
            return None, None, None

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        return start_date, end_date_str, starting_balance

    except Exception as e:
        colorful_log(f"❌ Error reading config file: {e}", emoji="🚨")
        return None, None, None


def calculate_stats(df, start_time):
    """Calculate comprehensive trading statistics."""
    equity = df['equity'].values
    balance = df['balance'].values
    
    # Returns
    total_return = (equity[-1] / equity[0] - 1) * 100
    
    # Drawdown calculation
    rolling_max = np.maximum.accumulate(equity)
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_drawdown = np.min(drawdown)
    
    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0] + 1
    drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0] + 1
    
    # Daily returns for calculations
    df_daily = df.set_index('datetime').resample('D').last().dropna()
    if len(df_daily) > 1:
        daily_returns = df_daily['equity'].pct_change().dropna()
        daily_returns_pct = daily_returns * 100
        
        # Basic stats
        sharpe_ratio = np.sqrt(365) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        sortino_denom = daily_returns[daily_returns < 0].std()
        sortino_ratio = np.sqrt(365) * daily_returns.mean() / sortino_denom if sortino_denom > 0 else 0
        
        # Win rate (positive days)
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
        
        # Best/worst days
        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100
        
        # Volatility
        annual_volatility = daily_returns.std() * np.sqrt(365) * 100
        
        # Omega Ratio (threshold = 0)
        gains = daily_returns[daily_returns > 0].sum()
        losses = abs(daily_returns[daily_returns < 0].sum())
        omega_ratio = gains / losses if losses > 0 else 0
        
        # Profit Factor
        gross_profit = daily_returns_pct[daily_returns_pct > 0].sum()
        gross_loss = abs(daily_returns_pct[daily_returns_pct < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Payoff Ratio
        avg_win = daily_returns_pct[daily_returns_pct > 0].mean() if (daily_returns_pct > 0).any() else 0
        avg_loss = abs(daily_returns_pct[daily_returns_pct < 0].mean()) if (daily_returns_pct < 0).any() else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Expectancy
        win_rate_decimal = win_rate / 100
        expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)
        
        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(daily_returns_pct, 5)
        cvar_95 = daily_returns_pct[daily_returns_pct <= var_95].mean()
        
        # Tail Ratio
        tail_gain = np.percentile(daily_returns_pct, 95)
        tail_loss = abs(np.percentile(daily_returns_pct, 5))
        tail_ratio = tail_gain / tail_loss if tail_loss > 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        
        # Consecutive wins/losses
        is_win = (daily_returns > 0).astype(int)
        win_streaks = []
        loss_streaks = []
        current_streak = 0
        current_type = None
        
        for win in is_win:
            if win == 1:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_type == 'loss' and current_streak > 0:
                        loss_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'win'
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_type == 'win' and current_streak > 0:
                        win_streaks.append(current_streak)
                    current_streak = 1
                    current_type = 'loss'
        
        # Add final streak
        if current_type == 'win' and current_streak > 0:
            win_streaks.append(current_streak)
        elif current_type == 'loss' and current_streak > 0:
            loss_streaks.append(current_streak)
        
        max_consecutive_wins = max(win_streaks) if win_streaks else 0
        max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
        
    else:
        sharpe_ratio = sortino_ratio = win_rate = best_day = worst_day = annual_volatility = 0
        omega_ratio = profit_factor = payoff_ratio = expectancy = 0
        var_95 = cvar_95 = tail_ratio = ulcer_index = 0
        max_consecutive_wins = max_consecutive_losses = 0
        avg_win = avg_loss = 0
    
    # Monthly returns
    df_monthly = df.set_index('datetime').resample('ME').last().dropna()
    if len(df_monthly) > 1:
        monthly_returns = df_monthly['equity'].pct_change().dropna()
        monthly_win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100
    else:
        monthly_win_rate = 0
    
    # Duration
    duration_days = (df['datetime'].iloc[-1] - start_time).days
    
    # CAGR
    years = duration_days / 365.25
    cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Calmar ratio (CAGR / abs(Max Drawdown))
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Recovery Factor (Net Profit / Max Drawdown)
    net_profit = total_return
    recovery_factor = net_profit / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Kelly Criterion
    kelly = (win_rate / 100) - ((1 - win_rate / 100) / payoff_ratio) if payoff_ratio > 0 else 0
    
    # Recovery analysis
    recovery_times = []
    for i, start_idx in enumerate(drawdown_starts):
        if i < len(drawdown_ends):
            end_idx = drawdown_ends[i]
            recovery_days = (df['datetime'].iloc[end_idx] - df['datetime'].iloc[start_idx]).days
            recovery_times.append(recovery_days)
    avg_recovery = np.mean(recovery_times) if recovery_times else 0
    max_recovery = np.max(recovery_times) if recovery_times else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'omega_ratio': omega_ratio,
        'win_rate': win_rate,
        'monthly_win_rate': monthly_win_rate,
        'best_day': best_day,
        'worst_day': worst_day,
        'annual_volatility': annual_volatility,
        'profit_factor': profit_factor,
        'payoff_ratio': payoff_ratio,
        'expectancy': expectancy,
        'recovery_factor': recovery_factor,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'tail_ratio': tail_ratio,
        'ulcer_index': ulcer_index,
        'kelly_criterion': kelly,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'duration_days': duration_days,
        'avg_recovery_days': avg_recovery,
        'max_recovery_days': max_recovery,
        'drawdown_series': drawdown,
        'rolling_max': rolling_max
    }


def calculate_weekly_returns(df):
    """Calculate weekly returns for heatmap."""
    df_weekly = df.set_index('datetime').resample('W').last().dropna()
    if len(df_weekly) > 1:
        weekly_returns = df_weekly['equity'].pct_change().dropna() * 100
        return weekly_returns
    return pd.Series()


def find_milestone_crossings(df, start_time):
    """Find when equity crosses 10^x milestones (2x, 5x, 10x, 20x, 50x, 100x, etc.)"""
    colorful_log("🎯 Analyzing milestone crossings...")

    milestones = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    crossings = []

    for milestone in milestones:
        crossing_rows = df[df['equity'] >= milestone]

        if not crossing_rows.empty:
            first_crossing = crossing_rows.iloc[0]
            crossing_date = first_crossing['datetime']
            crossing_value = first_crossing['equity']

            if hasattr(crossing_date, 'to_pydatetime'):
                crossing_date = crossing_date.to_pydatetime()

            days_from_start = (crossing_date - start_time).days

            crossings.append({
                'milestone': milestone,
                'date': crossing_date,
                'value': crossing_value,
                'days': days_from_start
            })

    if crossings:
        colorful_log("🚀 Milestone Crossings:")
        for crossing in crossings:
            milestone_str = f"{crossing['milestone']}x"
            colorful_log(f"   {milestone_str:>6} reached on {crossing['date'].strftime('%Y-%m-%d')} "
                         f"(day {crossing['days']:>4}) at {crossing['value']:.2f}x", emoji="📈")
    else:
        colorful_log("📊 No major milestones crossed in this timeframe")

    return crossings


def plot_balance_equity(csv_file, run_number, start_time, starting_balance, output_file=None, save_only=False):
    colorful_log("📖 Reading CSV file...")
    df = pd.read_csv(csv_file)

    df.rename(columns={df.columns[0]: 'minutes'}, inplace=True)
    df['datetime'] = df['minutes'].apply(lambda x: start_time + timedelta(minutes=x))

    end_time = df['datetime'].iloc[-1]
    colorful_log(f"📅 Start date: {start_time.strftime('%Y-%m-%d')}")
    colorful_log(f"📅 End date: {end_time.strftime('%Y-%m-%d')}")
    colorful_log(f"⏱️  Duration: {(end_time - start_time).days} days")

    df['balance'] = df['balance'] / starting_balance
    df['equity'] = df['equity'] / starting_balance

    crossings = find_milestone_crossings(df, start_time)

    colorful_log("🖌️ Preparing the plots...")

    # Calculate trend line
    x_vals = df['minutes'].values
    log_equity = np.log(df['equity'].values)

    def get_trend_line(y_values, x_values):
        y0, yn = y_values[0], y_values[-1]
        x0, xn = x_values[0], x_values[-1]
        slope = (yn - y0) / (xn - x0)
        return y0 + slope * (x_values - x0)

    equity_trend = np.exp(get_trend_line(log_equity, x_vals))

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{run_number} Balance & Equity (Linear Scale)",
            "Balance & Equity (Logarithmic Scale)"
        )
    )

    # Linear scale plot (top)
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['balance'], name='Balance',
                   line=dict(color='#3366cc', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['equity'], name='Equity',
                   line=dict(color='#109618', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=equity_trend, name='Log Trend',
                   line=dict(color='#dc3912', width=1.5, dash='dash')),
        row=1, col=1
    )

    # Log scale plot (bottom)
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['balance'], name='Balance',
                   line=dict(color='#3366cc', width=1.5), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['equity'], name='Equity',
                   line=dict(color='#109618', width=1.5), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=equity_trend, name='Trend Line',
                   line=dict(color='#dc3912', width=1.5, dash='dash'), showlegend=False),
        row=2, col=1
    )

    # Add milestone annotations on log scale
    for crossing in crossings:
        fig.add_annotation(
            x=crossing['date'],
            y=crossing['value'],
            text=f"{crossing['milestone']}x",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            ax=20,
            ay=-30,
            font=dict(size=10, color='#ff7f0e'),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig.update_yaxes(title_text="Value (x)", row=1, col=1)
    
    # Calculate proper log scale range
    min_val = max(df[['balance', 'equity']].min().min(), 0.1)  # Avoid log(0)
    max_val = df[['balance', 'equity']].max().max()
    log_min = np.floor(np.log10(min_val))
    log_max = np.ceil(np.log10(max_val))
    
    fig.update_yaxes(
        title_text="Value (x)", 
        type="log", 
        range=[log_min, log_max],
        row=2, col=1
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Output handling
    graphs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    if not output_file:
        if run_number != "N/A":
            output_file = os.path.join(graphs_dir, f"balance_equity_plot_{run_number}.html")
        else:
            output_file = os.path.join(graphs_dir, "balance_equity_plot.html")
    elif not os.path.isabs(output_file):
        output_file = os.path.join(graphs_dir, output_file)

    colorful_log(f"💾 Saving plot to {output_file}")
    fig.write_html(output_file)
    colorful_log(f"✅ Plot saved as {output_file}")

    # Generate detailed version
    detailed_output = output_file.replace('.html', '_detailed.html')
    plot_detailed(df, run_number, start_time, crossings, equity_trend, detailed_output)

    # if not save_only:
    #     colorful_log("🌐 Opening in browser...")
    #     webbrowser.open(f"file://{os.path.abspath(output_file)}")


def plot_detailed(df, run_number, start_time, crossings, equity_trend, output_file):
    """Generate detailed analysis HTML with all advanced features."""
    colorful_log("📊 Generating detailed analysis...")
    
    stats = calculate_stats(df, start_time)
    weekly_returns = calculate_weekly_returns(df)
    
    # Load fills data for positions chart
    # Try to construct fills path from run number
    fills_file = None
    if run_number != "N/A":
        base_path = os.getcwd()
        fills_file = os.path.join(base_path, f"backtests/optimizer/live/{run_number}/fills.csv")
        if not os.path.exists(fills_file):
            fills_file = os.path.join(base_path, f"backtests/optimizer/extremes/{run_number}/fills.csv")
        if not os.path.exists(fills_file):
            fills_file = None
    
    positions_data = None
    pnl_by_coin = None
    if fills_file and os.path.exists(fills_file):
        try:
            colorful_log(f"📂 Loading fills from: {fills_file}")
            fills_df = pd.read_csv(fills_file)
            fills_df['datetime'] = fills_df['minute'].apply(lambda x: start_time + timedelta(minutes=x))
            
            # Count entries and closes per day
            fills_df['date'] = fills_df['datetime'].dt.date
            fills_df['is_entry'] = fills_df['type'].str.contains('entry')
            fills_df['is_close'] = fills_df['type'].str.contains('close')
            
            daily_positions = fills_df.groupby('date').agg({
                'is_entry': 'sum',
                'is_close': 'sum'
            }).reset_index()
            daily_positions['date'] = pd.to_datetime(daily_positions['date'])
            positions_data = daily_positions
            
            # Calculate PNL by coin
            pnl_by_coin = fills_df.groupby('coin')['pnl'].sum().sort_values(ascending=False)
            
            colorful_log("📈 Loaded positions data from fills.csv")
        except Exception as e:
            colorful_log(f"⚠️  Could not load fills data: {e}")
    
    # Create figure with subplots
    fig = make_subplots(
        rows=6, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "domain"}, {"type": "xy"}],
            [{"colspan": 2, "type": "table"}, None]
        ],
        subplot_titles=(
            f"{run_number} Balance & Equity (Log Scale)",
            "Drawdown Analysis",
            "Positions Opened/Closed",
            "Daily Returns Distribution",
            "", "Weekly Returns",
            "PNL by Coin"
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.1,
        row_heights=[0.25, 0.15, 0.12, 0.12, 0.15, 0.21]
    )
    
    # Row 1: Main equity chart with log scale
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['balance'], name='Balance',
                   line=dict(color='#3366cc', width=1.5),
                   hovertemplate='%{x}<br>Balance: %{y:.2f}x<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['equity'], name='Equity',
                   line=dict(color='#109618', width=1.5),
                   hovertemplate='%{x}<br>Equity: %{y:.2f}x<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=equity_trend, name='Log Trend',
                   line=dict(color='#dc3912', width=1.5, dash='dash'),
                   hovertemplate='%{x}<br>Trend: %{y:.2f}x<extra></extra>'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=stats['rolling_max'], name='ATH',
                   line=dict(color='#ff7f0e', width=1, dash='dot'),
                   hovertemplate='%{x}<br>ATH: %{y:.2f}x<extra></extra>'),
        row=1, col=1
    )
    
    # Add milestone annotations
    for crossing in crossings:
        fig.add_annotation(
            x=crossing['date'], y=crossing['value'],
            text=f"{crossing['milestone']}x",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            ax=20, ay=-30, font=dict(size=10, color='#ff7f0e'),
            row=1, col=1
        )
    
    # Row 2: Drawdown chart
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=stats['drawdown_series'],
                   fill='tozeroy', name='Drawdown',
                   line=dict(color='#dc3912', width=1),
                   fillcolor='rgba(220, 57, 18, 0.3)',
                   hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'),
        row=2, col=1
    )
    
    # Add max drawdown line
    fig.add_hline(y=stats['max_drawdown'], line_dash="dash", line_color="red",
                  annotation_text=f"Max DD: {stats['max_drawdown']:.1f}%",
                  annotation_position="bottom right", row=2, col=1)
    
    # Row 3: Positions opened/closed stacked bar chart
    if positions_data is not None:
        fig.add_trace(
            go.Bar(x=positions_data['date'], y=positions_data['is_entry'], 
                   name='Opened', marker_color='#109618',
                   hovertemplate='%{x}<br>Opened: %{y}<extra></extra>'),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=positions_data['date'], y=positions_data['is_close'], 
                   name='Closed', marker_color='#dc3912',
                   hovertemplate='%{x}<br>Closed: %{y}<extra></extra>'),
            row=3, col=1
        )
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_layout(barmode='stack')
    
    # Row 4: Daily returns histogram with date labels for small bins
    df_daily = df.set_index('datetime').resample('D').last().dropna()
    if len(df_daily) > 1:
        daily_returns = df_daily['equity'].pct_change().dropna() * 100
        daily_dates = daily_returns.index
        
        # Manual binning to track dates per bin
        n_bins = 50
        min_ret, max_ret = daily_returns.min(), daily_returns.max()
        bin_width = (max_ret - min_ret) / n_bins if max_ret != min_ret else 1
        
        bins = {}
        bin_dates = {}
        
        for date, ret in zip(daily_dates, daily_returns):
            bin_idx = int((ret - min_ret) / bin_width) if bin_width > 0 else 0
            bin_idx = min(bin_idx, n_bins - 1)  # Handle edge case
            bin_center = min_ret + (bin_idx + 0.5) * bin_width
            
            if bin_center not in bins:
                bins[bin_center] = 0
                bin_dates[bin_center] = []
            bins[bin_center] += 1
            bin_dates[bin_center].append(date.strftime('%Y-%m-%d'))
        
        # Build hover text - show dates if count <= 5
        x_vals = sorted(bins.keys())
        y_vals = [bins[x] for x in x_vals]
        hover_texts = []
        for x in x_vals:
            count = bins[x]
            if count <= 5:
                dates_str = '<br>'.join(bin_dates[x])
                hover_texts.append(f"Return: {x:.2f}%<br>Count: {count}<br>Dates:<br>{dates_str}")
            else:
                hover_texts.append(f"Return: {x:.2f}%<br>Count: {count}")
        
        fig.add_trace(
            go.Bar(x=x_vals, y=y_vals, name='Daily Returns',
                   marker_color='#3366cc', opacity=0.7,
                   hovertext=hover_texts, hoverinfo='text'),
            row=4, col=1
        )
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, row=4, col=1)
        fig.add_vline(x=daily_returns.mean(), line_dash="dash", line_color="green",
                      annotation_text=f"Mean: {daily_returns.mean():.2f}%", row=4, col=1)
    
    # Row 5 Left: Stats gauge/indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=stats['sharpe_ratio'],
            title={'text': "Sharpe Ratio"},
            gauge={
                'axis': {'range': [-1, 4]},
                'bar': {'color': "#109618"},
                'steps': [
                    {'range': [-1, 0], 'color': "#ffcccc"},
                    {'range': [0, 1], 'color': "#ffffcc"},
                    {'range': [1, 2], 'color': "#ccffcc"},
                    {'range': [2, 4], 'color': "#99ff99"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75,
                    'value': stats['sharpe_ratio']
                }
            }
        ),
        row=5, col=1
    )
    
    # Row 5 Right: Weekly returns bar chart
    if len(weekly_returns) > 0:
        colors = ['#109618' if r > 0 else '#dc3912' for r in weekly_returns.values]
        fig.add_trace(
            go.Bar(x=weekly_returns.index, y=weekly_returns.values, name='Weekly Returns',
                   marker_color=colors,
                   hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'),
            row=5, col=2
        )
    
    # Row 6: PNL by Coin Table
    if pnl_by_coin is not None and len(pnl_by_coin) > 0:
        # Calculate color gradient
        min_pnl = pnl_by_coin.min()
        max_pnl = pnl_by_coin.max()
        
        def get_pnl_color(pnl_value):
            """Generate smooth gradient color from red (negative) through yellow (zero) to green (positive)"""
            if max_pnl == min_pnl:
                return 'rgb(255, 255, 0)'
            
            if pnl_value <= 0:
                if min_pnl == 0:
                    ratio = 1
                else:
                    ratio = 1 - (pnl_value / min_pnl)
                r = 255
                g = int(255 * ratio)
                b = 0
            else:
                if max_pnl == 0:
                    ratio = 0
                else:
                    ratio = pnl_value / max_pnl
                r = int(255 * (1 - ratio))
                g = 255
                b = 0
            
            return f'rgb({r}, {g}, {b})'
        
        # Prepare table data
        coins = pnl_by_coin.index.tolist()
        pnls = pnl_by_coin.values.tolist()
        colors = [get_pnl_color(pnl) for pnl in pnls]
        
        # Create table trace
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Coin</b>', '<b>PNL</b>'],
                    fill_color='#1a1a2e',
                    align=['left', 'right'],
                    font=dict(color='white', size=12),
                    height=30
                ),
                cells=dict(
                    values=[coins, [f'{pnl:+.2f}' for pnl in pnls]],
                    fill_color='#16213e',
                    align=['left', 'right'],
                    font=dict(color=[['white'] * len(coins), colors], size=11),
                    height=25
                )
            ),
            row=6, col=1
        )
    
    # Update axes
    min_val = max(df[['balance', 'equity']].min().min(), 0.1)
    max_val = df[['balance', 'equity']].max().max()
    log_min = np.floor(np.log10(min_val))
    log_max = np.ceil(np.log10(max_val))
    
    fig.update_yaxes(title_text="Value (x)", type="log", range=[log_min, log_max], row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=4, col=1)
    fig.update_xaxes(title_text="Daily Return (%)", row=4, col=1)
    fig.update_yaxes(title_text="Return (%)", row=5, col=2)
    
    # Add range slider to main chart
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.05),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='#f0f0f0',
            activecolor='#3366cc',
            x=0, y=1.15
        ),
        row=1, col=1
    )
    
    # Build stats HTML panel with toggle
    stats_html = f"""
    <button id="toggleStats" onclick="toggleStatsPanel()" style="
        position: fixed;
        top: 10px;
        right: 10px;
        padding: 8px 12px;
        background: #3366cc;
        color: white;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-weight: bold;
        z-index: 1001;
        font-family: 'Segoe UI', Arial, sans-serif;
    ">📊 Stats</button>
    <div id="statsPanel" style="
        position: fixed;
        top: 50px;
        right: 10px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13px;
        z-index: 1000;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        min-width: 240px;
        max-height: 85vh;
        overflow-y: auto;
        backdrop-filter: blur(10px);
        display: block;
    ">
        <div style="font-size: 16px; font-weight: bold; margin-bottom: 15px; 
                    border-bottom: 2px solid #3366cc; padding-bottom: 8px;">
            📊 Performance Stats
        </div>
        <div style="display: grid; gap: 8px;">
            <div style="font-weight: bold; color: #3366cc; margin-top: 5px;">Returns</div>
            <div style="display: flex; justify-content: space-between;" title="Total percentage gain/loss over the entire period">
                <span style="color: #aaa;">Total Return:</span>
                <span style="color: {'#4ade80' if stats['total_return'] > 0 else '#f87171'}; font-weight: bold;">
                    {stats['total_return']:.1f}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Compound Annual Growth Rate - annualized return assuming smooth growth">
                <span style="color: #aaa;">CAGR:</span>
                <span style="color: {'#4ade80' if stats['cagr'] > 0 else '#f87171'}; font-weight: bold;">
                    {stats['cagr']:.1f}%
                </span>
            </div>
            
            <div style="font-weight: bold; color: #3366cc; margin-top: 10px;">Risk-Adjusted Returns</div>
            <div style="display: flex; justify-content: space-between;" title="Return per unit of total volatility. >1 good, >2 great, >3 exceptional">
                <span style="color: #aaa;">Sharpe Ratio:</span>
                <span style="color: {'#4ade80' if stats['sharpe_ratio'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['sharpe_ratio']:.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Like Sharpe but only penalizes downside volatility. Higher is better">
                <span style="color: #aaa;">Sortino Ratio:</span>
                <span style="color: {'#4ade80' if stats['sortino_ratio'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['sortino_ratio']:.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="CAGR divided by max drawdown. Measures return per unit of downside risk">
                <span style="color: #aaa;">Calmar Ratio:</span>
                <span style="color: {'#4ade80' if stats['calmar_ratio'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['calmar_ratio']:.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Probability-weighted ratio of gains vs losses. >1 profitable, higher is better">
                <span style="color: #aaa;">Omega Ratio:</span>
                <span style="color: {'#4ade80' if stats['omega_ratio'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['omega_ratio']:.2f}
                </span>
            </div>
            
            <div style="font-weight: bold; color: #3366cc; margin-top: 10px;">Risk Metrics</div>
            <div style="display: flex; justify-content: space-between;" title="Largest peak-to-trough decline">
                <span style="color: #aaa;">Max Drawdown:</span>
                <span style="color: #f87171; font-weight: bold;">{stats['max_drawdown']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Annualized standard deviation of returns">
                <span style="color: #aaa;">Volatility:</span>
                <span style="font-weight: bold;">{stats['annual_volatility']:.1f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="95% confidence - you won't lose more than this on most days">
                <span style="color: #aaa;">VaR (95%):</span>
                <span style="color: #f87171; font-weight: bold;">{stats['var_95']:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Average loss when you exceed VaR - measures tail risk">
                <span style="color: #aaa;">CVaR (95%):</span>
                <span style="color: #f87171; font-weight: bold;">{stats['cvar_95']:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Measures depth and duration of drawdowns. Lower is better">
                <span style="color: #aaa;">Ulcer Index:</span>
                <span style="font-weight: bold;">{stats['ulcer_index']:.2f}</span>
            </div>
            
            <div style="font-weight: bold; color: #3366cc; margin-top: 10px;">Win/Loss Analysis</div>
            <div style="display: flex; justify-content: space-between;" title="Percentage of profitable days">
                <span style="color: #aaa;">Daily Win Rate:</span>
                <span style="color: {'#4ade80' if stats['win_rate'] > 50 else '#fbbf24'}; font-weight: bold;">
                    {stats['win_rate']:.1f}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Percentage of profitable months - shows consistency">
                <span style="color: #aaa;">Monthly Win Rate:</span>
                <span style="color: {'#4ade80' if stats['monthly_win_rate'] > 50 else '#fbbf24'}; font-weight: bold;">
                    {stats['monthly_win_rate']:.1f}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Gross profits divided by gross losses. >1 means profitable">
                <span style="color: #aaa;">Profit Factor:</span>
                <span style="color: {'#4ade80' if stats['profit_factor'] > 1 else '#f87171'}; font-weight: bold;">
                    {stats['profit_factor']:.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Average win divided by average loss. Shows if you cut losses and let winners run">
                <span style="color: #aaa;">Payoff Ratio:</span>
                <span style="color: {'#4ade80' if stats['payoff_ratio'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['payoff_ratio']:.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Average expected return per day">
                <span style="color: #aaa;">Expectancy:</span>
                <span style="color: {'#4ade80' if stats['expectancy'] > 0 else '#f87171'}; font-weight: bold;">
                    {stats['expectancy']:.2f}%
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="95th percentile gain divided by 95th percentile loss">
                <span style="color: #aaa;">Tail Ratio:</span>
                <span style="color: {'#4ade80' if stats['tail_ratio'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['tail_ratio']:.2f}
                </span>
            </div>
            
            <div style="font-weight: bold; color: #3366cc; margin-top: 10px;">Extremes</div>
            <div style="display: flex; justify-content: space-between;" title="Best single day return">
                <span style="color: #aaa;">Best Day:</span>
                <span style="color: #4ade80; font-weight: bold;">+{stats['best_day']:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Worst single day return">
                <span style="color: #aaa;">Worst Day:</span>
                <span style="color: #f87171; font-weight: bold;">{stats['worst_day']:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Longest streak of consecutive winning days">
                <span style="color: #aaa;">Max Win Streak:</span>
                <span style="font-weight: bold;">{stats['max_consecutive_wins']:.0f} days</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Longest streak of consecutive losing days">
                <span style="color: #aaa;">Max Loss Streak:</span>
                <span style="font-weight: bold;">{stats['max_consecutive_losses']:.0f} days</span>
            </div>
            
            <div style="font-weight: bold; color: #3366cc; margin-top: 10px;">Recovery & Position Sizing</div>
            <div style="display: flex; justify-content: space-between;" title="Net profit divided by max drawdown">
                <span style="color: #aaa;">Recovery Factor:</span>
                <span style="color: {'#4ade80' if stats['recovery_factor'] > 1 else '#fbbf24'}; font-weight: bold;">
                    {stats['recovery_factor']:.2f}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Average days to recover from drawdowns">
                <span style="color: #aaa;">Avg Recovery:</span>
                <span style="font-weight: bold;">{stats['avg_recovery_days']:.0f} days</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Longest time underwater before new high">
                <span style="color: #aaa;">Max Recovery:</span>
                <span style="font-weight: bold;">{stats['max_recovery_days']:.0f} days</span>
            </div>
            <div style="display: flex; justify-content: space-between;" title="Optimal position size based on win rate and payoff ratio. Use fraction of this">
                <span style="color: #aaa;">Kelly Criterion:</span>
                <span style="color: {'#4ade80' if stats['kelly_criterion'] > 0 else '#f87171'}; font-weight: bold;">
                    {stats['kelly_criterion']:.1%}
                </span>
            </div>
            
            <div style="border-top: 1px solid #444; margin: 10px 0;"></div>
            <div style="display: flex; justify-content: space-between;" title="Total days in backtest">
                <span style="color: #aaa;">Duration:</span>
                <span style="font-weight: bold;">{stats['duration_days']} days</span>
            </div>
        </div>
        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #444;">
            <button onclick="copyStats()" style="
                width: 100%;
                padding: 8px;
                background: #3366cc;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
            ">📋 Copy Stats</button>
        </div>
    </div>
    <script>
    function toggleStatsPanel() {{
        var panel = document.getElementById('statsPanel');
        var btn = document.getElementById('toggleStats');
        if (panel.style.display === 'none') {{
            panel.style.display = 'block';
            btn.textContent = '📊 Stats';
        }} else {{
            panel.style.display = 'none';
            btn.textContent = '📊 Show Stats';
        }}
    }}
    function copyStats() {{
        const stats = `Run: {run_number}
Total Return: {stats['total_return']:.1f}%
CAGR: {stats['cagr']:.1f}%
Sharpe Ratio: {stats['sharpe_ratio']:.2f}
Sortino Ratio: {stats['sortino_ratio']:.2f}
Calmar Ratio: {stats['calmar_ratio']:.2f}
Omega Ratio: {stats['omega_ratio']:.2f}
Max Drawdown: {stats['max_drawdown']:.1f}%
Volatility: {stats['annual_volatility']:.1f}%
VaR (95%): {stats['var_95']:.2f}%
CVaR (95%): {stats['cvar_95']:.2f}%
Ulcer Index: {stats['ulcer_index']:.2f}
Daily Win Rate: {stats['win_rate']:.1f}%
Monthly Win Rate: {stats['monthly_win_rate']:.1f}%
Profit Factor: {stats['profit_factor']:.2f}
Payoff Ratio: {stats['payoff_ratio']:.2f}
Expectancy: {stats['expectancy']:.2f}%
Tail Ratio: {stats['tail_ratio']:.2f}
Best Day: +{stats['best_day']:.2f}%
Worst Day: {stats['worst_day']:.2f}%
Max Win Streak: {stats['max_consecutive_wins']:.0f} days
Max Loss Streak: {stats['max_consecutive_losses']:.0f} days
Recovery Factor: {stats['recovery_factor']:.2f}
Avg Recovery: {stats['avg_recovery_days']:.0f} days
Max Recovery: {stats['max_recovery_days']:.0f} days
Kelly Criterion: {stats['kelly_criterion']:.1%}
Duration: {stats['duration_days']} days`;
        navigator.clipboard.writeText(stats);
        alert('Stats copied to clipboard!');
    }}
    </script>
    """
    
    # Update layout
    fig.update_layout(
        height=1600,
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=0.7
        ),
        margin=dict(r=250)  # Make room for stats panel
    )
    
    # Write HTML with custom stats panel
    html_content = fig.to_html(full_html=True, include_plotlyjs=True)
    html_content = html_content.replace('</body>', f'{stats_html}</body>')
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    colorful_log(f"✅ Detailed plot saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Balance and Equity from a CSV file.")
    parser.add_argument("csv_file_or_number", help="Path to the CSV file or run number (e.g. 48)")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format (overrides config.json if provided)")
    parser.add_argument("--output", "-o", help="Output filename for the plot (default: auto-generated)")
    parser.add_argument("--save", action="store_true", help="Save plot to file without opening browser")
    args = parser.parse_args()

    input_arg = args.csv_file_or_number
    if input_arg.isdigit():
        colorful_log(f"🔍 Interpreting '{input_arg}' as run number...")
        run_number = str(input_arg).zfill(6)
        csv_file = f"/home/myusuf/Projects/passivbot/backtests/optimizer/live/{run_number}/balance_and_equity.csv"
        if not os.path.isfile(csv_file):
            csv_file = f"/home/myusuf/Projects/passivbot/backtests/optimizer/extremes/{run_number}/balance_and_equity.csv"
        colorful_log(f"📂 Resolved path: {csv_file}")
    else:
        csv_file = input_arg
        run_number = "N/A"
        colorful_log(f"📂 Using direct CSV path: {csv_file}")

    if not os.path.isfile(csv_file):
        colorful_log(f"❌ File not found: {csv_file}", emoji="🚨")
        sys.exit(1)

    start_date = None
    starting_balance = 10000

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            colorful_log(f"📅 Using start date from command line: {args.start_date}")
        except ValueError:
            colorful_log(f"❌ Invalid start date format: {args.start_date}. Use YYYY-MM-DD.", emoji="🚨")
            sys.exit(1)
        _, _, starting_balance = get_dates_from_config(csv_file)
        if starting_balance is None:
            starting_balance = 10000
    else:
        start_date, config_end_date, starting_balance = get_dates_from_config(csv_file)

        if start_date is None:
            start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
            colorful_log(f"⚠️  Using fallback start date: 2020-01-01", emoji="🚨")

        if starting_balance is None:
            starting_balance = 10000
            colorful_log(f"⚠️  Using fallback starting balance: {starting_balance}", emoji="🚨")

    plot_balance_equity(csv_file, run_number, start_date, starting_balance, args.output, args.save)
