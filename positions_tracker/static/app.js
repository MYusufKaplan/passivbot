/* ===== Positions Tracker — Frontend Application Logic ===== */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
    data: null,              // latest valid API response
    charts: {},              // symbol -> { chart, candleSeries, markers, pricelines }
    chartTimeframes: {},     // symbol -> selected timeframe string
    ohlcvCache: {},          // "symbol|tf" -> { data: [], fetchedAt: timestamp }
    sortColumn: 'close_timestamp',
    sortDirection: 'desc',
    pollInterval: 2000,      // ms — 2s active, 5s idle
    timers: {
        poll: null,
        clientUpdate: null
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format seconds into "Xh Xm Xs" with zero-component omission. */
function formatDuration(totalSeconds) {
    if (totalSeconds == null || totalSeconds < 0) totalSeconds = 0;
    totalSeconds = Math.floor(totalSeconds);
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

/** Format a number with fixed decimals, adding sign for positive values. */
function fmtSigned(v, decimals = 2) {
    if (v == null) return '—';
    const n = Number(v);
    return (n >= 0 ? '+' : '') + n.toFixed(decimals);
}

/** Format a number with fixed decimals (no sign prefix). */
function fmt(v, decimals = 2) {
    if (v == null) return '—';
    return Number(v).toFixed(decimals);
}

/** Choose appropriate decimal places for a price value. */
function priceDp(price) {
    if (price == null) return 2;
    const p = Math.abs(Number(price));
    if (p >= 1000) return 2;
    if (p >= 1) return 4;
    return 6;
}

/** Apply PnL color class to an element. */
function applyPnlColor(el, value) {
    el.classList.remove('pnl-positive', 'pnl-negative');
    if (value > 0) el.classList.add('pnl-positive');
    else if (value < 0) el.classList.add('pnl-negative');
}


// ---------------------------------------------------------------------------
// 8.1 — Polling Logic
// ---------------------------------------------------------------------------

/** Fetch /api/data, update panels, handle errors. */
async function pollData() {
    try {
        const resp = await fetch('/api/data');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        state.data = data;
        hideConnectionLost();
        hideLoading();
        updateLastUpdated();

        // Adjust poll interval based on idle state
        const newInterval = data.idle ? 5000 : 2000;
        if (newInterval !== state.pollInterval) {
            state.pollInterval = newInterval;
            restartPolling();
        }

        // Update all panels
        renderSummaryBar(data);
        renderPositions(data);
        renderAccountPanel(data);
        renderHistoryTable(data);
        renderIdleState(data);
    } catch (_err) {
        showConnectionLost();
        // Retain last valid state — do not clear state.data
    }
}

function showConnectionLost() {
    const el = document.getElementById('connection-lost');
    if (el) el.style.display = '';
}

function hideConnectionLost() {
    const el = document.getElementById('connection-lost');
    if (el) el.style.display = 'none';
}

function hideLoading() {
    const el = document.getElementById('loading-indicator');
    if (el) el.style.display = 'none';
    // Show panels that were hidden until first data
    ['account-panel', 'history-panel', 'summary-bar'].forEach(id => {
        const panel = document.getElementById(id);
        if (panel) panel.style.display = '';
    });
}

function updateLastUpdated() {
    const el = document.getElementById('last-updated');
    if (el) el.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
}

function restartPolling() {
    if (state.timers.poll) clearInterval(state.timers.poll);
    state.timers.poll = setInterval(pollData, state.pollInterval);
}


// ---------------------------------------------------------------------------
// Summary Bar
// ---------------------------------------------------------------------------

function renderSummaryBar(data) {
    const account = data.account;
    const rate = data.usdt_try_rate || 1;

    // Profit in TRY (equity - starting balance)
    const profitEl = document.querySelector('[data-field="summary-profit-try"]');
    if (profitEl && account) {
        const profit = account.profit_try || 0;
        profitEl.textContent = fmtSigned(profit, 0) + ' TRY';
        applyPnlColor(profitEl, profit);
    }

    // Positions + unrealized PnL
    const posContainer = document.getElementById('summary-positions');
    if (posContainer) {
        posContainer.innerHTML = '';
        (data.positions || []).forEach(pos => {
            const item = document.createElement('span');
            item.className = 'summary-pos-item';

            const sym = document.createElement('span');
            sym.className = 'summary-pos-symbol';
            sym.textContent = pos.visual_symbol || pos.symbol;

            const pnl = document.createElement('span');
            pnl.className = 'summary-pos-pnl';
            const uPnl = pos.unrealized_pnl || 0;
            pnl.textContent = fmtSigned(uPnl);
            applyPnlColor(pnl, uPnl);

            item.appendChild(sym);
            item.appendChild(pnl);
            posContainer.appendChild(item);
        });
        if ((data.positions || []).length === 0) {
            posContainer.textContent = 'No positions';
        }
    }

    // ¢/s and daily rates
    if (account) {
        const projections = account.projections;
        ['24h', '7d', '30d'].forEach(window => {
            const stats = (account.rate_stats || {})[window];
            const proj = projections && projections[window];

            const cpsEl = document.querySelector(`[data-field="summary-cps-${window}"]`);
            if (cpsEl && stats) {
                cpsEl.textContent = fmt(stats.cents_per_sec, 4) + ' ¢/s';
            }

            const drEl = document.querySelector(`[data-field="summary-dr-${window}"]`);
            if (drEl && proj && proj.daily_rate != null) {
                drEl.textContent = '(' + (proj.daily_rate * 100).toFixed(2) + '%/d)';
            }
        });
    }
}

// ---------------------------------------------------------------------------
// 8.2 — Position Card Rendering
// ---------------------------------------------------------------------------

function renderPositions(data) {
    const container = document.getElementById('positions-container');
    if (!container) return;

    const positions = data.positions || [];
    const rate = data.usdt_try_rate || 1;

    // Remove cards for symbols no longer present
    const currentSymbols = new Set(positions.map(p => p.symbol));
    container.querySelectorAll('.position-card').forEach(card => {
        if (!currentSymbols.has(card.dataset.symbol)) {
            // Destroy chart if exists
            if (state.charts[card.dataset.symbol]) {
                state.charts[card.dataset.symbol].chart.remove();
                delete state.charts[card.dataset.symbol];
            }
            card.remove();
        }
    });

    positions.forEach(pos => {
        let card = container.querySelector(`.position-card[data-symbol="${pos.symbol}"]`);
        if (!card) {
            card = createPositionCard(pos);
            container.appendChild(card);
        }
        updatePositionCard(card, pos, rate);
    });
}

function createPositionCard(pos) {
    const tpl = document.getElementById('position-card-template');
    const clone = tpl.content.cloneNode(true);
    const card = clone.querySelector('.position-card');
    card.dataset.symbol = pos.symbol;
    return card;
}

function updatePositionCard(card, pos, rate) {
    // Header
    card.querySelector('.symbol-name').textContent = pos.visual_symbol || pos.symbol;
    card.querySelector('.contracts').textContent = `${pos.contracts} contracts`;
    const dp = priceDp(pos.current_price);
    card.querySelector('.current-price').textContent = fmt(pos.current_price, dp);

    // Price change badges
    const badges = card.querySelectorAll('.price-changes .badge');
    const timeframes = ['5m', '15m', '1h', '1d'];
    badges.forEach((badge, i) => {
        const tf = timeframes[i];
        const val = (pos.price_changes || {})[tf] || '—';
        badge.textContent = `${tf}: ${val}`;
        badge.classList.remove('positive', 'negative');
        if (typeof val === 'string') {
            if (val.startsWith('+')) badge.classList.add('positive');
            else if (val.startsWith('-')) badge.classList.add('negative');
        }
    });

    // PnL section
    const uPnl = pos.unrealized_pnl || 0;
    const rPnl = pos.realized_pnl || 0;
    const uPnlUsdt = card.querySelector('[data-field="unrealized-pnl-usdt"]');
    const uPnlTry = card.querySelector('[data-field="unrealized-pnl-try"]');
    const rPnlUsdt = card.querySelector('[data-field="realized-pnl-usdt"]');
    const rPnlTry = card.querySelector('[data-field="realized-pnl-try"]');

    if (uPnlUsdt) { uPnlUsdt.textContent = `${fmtSigned(uPnl)} USDT`; applyPnlColor(uPnlUsdt, uPnl); }
    if (uPnlTry) { uPnlTry.textContent = `${fmtSigned(uPnl * rate)} TRY`; applyPnlColor(uPnlTry, uPnl); }
    if (rPnlUsdt) { rPnlUsdt.textContent = `${fmtSigned(rPnl)} USDT`; applyPnlColor(rPnlUsdt, rPnl); }
    if (rPnlTry) { rPnlTry.textContent = `${fmtSigned(rPnl * rate)} TRY`; applyPnlColor(rPnlTry, rPnl); }

    // Entry / Break-even prices
    const entryDp = priceDp(pos.entry_price);
    const entryEl = card.querySelector('[data-field="entry-price"]');
    if (entryEl) entryEl.textContent = fmt(pos.entry_price, entryDp);

    const beDp = priceDp(pos.break_even_price);
    const beEl = card.querySelector('[data-field="break-even-price"]');
    if (beEl) beEl.textContent = pos.break_even_price != null ? fmt(pos.break_even_price, beDp) : '—';

    // Chart — fetched async via setupTimeframeSelector
    setupTimeframeSelector(card, pos);
    // Refresh chart data on each poll (client cache absorbs redundant fetches)
    const tf = state.chartTimeframes[pos.symbol] || '15m';
    fetchOhlcvCached(pos.symbol, tf).then(ohlcv => renderChart(card, pos, ohlcv));

    // Orders
    renderOrders(card, pos);

    // Funding
    renderFunding(card, pos);

    // Duration — store open_timestamp for client-side timer
    const durEl = card.querySelector('[data-field="trade-duration"]');
    if (durEl) {
        durEl.dataset.openTs = pos.open_timestamp || 0;
        const elapsed = Math.floor(Date.now() / 1000) - (pos.open_timestamp || 0);
        durEl.textContent = formatDuration(elapsed);
    }

    // Risk indicators
    const levEl = card.querySelector('[data-field="leverage"]');
    if (levEl) levEl.textContent = pos.leverage ? `Lev: ${pos.leverage}` : '';

    const liqEl = card.querySelector('[data-field="liquidation-price"]');
    if (liqEl) liqEl.textContent = pos.liquidation_price != null ? `Liq: ${fmt(pos.liquidation_price, priceDp(pos.liquidation_price))}` : '';

    const mrEl = card.querySelector('[data-field="margin-ratio"]');
    if (mrEl) mrEl.textContent = pos.margin_ratio != null ? `Margin: ${(pos.margin_ratio * 100).toFixed(1)}%` : '';
}


// ---------------------------------------------------------------------------
// 8.3 — TradingView Lightweight Charts
// ---------------------------------------------------------------------------

const OHLCV_CACHE_TTL = 10000; // 10s client-side cache (server caches for 30s anyway)

/** Fetch OHLCV data with client-side caching. Returns array or []. */
async function fetchOhlcvCached(symbol, tf) {
    const key = `${symbol}|${tf}`;
    const cached = state.ohlcvCache[key];
    const now = Date.now();
    if (cached && (now - cached.fetchedAt) < OHLCV_CACHE_TTL) {
        return cached.data;
    }
    try {
        const resp = await fetch(`/api/ohlcv?symbol=${encodeURIComponent(symbol)}&timeframe=${tf}&limit=60`);
        const json = await resp.json();
        const data = json.ohlcv || [];
        state.ohlcvCache[key] = { data, fetchedAt: now };
        return data;
    } catch (_e) {
        return cached ? cached.data : [];
    }
}

function setupTimeframeSelector(card, pos) {
    const symbol = pos.symbol;
    const selector = card.querySelector('.timeframe-selector');
    if (!selector || selector.dataset.bound) return;
    selector.dataset.bound = '1';

    selector.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const tf = btn.dataset.tf;
            selector.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.chartTimeframes[symbol] = tf;
            const ohlcv = await fetchOhlcvCached(symbol, tf);
            renderChart(card, pos, ohlcv);
        });
    });

    // Fetch default timeframe on first render
    const defaultTf = state.chartTimeframes[symbol] || '15m';
    fetchOhlcvCached(symbol, defaultTf).then(ohlcv => renderChart(card, pos, ohlcv));
}

function renderChart(card, pos, ohlcv) {
    const container = card.querySelector('.chart-container');
    if (!container) return;

    const symbol = pos.symbol;
    if (!ohlcv) ohlcv = [];

    // Check if LightweightCharts is available
    if (typeof LightweightCharts === 'undefined') {
        container.textContent = 'Chart unavailable';
        return;
    }

    // Create or reuse chart instance
    let chartObj = state.charts[symbol];
    if (!chartObj) {
        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight || 300,
            layout: {
                background: { color: '#1a1a2e' },
                textColor: '#a0a0a0',
            },
            grid: {
                vertLines: { color: 'rgba(15, 52, 96, 0.3)' },
                horzLines: { color: 'rgba(15, 52, 96, 0.3)' },
            },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
            timeScale: { timeVisible: true, secondsVisible: false },
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#4caf50',
            downColor: '#f44336',
            borderUpColor: '#4caf50',
            borderDownColor: '#f44336',
            wickUpColor: '#4caf50',
            wickDownColor: '#f44336',
        });

        chartObj = { chart, candleSeries, priceLines: [], markers: [] };
        state.charts[symbol] = chartObj;

        // Resize observer
        const ro = new ResizeObserver(() => {
            chart.applyOptions({ width: container.clientWidth });
        });
        ro.observe(container);
    }

    // Update candle data
    if (ohlcv.length > 0) {
        const candles = ohlcv.map(c => ({
            time: Math.floor(c[0] / 1000) + 3 * 3600,  // ms -> s, UTC+3
            open: c[1],
            high: c[2],
            low: c[3],
            close: c[4],
        }));
        chartObj.candleSeries.setData(candles);
    }

    // Remove old price lines
    chartObj.priceLines.forEach(pl => {
        try { chartObj.candleSeries.removePriceLine(pl); } catch (_e) { /* ignore */ }
    });
    chartObj.priceLines = [];

    // Break-even price line
    if (pos.break_even_price != null) {
        const pl = chartObj.candleSeries.createPriceLine({
            price: pos.break_even_price,
            color: '#e0e0e0',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: 'BE',
        });
        chartObj.priceLines.push(pl);
    }

    // Order price lines
    let buyLineIdx = 0;
    let sellLineIdx = 0;
    (pos.orders || []).forEach(order => {
        const isBuy = order.side === 'buy';
        const idx = isBuy ? ++buyLineIdx : ++sellLineIdx;
        const color = isBuy ? '#4caf50' : '#f44336';
        const label = isBuy ? `B${idx}` : `S${idx}`;
        const pl = chartObj.candleSeries.createPriceLine({
            price: order.price,
            color: color,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: label,
        });
        chartObj.priceLines.push(pl);
    });

    // Trade execution markers — disabled (too distracting)
    chartObj.candleSeries.setMarkers([]);
}


// ---------------------------------------------------------------------------
// 8.4 — Order Progress Bars
// ---------------------------------------------------------------------------

function renderOrders(card, pos) {
    const container = card.querySelector('[data-field="orders"]');
    if (!container) return;
    container.innerHTML = '';

    const orders = pos.orders || [];
    if (orders.length === 0) return;

    // Orders are already sorted by price descending from backend
    let buyIdx = 0;
    let sellIdx = 0;

    orders.forEach(order => {
        const isBuy = order.side === 'buy';
        const idx = isBuy ? ++buyIdx : ++sellIdx;
        const label = isBuy ? `Buy ${idx}` : `Sell ${idx}`;

        const row = document.createElement('div');
        row.className = 'order-row';

        // Side label
        const labelEl = document.createElement('span');
        labelEl.className = `order-label ${order.side}`;
        labelEl.textContent = label;

        // Details: qty @ price
        const detailsEl = document.createElement('span');
        detailsEl.className = 'order-details';
        detailsEl.textContent = `${fmt(order.amount, 4)} @ ${fmt(order.price, priceDp(order.price))}`;

        // Progress bar wrapper
        const barWrapper = document.createElement('div');
        barWrapper.className = 'order-progress-bar';

        const fill = document.createElement('div');
        fill.className = 'order-progress-fill';
        const pct = Math.min(100, Math.max(0, order.progress_percent || 0));
        fill.style.width = pct + '%';
        fill.style.backgroundColor = order.color || '#555';

        const pctText = document.createElement('span');
        pctText.className = 'order-progress-text';
        pctText.textContent = pct.toFixed(1) + '%';

        barWrapper.appendChild(fill);
        barWrapper.appendChild(pctText);

        // Extra info: cumulative avg for buys, estimated PnL for sells
        const extraEl = document.createElement('span');
        extraEl.className = 'order-extra';
        if (isBuy && order.cumulative_avg_price != null) {
            extraEl.textContent = `Avg: ${fmt(order.cumulative_avg_price, priceDp(order.cumulative_avg_price))}`;
        } else if (!isBuy && order.estimated_pnl != null) {
            extraEl.textContent = `PnL: ${fmtSigned(order.estimated_pnl)}`;
            applyPnlColor(extraEl, order.estimated_pnl);
        }

        // Tooltip on hover
        barWrapper.title = [
            `Order: ${order.id}`,
            `Side: ${order.side}`,
            `Price: ${fmt(order.price, priceDp(order.price))}`,
            `Amount: ${fmt(order.amount, 4)}`,
            `Remaining: ${fmt(order.remaining, 4)}`,
            `Distance: ${pct.toFixed(1)}%`,
            order.cumulative_avg_price != null ? `Cum Avg: ${fmt(order.cumulative_avg_price, priceDp(order.cumulative_avg_price))}` : '',
            order.estimated_pnl != null ? `Est PnL: ${fmtSigned(order.estimated_pnl)}` : '',
        ].filter(Boolean).join('\n');

        row.appendChild(labelEl);
        row.appendChild(detailsEl);
        row.appendChild(barWrapper);
        row.appendChild(extraEl);
        container.appendChild(row);
    });
}


// ---------------------------------------------------------------------------
// 8.5 — Client-Side Timers (1s interval)
// ---------------------------------------------------------------------------

/** Update all client-side timers: trade duration, funding countdown, time ago. */
function updateClientTimers() {
    const now = Math.floor(Date.now() / 1000);

    // Trade duration timers
    document.querySelectorAll('[data-field="trade-duration"]').forEach(el => {
        const openTs = parseInt(el.dataset.openTs, 10);
        if (openTs > 0) {
            el.textContent = formatDuration(now - openTs);
        }
    });

    // Funding countdown timers
    document.querySelectorAll('[data-field="funding-countdown"]').forEach(el => {
        const nextTs = parseInt(el.dataset.nextFundingTs, 10);
        if (nextTs > 0) {
            const remaining = Math.max(0, nextTs - now);
            el.textContent = formatDuration(remaining);
        }
    });

    // History "time ago" cells
    document.querySelectorAll('[data-field="time-ago"]').forEach(el => {
        const closeTs = parseInt(el.dataset.closeTs, 10);
        if (closeTs > 0) {
            el.textContent = formatDuration(now - closeTs) + ' ago';
        }
    });

    // Idle timer
    const idleTimerEl = document.querySelector('[data-field="idle-timer"]');
    if (idleTimerEl && state.data && state.data.idle) {
        // Show time since last server timestamp
        const serverTs = state.data.timestamp || now;
        idleTimerEl.textContent = formatDuration(now - Math.floor(serverTs));
    }
}


// ---------------------------------------------------------------------------
// 8.6 — Funding Info Display
// ---------------------------------------------------------------------------

function renderFunding(card, pos) {
    const funding = pos.funding || {};

    // Funding rate
    const rateEl = card.querySelector('[data-field="funding-rate"]');
    if (rateEl) {
        const rateVal = funding.rate || 0;
        rateEl.textContent = (rateVal * 100).toFixed(4) + '%';
    }

    // Expected payment
    const payEl = card.querySelector('[data-field="funding-payment"]');
    if (payEl) {
        const payment = funding.expected_payment || 0;
        payEl.textContent = fmtSigned(payment) + ' USDT';
        applyPnlColor(payEl, -payment); // negative payment = cost
    }

    // Funding countdown — store timestamp for client-side timer
    const cdEl = card.querySelector('[data-field="funding-countdown"]');
    if (cdEl) {
        cdEl.dataset.nextFundingTs = funding.next_funding_timestamp || 0;
        const now = Math.floor(Date.now() / 1000);
        const remaining = Math.max(0, (funding.next_funding_timestamp || 0) - now);
        cdEl.textContent = formatDuration(remaining);
    }

    // Last funding payment
    const lastEl = card.querySelector('[data-field="last-funding-payment"]');
    if (lastEl) {
        const lastPay = funding.last_payment || 0;
        lastEl.textContent = fmtSigned(lastPay) + ' USDT';
        applyPnlColor(lastEl, lastPay);
    }
}


// ---------------------------------------------------------------------------
// 8.7 — Account Metrics Panel
// ---------------------------------------------------------------------------

function renderAccountPanel(data) {
    const account = data.account;
    if (!account) return;

    const rate = data.usdt_try_rate || 1;

    // Equity headline
    const eqUsdt = document.querySelector('[data-field="equity-usdt"]');
    if (eqUsdt) eqUsdt.textContent = fmt(account.equity_usdt);

    const eqTry = document.querySelector('[data-field="equity-try"]');
    if (eqTry) eqTry.textContent = fmt(account.equity_try, 0);

    // Profit from STARTING_BALANCE
    const profitEl = document.querySelector('[data-field="profit-try"]');
    if (profitEl) {
        const profit = account.profit_try || 0;
        profitEl.textContent = fmtSigned(profit, 0) + ' TRY';
        profitEl.classList.remove('pnl-positive', 'pnl-negative');
        if (profit > 0) profitEl.classList.add('pnl-positive');
        else if (profit < 0) profitEl.classList.add('pnl-negative');
    }

    // PnL rate stat cards
    renderRateStats(account.rate_stats);

    // Projection table
    renderProjections(account.projections, rate);

    // Custom target ETA
    renderCustomTarget(account.projections, account.equity_usdt);

    // Custom income ETA
    renderCustomIncome(account.projections, account.equity_usdt);

    // Risk indicators per position (leverage, liquidation, margin)
    // These are rendered inside each position card in updatePositionCard
}

function renderRateStats(rateStats) {
    if (!rateStats) return;

    ['24h', '7d', '30d'].forEach(window => {
        const stats = rateStats[window];
        if (!stats) return;

        const rateEl = document.querySelector(`[data-field="rate-${window}"]`);
        if (rateEl) {
            rateEl.textContent = fmt(stats.cents_per_sec, 4) + ' ¢/s';
            rateEl.classList.remove('pnl-positive', 'pnl-negative');
            if (stats.cents_per_sec > 0) rateEl.classList.add('pnl-positive');
            else if (stats.cents_per_sec < 0) rateEl.classList.add('pnl-negative');
        }

        const totalEl = document.querySelector(`[data-field="total-pnl-${window}"]`);
        if (totalEl) totalEl.textContent = fmtSigned(stats.total_pnl) + ' USDT';

        const countEl = document.querySelector(`[data-field="count-${window}"]`);
        if (countEl) countEl.textContent = `${stats.count || 0} trades`;
    });
}

function renderProjections(projections, rate) {
    if (!projections) return;

    ['24h', '7d', '30d'].forEach(window => {
        const proj = projections[window];
        if (!proj) return;

        // Daily rate
        const drEl = document.querySelector(`[data-field="proj-daily-rate-${window}"]`);
        if (drEl) drEl.textContent = proj.daily_rate != null ? (proj.daily_rate * 100).toFixed(2) + '%' : '—';

        // 1d / 1w / 1m projected equity
        ['1d', '1w', '1m'].forEach(horizon => {
            const el = document.querySelector(`[data-field="proj-${horizon}-${window}"]`);
            if (el && proj[horizon] != null) {
                const usdt = proj[horizon];
                const tryVal = usdt * rate;
                el.textContent = `${fmt(usdt)} / ${fmt(tryVal, 0)} TRY`;
            }
        });

        // Milestone ETAs: 2x, 5x, 10x
        ['2x', '5x', '10x'].forEach(milestone => {
            const el = document.querySelector(`[data-field="proj-${milestone}-${window}"]`);
            if (el && proj[milestone] != null) {
                el.textContent = formatMilestoneEta(proj[milestone]);
            }
        });
    });
}

/** Format milestone ETA in days to human-readable string. */
function formatMilestoneEta(days) {
    if (days == null || days <= 0) return '—';
    if (days < 1) return formatDuration(days * 86400);
    if (days < 365) return Math.round(days) + 'd';
    return (days / 365).toFixed(1) + 'y';
}

/** Compute ETA to reach a custom target equity using compound growth. */
function renderCustomTarget(projections, equity) {
    const input = document.getElementById('custom-target-input');
    if (!input) return;

    const targetProfitTry = parseFloat(input.value);
    const rate = state.data && state.data.usdt_try_rate ? state.data.usdt_try_rate : 1;
    const STARTING_BALANCE = 150000;
    const target = (targetProfitTry + STARTING_BALANCE) / rate; // profit TRY -> equity USDT

    ['24h', '7d', '30d'].forEach(window => {
        const el = document.querySelector(`[data-field="proj-custom-${window}"]`);
        if (!el) return;

        if (!target || target <= 0 || !equity || equity <= 0 || target <= equity) {
            el.textContent = '—';
            return;
        }

        const proj = projections && projections[window];
        if (!proj || !proj.daily_rate || proj.daily_rate <= 0) {
            el.textContent = '—';
            return;
        }

        const days = Math.log(target / equity) / Math.log(1 + proj.daily_rate);
        el.textContent = formatMilestoneEta(days);
    });
}


/** Compute ETA until daily income reaches a custom TRY target per period. */
function renderCustomIncome(projections, equity) {
    const input = document.getElementById('custom-income-input');
    const periodSelect = document.getElementById('custom-income-period');
    if (!input) return;

    const targetTry = parseFloat(input.value);
    const period = periodSelect ? periodSelect.value : 'day';
    const periodDays = period === 'month' ? 30 : period === 'week' ? 7 : 1;
    const rate = state.data && state.data.usdt_try_rate ? state.data.usdt_try_rate : 1;
    const targetUsdtPerDay = targetTry / rate / periodDays;

    ['24h', '7d', '30d'].forEach(window => {
        const el = document.querySelector(`[data-field="proj-income-${window}"]`);
        if (!el) return;

        if (!targetUsdtPerDay || targetUsdtPerDay <= 0 || !equity || equity <= 0) {
            el.textContent = '—';
            return;
        }

        const proj = projections && projections[window];
        if (!proj || !proj.daily_rate || proj.daily_rate <= 0) {
            el.textContent = '—';
            return;
        }

        const currentDailyUsdt = equity * proj.daily_rate;
        if (currentDailyUsdt >= targetUsdtPerDay) {
            el.textContent = 'Now';
            return;
        }

        const days = Math.log(targetUsdtPerDay / currentDailyUsdt) / Math.log(1 + proj.daily_rate);
        el.textContent = formatMilestoneEta(days);
    });
}


// ---------------------------------------------------------------------------
// 8.8 — History Table
// ---------------------------------------------------------------------------

function renderHistoryTable(data) {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;

    let history = data.history || [];

    // Sort
    history = sortHistory(history);

    // Compute gradient normalizations
    const pnlValues = history.map(h => h.pnl);
    const pnlRateValues = history.map(h => h.pnl_per_sec);
    const pnlNorm = normalizePnl(pnlValues);
    const pnlRateNorm = normalizePnlRate(pnlRateValues);

    const now = Math.floor(Date.now() / 1000);

    tbody.innerHTML = '';
    history.forEach((item, i) => {
        const tr = document.createElement('tr');

        // Symbol
        const tdSym = document.createElement('td');
        tdSym.textContent = item.symbol || '—';
        tr.appendChild(tdSym);

        // PnL with gradient background
        const tdPnl = document.createElement('td');
        tdPnl.className = 'pnl-cell';
        tdPnl.textContent = fmtSigned(item.pnl);
        applyPnlGradient(tdPnl, pnlNorm[i], item.pnl);
        tr.appendChild(tdPnl);

        // PnL/s with log gradient
        const tdRate = document.createElement('td');
        tdRate.className = 'pnl-cell pnl-rate-gradient';
        tdRate.textContent = item.pnl_per_sec != null ? (item.pnl_per_sec * 100).toFixed(4) + ' ¢/s' : '—';
        applyPnlRateGradient(tdRate, pnlRateNorm[i]);
        tr.appendChild(tdRate);

        // Duration
        const tdDur = document.createElement('td');
        tdDur.textContent = formatDuration(item.duration_seconds);
        tr.appendChild(tdDur);

        // Time ago (client-side updated)
        const tdAgo = document.createElement('td');
        tdAgo.dataset.field = 'time-ago';
        tdAgo.dataset.closeTs = item.close_timestamp || 0;
        const elapsed = now - (item.close_timestamp || 0);
        tdAgo.textContent = formatDuration(elapsed) + ' ago';
        tr.appendChild(tdAgo);

        tbody.appendChild(tr);
    });
}

function sortHistory(history) {
    const col = state.sortColumn;
    const dir = state.sortDirection === 'asc' ? 1 : -1;

    return [...history].sort((a, b) => {
        let va = a[col];
        let vb = b[col];
        if (col === 'symbol') {
            return dir * String(va || '').localeCompare(String(vb || ''));
        }
        va = Number(va) || 0;
        vb = Number(vb) || 0;
        return dir * (va - vb);
    });
}

/** Linear min-max normalization for PnL values → [0, 1]. */
function normalizePnl(values) {
    if (values.length === 0) return [];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    if (range === 0) return values.map(() => 0.5);
    return values.map(v => (v - min) / range);
}

/** Logarithmic normalization for PnL/s values → [0, 1]. */
function normalizePnlRate(values) {
    if (values.length === 0) return [];
    // Shift all values to positive, then log-normalize
    const absValues = values.map(v => Math.abs(v));
    const maxAbs = Math.max(...absValues);
    if (maxAbs === 0) return values.map(() => 0);
    return absValues.map(v => {
        if (v === 0) return 0;
        return Math.log1p(v) / Math.log1p(maxAbs);
    });
}

/** Apply red→green gradient to PnL cell based on normalized value. */
function applyPnlGradient(td, norm, rawPnl) {
    td.classList.remove('pnl-gradient-loss', 'pnl-gradient-profit');
    if (rawPnl >= 0) {
        const alpha = 0.1 + norm * 0.3;
        td.style.backgroundColor = `rgba(76, 175, 80, ${alpha})`;
        td.style.color = '#4caf50';
    } else {
        const alpha = 0.1 + (1 - norm) * 0.3;
        td.style.backgroundColor = `rgba(244, 67, 54, ${alpha})`;
        td.style.color = '#f44336';
    }
}

/** Apply neutral→blue gradient to PnL/s cell based on log-normalized value. */
function applyPnlRateGradient(td, norm) {
    const alpha = norm * 0.35;
    td.style.backgroundColor = `rgba(33, 150, 243, ${alpha})`;
}

// Sort click handlers
function initSortHandlers() {
    document.querySelectorAll('#history-table th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (state.sortColumn === col) {
                state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                state.sortColumn = col;
                state.sortDirection = 'desc';
            }
            updateSortIndicators();
            if (state.data) renderHistoryTable(state.data);
        });
    });
}

function updateSortIndicators() {
    document.querySelectorAll('#history-table th.sortable').forEach(th => {
        const indicator = th.querySelector('.sort-indicator');
        th.classList.remove('active');
        if (indicator) indicator.textContent = '';
        if (th.dataset.sort === state.sortColumn) {
            th.classList.add('active');
            if (indicator) indicator.textContent = state.sortDirection === 'asc' ? '▲' : '▼';
        }
    });
}


// ---------------------------------------------------------------------------
// 8.9 — Idle State
// ---------------------------------------------------------------------------

function renderIdleState(data) {
    const idleSection = document.getElementById('idle-state');
    const posContainer = document.getElementById('positions-container');

    if (!idleSection) return;

    if (data.idle) {
        // Show idle state, hide positions container
        idleSection.style.display = '';
        if (posContainer) posContainer.style.display = 'none';

        // Update idle timer
        const timerEl = idleSection.querySelector('[data-field="idle-timer"]');
        if (timerEl) {
            const now = Math.floor(Date.now() / 1000);
            const serverTs = data.timestamp || now;
            timerEl.textContent = formatDuration(now - Math.floor(serverTs));
        }
    } else {
        // Hide idle state, show positions
        idleSection.style.display = 'none';
        if (posContainer) posContainer.style.display = '';
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

function init() {
    // Set up sort handlers
    initSortHandlers();
    updateSortIndicators();

    // Custom target input — recalculate on change
    const targetInput = document.getElementById('custom-target-input');
    if (targetInput) {
        targetInput.addEventListener('input', () => {
            if (state.data && state.data.account) {
                renderCustomTarget(state.data.account.projections, state.data.account.equity_usdt);
            }
        });
    }

    // Custom income input — recalculate on change
    const incomeInput = document.getElementById('custom-income-input');
    const incomePeriod = document.getElementById('custom-income-period');
    const recalcIncome = () => {
        if (state.data && state.data.account) {
            renderCustomIncome(state.data.account.projections, state.data.account.equity_usdt);
        }
    };
    if (incomeInput) incomeInput.addEventListener('input', recalcIncome);
    if (incomePeriod) incomePeriod.addEventListener('change', recalcIncome);

    // Initial poll
    pollData();

    // Start polling interval
    state.timers.poll = setInterval(pollData, state.pollInterval);

    // Start client-side timer updates (1s)
    state.timers.clientUpdate = setInterval(updateClientTimers, 1000);
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
