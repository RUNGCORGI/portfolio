// web/js/app.js
function app() {
    return {
        activeTab: 'overview',
        loading: false,
        error: null,
        sysInfo: {},
        currentTime: '',
        clockInterval: null,

        data: {
            overview: null, allocations: null, walkforward: null,
            validation: null, regimes: null, factors: null, alerts: null,
            scenarios: null, vn: null, journal: null, quantamental: null
        },
        
        form: {
            tickers: 'AAPL, MSFT, NVDA, JNJ, XOM, KO',
            start: '2018-01-01', method: 'HRP', market: 'us', robust: true, max_weight: 0.4
        },
        wfConfig: { lookback: 252, step: 21, cost_bps: 10 },
        tradeForm: { date: new Date().toISOString().split('T')[0], ticker: '', side: 'MUA', qty: 100, price: 0 },

        tabs: [
            { id: 'overview', name: 'Tổng quan', icon: 'layout-dashboard' },
            { id: 'xray', name: 'X-Ray', icon: 'shield-alert' },
            { id: 'allocations', name: 'Phân bổ', icon: 'pie-chart' },
            { id: 'walkforward', name: 'Walk-forward', icon: 'git-commit' },
            { id: 'validation', name: 'Kiểm định', icon: 'check-circle' },
            { id: 'regimes', name: 'Chế độ', icon: 'activity' },
            { id: 'factors', name: 'Nhân tố', icon: 'layers' },
            { id: 'scenarios', name: 'Kịch bản', icon: 'zap' },
            { id: 'quantamental', name: '🧠 Quantamental AI', icon: 'brain' },
            { id: 'vnmarket', name: 'VN Market', icon: 'flag', marketOnly: 'vn' },
            { id: 'journal', name: 'Nhật ký', icon: 'book' },
            { id: 'monitor', name: 'Giám sát', icon: 'bell' }
        ],

        async init() {
            this.startClock();
            try { this.sysInfo = await api.getSystemInfo(); } catch (e) { console.error(e); }
            this.fetchActiveTab(true);
            this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
        },

        startClock() {
            if (this.clockInterval) clearInterval(this.clockInterval);
            
            const updateTime = () => {
                const now = new Date();
                // Nếu chọn VN: Lấy giờ VN. Nếu chọn Mỹ: Lấy giờ New York.
                const tz = this.form.market === 'vn' ? 'Asia/Ho_Chi_Minh' : 'America/New_York';
                const label = this.form.market === 'vn' ? 'GMT+7' : 'EST/EDT';
                
                this.currentTime = now.toLocaleTimeString('vi-VN', { timeZone: tz }) + ' ' + label;
            };
            updateTime();
            this.clockInterval = setInterval(updateTime, 1000);
        },

        setMarket(mkt) {
            this.form.market = mkt;
            if (mkt === 'vn') this.form.tickers = 'VCB, FPT, HPG, VNM, SSI, GAS';
            else this.form.tickers = 'AAPL, MSFT, NVDA, JNJ, XOM, KO';
            
            this.startClock(); // Cập nhật lại đồng hồ theo múi giờ mới
            
            if (this.activeTab === 'vnmarket') this.activeTab = 'overview';
            this.fetchActiveTab(true);
        },

        switchTab(tabId) {
            this.activeTab = tabId;
            this.fetchActiveTab(false);
            this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
        },

        getBasePayload() {
            return {
                tickers: this.form.tickers.split(',').map(t => t.trim().toUpperCase()).filter(t => t),
                start: this.form.start, method: this.form.method,
                max_weight: parseFloat(this.form.max_weight), market: this.form.market, robust: this.form.robust
            };
        },

        async fetchActiveTab(force = false) {
            if (this.data[this.activeTab] && !force) return;
            this.loading = true;
            this.error = null;
            const p = this.getBasePayload();

            try {
                if (this.activeTab === 'overview' || this.activeTab === 'xray') {
                    const res = await api.getOverview(p);
                    this.data.overview = res;
                    this.drawOverviewCharts(res);
                }
                else if (this.activeTab === 'allocations') {
                    this.data.allocations = await api.getAllocations(p);
                    if (this.data.allocations.scatter) {
                        renderEfficientFrontier(this.data.allocations.scatter);
                    }
                }
                else if (this.activeTab === 'walkforward') {
                    p.methods = ["Chia đều", "Max Sharpe", "HRP"]; p.cost_bps = this.wfConfig.cost_bps; p.lookback = this.wfConfig.lookback; p.step = this.wfConfig.step;
                    this.data.walkforward = await api.getWalkForward(p);
                    this.drawWalkForwardChart();
                }
                else if (this.activeTab === 'validation') {
                    p.n_trials = 20;
                    this.data.validation = await api.getValidation(p);
                    if (this.data.validation.bootstrap) {
                        renderValidationChart(this.data.validation.bootstrap.counts, this.data.validation.bootstrap.edges, this.data.validation.bootstrap.observed);
                    }
                }
                else if (this.activeTab === 'regimes') {
                    this.data.regimes = await api.getRegimes(p);
                    if (this.data.regimes.probabilities) {
                        renderRegimeChart(this.data.regimes.probabilities.dates, this.data.regimes.probabilities.series);
                    }
                    if (this.data.regimes.strategy) {
                        renderRegimeEquity(
                            this.data.regimes.strategy.switching.dates,
                            this.data.regimes.strategy.switching.values,
                            this.data.regimes.strategy.buy_hold.values
                        );
                    }
                }
                else if (this.activeTab === 'factors') {
                    this.data.factors = await api.getFactors(p);
                    if (this.data.factors.betas && this.data.factors.betas.rows) {
                        renderFactorChart(this.data.factors.betas.rows.map(r => r[0]), this.data.factors.betas.rows.map(r => r[1]));
                    }
                    if (this.data.factors.rolling) {
                        renderBetaHeatmap(this.data.factors.rolling.dates, this.data.factors.rolling.series);
                    }
                    if (this.data.factors.active_return) {
                        renderActiveReturn(this.data.factors.active_return.dates, this.data.factors.active_return.values);
                    }
                }
                else if (this.activeTab === 'scenarios') {
                    p.shocks = {"SPY": -0.20, "^VIX": 0.80, "QQQ": -0.25}; p.n_paths = 3000; p.fat_tails = true;
                    this.data.scenarios = await api.getScenarios(p);
                    if (this.data.scenarios.shock && this.data.scenarios.shock.detail) {
                        renderScenarioChart(this.data.scenarios.shock.detail.map(r => r.factor), this.data.scenarios.shock.detail.map(r => r.impact * 100));
                    }
                }
                else if (this.activeTab === 'quantamental') {
                    this.data.quantamental = await api.getQuantamental(p);
                    this.$nextTick(() => {
                        if (this.data.quantamental?.radar) {
                            renderQuantamentalRadar(
                                this.data.quantamental.radar.categories,
                                this.data.quantamental.radar.series
                            );
                        }
                        if (this.data.quantamental?.ab_testing?.curves) {
                            const curves = this.data.quantamental.ab_testing.curves;
                            renderQuantamentalCurves(
                                curves.quant.dates,
                                curves.quant.values,
                                curves.trad.values
                            );
                        }
                    });
                }
                else if (this.activeTab === 'vnmarket') {
                    this.data.vn = await api.getVNUniverse();
                }
                else if (this.activeTab === 'journal') {
                    const [trades, positions] = await Promise.all([api.getJournalTrades(), api.getJournalPositions(p)]);
                    this.data.journal = { trades, positions: positions.positions };
                }
                else if (this.activeTab === 'monitor') {
                    p.thresholds = {}; this.data.alerts = await api.getAlerts(p);
                }

            } catch (e) {
                this.error = e.message;
            } finally {
                this.loading = false;
                this.$nextTick(() => { if (window.lucide) lucide.createIcons(); });
            }
        },

        async submitTrade() {
            this.loading = true;
            try {
                const payload = {
                    trade_date: this.tradeForm.date, ticker: this.tradeForm.ticker.toUpperCase(), side: this.tradeForm.side,
                    quantity: parseFloat(this.tradeForm.qty), price: parseFloat(this.tradeForm.price), note: "", signal: "", followed: true
                };
                await api.addTrade(payload);
                this.data.journal = null; this.fetchActiveTab(true); this.tradeForm.ticker = ''; this.tradeForm.price = 0;
            } catch (e) {
                alert("Lỗi ghi lệnh: " + e.message);
                this.loading = false;
            }
        },

        drawOverviewCharts(res) {
            if (res.curves && res.curves['Danh mục']) {
                renderEquityChart(res.curves['Danh mục'].dates, res.curves['Danh mục'].values, res.curves['Thị trường'] ? res.curves['Thị trường'].values : []);
            }
            if (res.allocation) {
                renderAllocChart(res.allocation.map(a => a.ticker), res.allocation.map(a => a.weight * 100));
            }
            if (res.risk_contribution) {
                renderRiskChart(res.risk_contribution.map(r => r.ticker), res.risk_contribution.map(r => r.capital * 100), res.risk_contribution.map(r => r.risk * 100));
            }
            if (res.price_data) {
                renderPriceChart(res.price_data.dates, res.price_data.prices, res.price_data.ma20, res.price_data.ma50);
                if (res.price_data.hist_counts && res.price_data.hist_edges) {
                    renderReturnsHistogram(res.price_data.hist_counts, res.price_data.hist_edges);
                }
            }
        },

        drawWalkForwardChart() {
            const dataWF = this.data.walkforward;
            if (!dataWF || !dataWF.curves) return;
            const best = dataWF.best_method || Object.keys(dataWF.curves)[0];
            const eq = dataWF.curves[best];
            if (eq) {
                const bench = dataWF.curves['Thị trường'];
                renderWFChart(eq.dates, eq.values, bench ? bench.dates : eq.dates, bench ? bench.values : eq.values);
            }
        },

        fmtPct(val, key) {
            if (typeof val !== 'number') return '...';
            if (['Tổng lợi nhuận', 'CAGR', 'Sụt giảm tối đa', 'Biến động năm', 'VaR 95%', 'CVaR 95%'].includes(key)) {
                return (val * 100).toFixed(2) + '%';
            }
            return val.toFixed(2);
        }
    }
}
