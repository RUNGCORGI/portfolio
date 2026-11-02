// web/js/charts.js
let equityChartInstance = null;
let allocChartInstance = null;
let riskChartInstance = null;
let wfChartInstance = null;
let validationChartInstance = null;
let regimeChartInstance = null;
let factorChartInstance = null;
let scenarioChartInstance = null;
let priceChartInstance = null;
let histogramChartInstance = null;
let efficientFrontierChartInstance = null;
let regimeEquityChartInstance = null;
let betaHeatmapChartInstance = null;
let activeReturnChartInstance = null;

const chartColors = ['#2962FF', '#00C853', '#FF3D00', '#FFD600', '#AA00FF', '#00B8D4', '#FF6D00', '#C51162'];

function renderEquityChart(dates, portfolioValues, benchValues) {
    const ctx = document.getElementById('equityChart').getContext('2d');
    if (equityChartInstance) equityChartInstance.destroy();
    equityChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: [
            { label: 'Danh mục', data: portfolioValues, borderColor: '#2962FF', backgroundColor: 'rgba(41, 98, 255, 0.1)', borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1 },
            { label: 'Thị trường (SPY)', data: benchValues, borderColor: '#8B96A5', borderWidth: 1.5, borderDash: [5, 5], pointRadius: 0, tension: 0.1 }
        ]},
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5', font: { family: 'Inter' } } } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5', maxTicksLimit: 10 } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderAllocChart(labels, data) {
    const ctx = document.getElementById('allocChart').getContext('2d');
    if (allocChartInstance) allocChartInstance.destroy();
    allocChartInstance = new Chart(ctx, { type: 'doughnut', data: { labels: labels, datasets: [{ data: data, backgroundColor: chartColors, borderWidth: 0, hoverOffset: 4 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: '70%', plugins: { legend: { position: 'right', labels: { color: '#E2E8F0', font: { family: 'Inter' } } } } } });
}

function renderRiskChart(labels, capitalData, riskData) {
    const ctx = document.getElementById('riskChart').getContext('2d');
    if (riskChartInstance) riskChartInstance.destroy();
    riskChartInstance = new Chart(ctx, { type: 'bar', data: { labels: labels, datasets: [
        { label: 'Trọng số vốn (%)', data: capitalData, backgroundColor: '#8B96A5', borderRadius: 4 },
        { label: 'Tỷ trọng rủi ro (%)', data: riskData, backgroundColor: '#2962FF', borderRadius: 4 }
    ]}, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5' } } }, scales: { x: { grid: { display: false }, ticks: { color: '#E2E8F0' } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } } });
}

function renderWFChart(insDates, insValues, oosDates, oosValues) {
    const ctx = document.getElementById('wfChart').getContext('2d');
    if (wfChartInstance) wfChartInstance.destroy();
    wfChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: insDates, datasets: [
            { label: 'In-sample', data: insValues, borderColor: '#FF1744', borderWidth: 1.5, pointRadius: 0 },
            { label: 'Out-of-sample', data: oosValues, borderColor: '#00E676', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(0, 230, 118, 0.1)' }
        ]},
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5' } } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderValidationChart(counts, edges, observedSharpe) {
    const ctx = document.getElementById('valChart').getContext('2d');
    if (validationChartInstance) validationChartInstance.destroy();
    const labels = edges.slice(0, -1).map((e, i) => e.toFixed(2));
    validationChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: 'Tần suất', data: counts, backgroundColor: '#2962FF', borderRadius: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderRegimeChart(dates, series) {
    const ctx = document.getElementById('regimeChart').getContext('2d');
    if (regimeChartInstance) regimeChartInstance.destroy();
    const datasets = Object.keys(series).map((key, i) => ({
        label: key, data: series[key], borderColor: chartColors[i], backgroundColor: chartColors[i] + '40', borderWidth: 1, pointRadius: 0, fill: true
    }));
    regimeChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: datasets },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5' } } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5', maxTicksLimit: 8 } }, y: { stacked: true, max: 1, grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderFactorChart(labels, betas) {
    const ctx = document.getElementById('factorChart').getContext('2d');
    if (factorChartInstance) factorChartInstance.destroy();
    factorChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: 'Beta', data: betas, backgroundColor: '#00E5FF', borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#E2E8F0' } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderScenarioChart(labels, impacts) {
    const ctx = document.getElementById('scenarioChart').getContext('2d');
    if (scenarioChartInstance) scenarioChartInstance.destroy();
    scenarioChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: 'Đóng góp (%)', data: impacts, backgroundColor: impacts.map(v => v < 0 ? '#FF1744' : '#00E676'), borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#E2E8F0' } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

// ================= MỚI THÊM =================
function renderPriceChart(dates, prices, ma20, ma50) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    if (priceChartInstance) priceChartInstance.destroy();
    priceChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: [
            { label: 'Giá', data: prices, borderColor: '#2962FF', borderWidth: 2, pointRadius: 0 },
            { label: 'MA20', data: ma20, borderColor: '#FFD600', borderWidth: 1.5, pointRadius: 0 },
            { label: 'MA50', data: ma50, borderColor: '#00C853', borderWidth: 1.5, pointRadius: 0 }
        ]},
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5' } } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5', maxTicksLimit: 8 } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderReturnsHistogram(counts, edges) {
    const ctx = document.getElementById('returnsHistogram').getContext('2d');
    if (histogramChartInstance) histogramChartInstance.destroy();
    const labels = edges.slice(0, -1).map(e => (e*100).toFixed(1) + '%');
    histogramChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels: labels, datasets: [{ label: 'Tần suất', data: counts, backgroundColor: '#2962FF', borderRadius: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderEfficientFrontier(scatterData) {
    const ctx = document.getElementById('efficientFrontierChart').getContext('2d');
    if (efficientFrontierChartInstance) efficientFrontierChartInstance.destroy();
    const datasets = scatterData.map(d => ({
        label: d.method,
        data: [{ x: d.vol * 100, y: d.ret * 100 }],
        backgroundColor: '#00E5FF',
        pointRadius: 8
    }));
    efficientFrontierChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: datasets },
        options: { 
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#8B96A5' } } },
            scales: {
                x: { title: { display: true, text: 'Volatility (%)', color: '#8B96A5' }, grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } },
                y: { title: { display: true, text: 'Expected Return (%)', color: '#8B96A5' }, grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } }
            }
        }
    });
}

function renderRegimeEquity(dates, switching, buyHold) {
    const ctx = document.getElementById('regimeEquityChart').getContext('2d');
    if (regimeEquityChartInstance) regimeEquityChartInstance.destroy();
    regimeEquityChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: [
            { label: 'Switching', data: switching, borderColor: '#00C853', borderWidth: 2, pointRadius: 0 },
            { label: 'Buy & Hold', data: buyHold, borderColor: '#8B96A5', borderWidth: 1.5, borderDash: [5, 5], pointRadius: 0 }
        ]},
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5' } } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderBetaHeatmap(dates, series) {
    const ctx = document.getElementById('betaHeatmapChart').getContext('2d');
    if (betaHeatmapChartInstance) betaHeatmapChartInstance.destroy();
    // Chart.js không hỗ trợ heatmap mượt, dùng Line chart nhiều màu thay thế
    const datasets = Object.keys(series).map((key, i) => ({
        label: key, data: series[key], borderColor: chartColors[i], borderWidth: 1.5, pointRadius: 0
    }));
    betaHeatmapChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: datasets },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#8B96A5' } } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5', maxTicksLimit: 8 } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}

function renderActiveReturn(dates, activeReturns) {
    const ctx = document.getElementById('activeReturnChart').getContext('2d');
    if (activeReturnChartInstance) activeReturnChartInstance.destroy();
    activeReturnChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: [
            { label: 'Active Return', data: activeReturns, borderColor: '#AA00FF', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(170, 0, 255, 0.1)' }
        ]},
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } }, y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } } }
    });
}
// Thêm vào cuối file web/js/charts.js
function renderActiveReturn(dates, activeReturns) {
    const canvas = document.getElementById('activeReturnChart');
    if (!canvas) return; // Nếu không tìm thấy thẻ canvas thì dừng
    const ctx = canvas.getContext('2d');
    if (activeReturnChartInstance) activeReturnChartInstance.destroy();
    
    activeReturnChartInstance = new Chart(ctx, {
        type: 'line',
        data: { 
            labels: dates, 
            datasets: [
                { 
                    label: 'Active Return', 
                    data: activeReturns, 
                    borderColor: '#AA00FF', 
                    borderWidth: 2, 
                    pointRadius: 0, 
                    fill: true, 
                    backgroundColor: 'rgba(170, 0, 255, 0.1)' 
                }
            ]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false, 
            plugins: { legend: { display: false } }, 
            scales: { 
                x: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } }, 
                y: { grid: { color: '#242C3D' }, ticks: { color: '#8B96A5' } } 
            } 
        }
    });
}

let quantRadarChartInstance = null;
function renderQuantamentalRadar(categories, seriesList) {
    const canvas = document.getElementById('quantRadarChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (quantRadarChartInstance) quantRadarChartInstance.destroy();

    const datasets = (seriesList || []).map(s => {
        const color = s.color || '#00E5FF';
        return {
            label: ${s.ticker} (),
            data: s.values,
            borderColor: color,
            backgroundColor: color.startsWith('#') ? color + '22' : 'rgba(0, 229, 255, 0.15)',
            borderWidth: 2,
            pointBackgroundColor: color,
            pointBorderColor: '#fff',
            pointRadius: 3
        };
    });

    quantRadarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: categories,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#E2E8F0', font: { family: 'Inter', size: 11 }, boxWidth: 12 }
                }
            },
            scales: {
                r: {
                    angleLines: { color: '#242C3D' },
                    grid: { color: '#242C3D' },
                    pointLabels: { color: '#00E5FF', font: { family: 'Inter', size: 12, weight: '600' } },
                    ticks: { color: '#8B96A5', backdropColor: 'transparent', min: 0, max: 1.0, stepSize: 0.2 }
                }
            }
        }
    });
}

let quantCurvesChartInstance = null;
function renderQuantamentalCurves(dates, quantValues, tradValues) {
    const canvas = document.getElementById('quantCurvesChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (quantCurvesChartInstance) quantCurvesChartInstance.destroy();

    quantCurvesChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: '🟢 Danh mục Quantamental AI (Đã lọc bẫy)',
                    data: quantValues,
                    borderColor: '#00CC96',
                    backgroundColor: 'rgba(0, 204, 150, 0.08)',
                    borderWidth: 2.2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1
                },
                {
                    label: '🔴 Danh mục Truyền thống (Chưa lọc bẫy)',
                    data: tradValues,
                    borderColor: '#EF553B',
                    borderWidth: 1.6,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#E2E8F0', font: { family: 'Inter', size: 12 } }
                }
            },
            scales: {
                x: {
                    grid: { color: '#242C3D' },
                    ticks: { color: '#8B96A5', maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: '#242C3D' },
                    ticks: { color: '#8B96A5' }
                }
            }
        }
    });
}
