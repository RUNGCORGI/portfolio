// web/js/api.js
const API_BASE = '';


async function fetchAPI(endpoint, method = 'POST', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Lỗi HTTP: ${response.status}`);
    }
    return await response.json();
}


const api = {
    getSystemInfo: () => fetchAPI('/api/system/info', 'GET'),
    getOverview: (p) => fetchAPI('/api/overview', 'POST', p),
    getAllocations: (p) => fetchAPI('/api/allocations', 'POST', p),
    getWalkForward: (p) => fetchAPI('/api/walkforward', 'POST', p),
    getValidation: (p) => fetchAPI('/api/validation', 'POST', p),
    getRegimes: (p) => fetchAPI('/api/regimes', 'POST', p),
    getFactors: (p) => fetchAPI('/api/factors', 'POST', p),
    getAlerts: (p) => fetchAPI('/api/alerts', 'POST', p),
   
    // API mới
    getScenarios: (p) => fetchAPI('/api/scenarios', 'POST', p),
    getQuantamental: (p) => fetchAPI('/api/quantamental', 'POST', p),
    getVNUniverse: () => fetchAPI('/api/market/vn/universe', 'GET'),
    getJournalTrades: () => fetchAPI('/api/journal/trades', 'GET'),
    addTrade: (p) => fetchAPI('/api/journal/trades', 'POST', p),
    getJournalPositions: (p) => fetchAPI('/api/journal/positions', 'POST', p)
};
