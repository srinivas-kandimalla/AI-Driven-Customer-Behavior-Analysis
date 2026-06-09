/* Dashboard Logic: AI-Driven Customer Behavior Analysis Platform */

// Global Chart references to allow clean redraws on refresh
let chartSegmentsInstance = null;
let chartRevenueInstance = null;
let chartAovInstance = null;

// Base API URI (uses relative paths if served together, falls back to port 5000)
const API_BASE = '/api';
// Icon mapping for product categories
const categoryIcons = {
    'Electronics': 'fa-laptop-code',
    'Fashion': 'fa-shirt',
    'Food': 'fa-utensils',
    'Sports': 'fa-dumbbell',
    'Beauty': 'fa-sparkles'
};

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Tabs Navigation
    initTabs();
    
    // 2. Load Dashboard Statistics and Churn Risks
    loadDashboardData();
    
    // 3. Load Customer Segments Table
    loadCustomerSegments();
    
    // 4. Attach Event Listeners
    document.getElementById('btn-refresh').addEventListener('click', refreshAllData);
    document.getElementById('purchase-form').addEventListener('submit', handlePurchasePrediction);
    document.getElementById('btn-get-recs').addEventListener('click', handleGetRecommendations);
    document.getElementById('segment-search').addEventListener('input', filterCustomersTable);
});

/* Tabs Switching System */
function initTabs() {
    const sidebarItems = document.querySelectorAll('.sidebar-item');
    const tabContents = document.querySelectorAll('.tab-content');
    
    sidebarItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all items
            sidebarItems.forEach(i => i.classList.remove('active'));
            
            // Add active class to clicked item
            item.classList.add('active');
            
            // Get target tab id
            const targetTab = item.getAttribute('data-tab');
            
            // Show/Hide tabs
            tabContents.forEach(tab => {
                if (tab.id === `tab-${targetTab}`) {
                    tab.style.display = 'block';
                } else {
                    tab.style.display = 'none';
                }
            });
            
            // Update Page Header Info
            updateHeaderTitle(targetTab);
        });
    });
}

function updateHeaderTitle(tab) {
    const titleEl = document.getElementById('page-title');
    const descEl = document.getElementById('page-desc');
    
    if (tab === 'dashboard') {
        titleEl.textContent = 'Dashboard Overview';
        descEl.textContent = 'Real-time behavior analysis, segmentation, and machine learning insights.';
    } else if (tab === 'segments') {
        titleEl.textContent = 'Customer Segments Analysis';
        descEl.textContent = 'Detailed demographics and behaviors grouped by K-Means designations.';
    } else if (tab === 'predictions') {
        titleEl.textContent = 'Propensity & Recommendations Center';
        descEl.textContent = 'Interact with Random Forest propensity models and SVD collaborative filters.';
    }
}

/* Refresh all data */
function refreshAllData() {
    const refreshBtn = document.getElementById('btn-refresh');
    refreshBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Refreshing...';
    refreshBtn.disabled = true;
    
    Promise.all([
        loadDashboardData(),
        loadCustomerSegments()
    ]).finally(() => {
        refreshBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Refresh Data';
        refreshBtn.disabled = false;
    });
}

/* Load KPI stats & charts */
async function loadDashboardData() {
    try {
        const response = await fetch(`${API_BASE}/dashboard/stats`);
        const result = await response.json();
        
        if (result.status === 'success') {
            const stats = result.data;
            
            // Update KPIs
            document.getElementById('kpi-total-customers').textContent = stats.total_customers;
            document.getElementById('kpi-churn-risk').textContent = `${stats.churn_risk_percent}%`;
            document.getElementById('kpi-spending-score').textContent = stats.avg_spending_score;
            document.getElementById('kpi-top-segment').textContent = stats.top_segment;
            
            // Render Charts
            renderSegmentsChart(stats.segment_breakdown);
            renderRevenueChart(stats.monthly_revenue);
            renderAovChart(stats.aov_by_segment);
            populateSegmentSummaryTable(stats.segment_breakdown);
            
            // Load Top Churn risk table after KPIs load
            await loadChurnRiskTable();
        } else {
            console.error('Error fetching dashboard stats:', result.message);
        }
    } catch (err) {
        console.error('Network error fetching dashboard stats:', err);
    }
}

/* Render Segment Doughnut Chart */
function renderSegmentsChart(breakdown) {
    const ctx = document.getElementById('chart-segments').getContext('2d');
    
    const labels = Object.keys(breakdown);
    const counts = labels.map(label => breakdown[label].count);
    
    // Custom harmonious color palette
    const colors = {
        'Premium': '#818cf8', // Indigo
        'Loyal': '#10b981',   // Emerald
        'New': '#06b6d4',     // Cyan
        'At-Risk': '#f59e0b', // Amber/Orange
        'Dormant': '#ef4444'  // Red
    };
    
    const bgColors = labels.map(label => colors[label] || '#64748b');
    
    if (chartSegmentsInstance) {
        chartSegmentsInstance.destroy();
    }
    
    chartSegmentsInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: counts,
                backgroundColor: bgColors,
                borderWidth: 2,
                borderColor: '#111625',
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#94a3b8',
                        font: { family: 'Inter', size: 12 },
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const val = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${label}: ${val} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

/* Render Revenue Trend Bar Chart */
function renderRevenueChart(revenueData) {
    const ctx = document.getElementById('chart-revenue').getContext('2d');
    
    const labels = revenueData.map(item => item.month);
    const revenues = revenueData.map(item => item.revenue);
    
    if (chartRevenueInstance) {
        chartRevenueInstance.destroy();
    }
    
    // Create a beautiful linear gradient for the bars
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, '#6366f1'); // Primary Indigo
    gradient.addColorStop(1, '#06b6d4'); // Secondary Cyan
    
    chartRevenueInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Monthly Revenue ($)',
                data: revenues,
                backgroundColor: gradient,
                borderRadius: 6,
                borderWidth: 0,
                barPercentage: 0.6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` Revenue: $${context.raw.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#64748b', font: { family: 'Inter' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { 
                        color: '#64748b', 
                        font: { family: 'Inter' },
                        callback: function(value) {
                            return '$' + (value >= 1000 ? (value / 1000) + 'k' : value);
                        }
                    }
                }
            }
        }
    });
}

/* Render Average Order Value Bar Chart */
function renderAovChart(aovData) {
    const ctx = document.getElementById('chart-aov').getContext('2d');
    
    const labels = Object.keys(aovData);
    const values = labels.map(label => aovData[label]);
    
    if (chartAovInstance) {
        chartAovInstance.destroy();
    }
    
    const colors = {
        'Premium': '#818cf8',
        'Loyal': '#10b981',
        'New': '#06b6d4',
        'At-Risk': '#f59e0b',
        'Dormant': '#ef4444'
    };
    const bgColors = labels.map(label => colors[label] || '#64748b');
    
    chartAovInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Average Order Value ($)',
                data: values,
                backgroundColor: bgColors,
                borderRadius: 6,
                borderWidth: 0,
                barPercentage: 0.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#64748b', font: { family: 'Inter' } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { 
                        color: '#64748b', 
                        font: { family: 'Inter' },
                        callback: value => `$${value}`
                    }
                }
            }
        }
    });
}

/* Populate Segment Summary Table */
function populateSegmentSummaryTable(breakdown) {
    const summaryBody = document.getElementById('segment-summary-body');
    summaryBody.innerHTML = '';
    
    const colors = {
        'Premium': 'badge-success',
        'Loyal': 'badge-info',
        'New': 'badge-success', // custom
        'At-Risk': 'badge-warning',
        'Dormant': 'badge-danger'
    };
    
    Object.keys(breakdown).forEach(segmentName => {
        const info = breakdown[segmentName];
        const badgeClass = colors[segmentName] || 'badge-info';
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span class="badge ${badgeClass}">${segmentName}</span></td>
            <td>${info.count}</td>
            <td><strong>${info.percentage}%</strong></td>
        `;
        summaryBody.appendChild(tr);
    });
}

/* Load Top 20 Churn Risks Table */
async function loadChurnRiskTable() {
    const tableBody = document.getElementById('churn-table-body');
    tableBody.innerHTML = `
        <tr>
            <td colspan="7" style="text-align: center;">
                <i class="fa-solid fa-spinner fa-spin"></i> Refreshing table...
            </td>
        </tr>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/churn-risk`);
        const result = await response.json();
        
        if (result.status === 'success') {
            tableBody.innerHTML = '';
            
            const customers = result.data;
            customers.forEach(cust => {
                const tr = document.createElement('tr');
                
                // Colorize risk percentage
                const riskPercent = Math.round(cust.churn_probability * 100);
                let riskClass = 'badge-success';
                if (riskPercent >= 75) {
                    riskClass = 'badge-danger';
                } else if (riskPercent >= 50) {
                    riskClass = 'badge-warning';
                }
                
                tr.innerHTML = `
                    <td><strong>${cust.customer_id}</strong></td>
                    <td>${cust.age}</td>
                    <td><span class="badge badge-info">${cust.segment}</span></td>
                    <td>${cust.last_purchase_days} days ago</td>
                    <td>$${cust.average_order_value}</td>
                    <td><span class="badge ${riskClass}">${riskPercent}%</span></td>
                    <td>
                        <button class="btn btn-outline-cyan btn-sm" onclick="quickRecommend('${cust.customer_id}')">
                            <i class="fa-solid fa-wand-magic-sparkles"></i> Recommend
                        </button>
                    </td>
                `;
                tableBody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error('Error fetching churn risk table:', err);
        tableBody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; color: var(--danger);">
                    Failed to load churn risk records. Check API connection.
                </td>
            </tr>
        `;
    }
}

/* Quick recommendation action from table */
function quickRecommend(customerId) {
    // Find predict tab and click it
    const predictTabItem = document.querySelector('[data-tab="predictions"]');
    if (predictTabItem) {
        predictTabItem.click();
        
        // Fill input
        const inputEl = document.getElementById('recs-customer-id');
        inputEl.value = customerId;
        
        // Trigger recommendation query
        handleGetRecommendations();
    }
}

/* Load Customer Segments Tab List */
async function loadCustomerSegments() {
    const tableBody = document.getElementById('all-customers-body');
    try {
        const response = await fetch(`${API_BASE}/segments`);
        const result = await response.json();
        
        if (result.status === 'success') {
            tableBody.innerHTML = '';
            const customers = result.data;
            
            // Render rows
            customers.forEach(cust => {
                const tr = document.createElement('tr');
                tr.setAttribute('data-id', cust.customer_id.toLowerCase());
                
                let segmentClass = 'badge-info';
                if (cust.segment === 'Premium') segmentClass = 'badge-success';
                else if (cust.segment === 'Loyal') segmentClass = 'badge-success';
                else if (cust.segment === 'At-Risk') segmentClass = 'badge-warning';
                else if (cust.segment === 'Dormant') segmentClass = 'badge-danger';
                
                tr.innerHTML = `
                    <td><strong>${cust.customer_id}</strong></td>
                    <td>$${cust.annual_income.toLocaleString()}</td>
                    <td>${cust.spending_score}</td>
                    <td>${cust.purchase_frequency} purchases/yr</td>
                    <td><span class="badge ${segmentClass}">${cust.segment}</span></td>
                `;
                tableBody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error('Error loading full customer segments list:', err);
    }
}

/* Search filter in Customer Segments list */
function filterCustomersTable() {
    const query = document.getElementById('segment-search').value.toLowerCase().trim();
    const rows = document.querySelectorAll('#all-customers-table tbody tr');
    
    rows.forEach(row => {
        const id = row.getAttribute('data-id');
        if (!id) return;
        
        if (id.includes(query)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

/* Handle Purchase Prediction Form Submission */
async function handlePurchasePrediction(e) {
    e.preventDefault();
    
    const outputEl = document.getElementById('purchase-output');
    outputEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing model prediction...';
    outputEl.className = 'prediction-result visible';
    
    // Retrieve values
    const age = parseFloat(document.getElementById('pred-age').value);
    const annual_income = parseFloat(document.getElementById('pred-income').value);
    const spending_score = parseFloat(document.getElementById('pred-score').value);
    const loyalty_years = parseFloat(document.getElementById('pred-loyalty').value);
    const average_order_value = parseFloat(document.getElementById('pred-aov').value);
    
    const payload = { age, annual_income, spending_score, loyalty_years, average_order_value };
    
    try {
        const response = await fetch(`${API_BASE}/predict/purchase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            const data = result.data;
            const willPurchase = data.will_purchase === 1;
            const probPercent = Math.round(data.purchase_probability * 100);
            
            let resultClass = willPurchase ? 'success' : 'danger';
            let icon = willPurchase ? 'fa-circle-check' : 'fa-circle-xmark';
            let color = willPurchase ? 'var(--success)' : 'var(--danger)';
            let titleText = willPurchase ? 'Will Purchase' : 'Unlikely to Purchase';
            
            outputEl.innerHTML = `
                <div style="color: ${color}; font-size: 40px; margin-bottom: 12px;">
                    <i class="fa-solid ${icon}"></i>
                </div>
                <div class="prediction-val" style="color: ${color};">${titleText}</div>
                <div class="prediction-probability">
                    Probability: <strong>${probPercent}%</strong> (${data.confidence_label})
                </div>
            `;
            
            // Add gradient border based on result
            outputEl.style.border = `1px solid ${color}`;
            outputEl.style.background = willPurchase 
                ? 'rgba(16, 185, 129, 0.04)' 
                : 'rgba(239, 68, 68, 0.04)';
        } else {
            outputEl.innerHTML = `<span style="color: var(--danger);"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${result.message}</span>`;
        }
    } catch (err) {
        console.error('Purchase Prediction request failed:', err);
        outputEl.innerHTML = `<span style="color: var(--danger);"><i class="fa-solid fa-triangle-exclamation"></i> Connection Failed. Ensure API is running.</span>`;
    }
}

/* Handle Recommendation Query */
async function handleGetRecommendations() {
    const inputEl = document.getElementById('recs-customer-id');
    const container = document.getElementById('recs-container');
    const titleEl = document.getElementById('recs-title');
    const targetIdSpan = document.getElementById('recs-target-id');
    
    const customerId = inputEl.value.toUpperCase().trim();
    if (!customerId) {
        alert("Please enter a valid customer ID.");
        return;
    }
    
    container.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 40px 0;">
            <i class="fa-solid fa-spinner fa-spin" style="font-size: 32px; color: var(--secondary);"></i>
            <p style="margin-top: 12px; color: var(--text-secondary);">Querying collaborative filtering SVD models...</p>
        </div>
    `;
    titleEl.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/recommend/${customerId}`);
        const result = await response.json();
        
        if (result.status === 'success') {
            const data = result.data;
            
            // Update title
            targetIdSpan.textContent = data.customer_id;
            if (!data.customer_exists) {
                targetIdSpan.textContent += ' (New Customer / Cold Start)';
            }
            titleEl.style.display = 'block';
            
            // Empty container
            container.innerHTML = '';
            
            // Populate cards
            data.recommendations.forEach((cat, index) => {
                const card = document.createElement('div');
                card.className = 'rec-card';
                
                const iconClass = categoryIcons[cat] || 'fa-box';
                
                card.innerHTML = `
                    <div class="rec-rank">Rank #${index + 1}</div>
                    <div class="rec-icon"><i class="fa-solid ${iconClass}"></i></div>
                    <div class="rec-name">${cat}</div>
                `;
                
                container.appendChild(card);
            });
        } else {
            container.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; color: var(--danger); padding: 40px 0;">
                    <i class="fa-solid fa-triangle-exclamation" style="font-size: 40px; margin-bottom: 12px;"></i>
                    <p>Error: ${result.message}</p>
                </div>
            `;
        }
    } catch (err) {
        console.error('Recommendation query failed:', err);
        container.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; color: var(--danger); padding: 40px 0;">
                <i class="fa-solid fa-triangle-exclamation" style="font-size: 40px; margin-bottom: 12px;"></i>
                <p>Connection failed. Make sure Flask API is online.</p>
            </div>
        `;
    }
}
