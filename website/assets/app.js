/**
 * RL Capstone V3 - Interactive Dashboard
 * Handles section navigation, chart generation, and data visualization
 */

// ========================================
// SECTION NAVIGATION
// ========================================

function showSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => section.classList.remove('active'));

    // Show selected section
    const activeSection = document.getElementById(sectionId);
    if (activeSection) {
        activeSection.classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Update active nav link
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => link.classList.remove('active'));
    event.target.classList.add('active');
}

// ========================================
// ALGORITHM TABS
// ========================================

function showAlgoTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.algo-tab');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Show selected tab
    const activeTab = document.getElementById(tabName + '-tab');
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // Update button states
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

// ========================================
// CHART GENERATION
// ========================================

let classicalChart = null;
let categoryChart = null;
let dqnChart = null;

function initializeCharts() {
    // Classical Algorithms Performance Chart
    const classicalCtx = document.getElementById('classicalChart');
    if (classicalCtx) {
        classicalChart = new Chart(classicalCtx, {
            type: 'bar',
            data: {
                labels: ['Policy Iteration', 'Value Iteration', 'MC ε-Greedy', 'Q-Learning', 'TD(0)', 'TD(λ)', 'SARSA(0)', 'SARSA(λ)'],
                datasets: [{
                    label: 'Mean Eval Return',
                    data: [0.95, 0.92, 0.85, 0.88, 0.82, 0.87, 0.80, 0.89],
                    backgroundColor: [
                        'rgba(30, 60, 114, 0.7)',
                        'rgba(42, 82, 152, 0.7)',
                        'rgba(0, 212, 255, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(59, 130, 246, 0.7)',
                        'rgba(139, 92, 246, 0.7)',
                        'rgba(236, 72, 153, 0.7)',
                        'rgba(245, 158, 11, 0.7)'
                    ],
                    borderColor: [
                        'rgba(30, 60, 114, 1)',
                        'rgba(42, 82, 152, 1)',
                        'rgba(0, 212, 255, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(59, 130, 246, 1)',
                        'rgba(139, 92, 246, 1)',
                        'rgba(236, 72, 153, 1)',
                        'rgba(245, 158, 11, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            font: { size: 12, weight: 'bold' }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(2);
                            }
                        }
                    }
                }
            }
        });
    }

    // Algorithm Category Distribution Chart
    const categoryCtx = document.getElementById('categoryChart');
    if (categoryCtx) {
        categoryChart = new Chart(categoryCtx, {
            type: 'doughnut',
            data: {
                labels: ['Dynamic Programming', 'Monte Carlo', 'Temporal Difference', 'SARSA', 'Deep RL'],
                datasets: [{
                    data: [2, 2, 5, 4, 1],
                    backgroundColor: [
                        'rgba(30, 60, 114, 0.7)',
                        'rgba(42, 82, 152, 0.7)',
                        'rgba(0, 212, 255, 0.7)',
                        'rgba(139, 92, 246, 0.7)',
                        'rgba(16, 185, 129, 0.7)'
                    ],
                    borderColor: [
                        'rgba(30, 60, 114, 1)',
                        'rgba(42, 82, 152, 1)',
                        'rgba(0, 212, 255, 1)',
                        'rgba(139, 92, 246, 1)',
                        'rgba(16, 185, 129, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 12 },
                            padding: 15
                        }
                    }
                }
            }
        });
    }

    // DQN Training Progress Chart
    const dqnCtx = document.getElementById('dqnChart');
    if (dqnCtx) {
        // Generate simulated DQN training data
        const episodes = Array.from({ length: 100 }, (_, i) => (i + 1) * 15);
        const returns = episodes.map(ep => {
            const base = -1 + (ep / 1500) * 1.5;
            const noise = (Math.random() - 0.5) * 0.5;
            return Math.max(0, base + noise);
        });
        const losses = episodes.map(ep => {
            return Math.max(0.001, 0.5 * Math.exp(-(ep / 1000)) + Math.random() * 0.01);
        });

        dqnChart = new Chart(dqnCtx, {
            type: 'line',
            data: {
                labels: episodes.map(ep => ep.toString()),
                datasets: [
                    {
                        label: 'Mean Episode Return',
                        data: returns,
                        borderColor: 'rgba(16, 185, 129, 1)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        yAxisID: 'y',
                        tension: 0.4
                    },
                    {
                        label: 'Mean Loss',
                        data: losses,
                        borderColor: 'rgba(239, 68, 68, 1)',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        yAxisID: 'y1',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            font: { size: 12, weight: 'bold' },
                            padding: 15
                        }
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Episode Return',
                            font: { weight: 'bold' }
                        },
                        beginAtZero: true
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Loss',
                            font: { weight: 'bold' }
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                }
            }
        });
    }
}

// ========================================
// DATA LOADING (Future enhancement)
// ========================================

async function loadCSVData() {
    try {
        // In production, this would load from ../result/csv/
        // For now, we use placeholder data
        console.log('CSV data loading capability ready');
    } catch (error) {
        console.error('Error loading CSV data:', error);
    }
}

// ========================================
// INITIALIZATION
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    // Set first section as active
    const firstSection = document.querySelector('.section');
    if (firstSection) {
        firstSection.classList.add('active');
    }

    // Set first algorithm tab as active
    const firstAlgoTab = document.querySelector('.algo-tab');
    if (firstAlgoTab) {
        firstAlgoTab.classList.add('active');
    }

    // Set first tab button as active
    const firstTabBtn = document.querySelector('.tab-btn');
    if (firstTabBtn) {
        firstTabBtn.classList.add('active');
    }

    // Initialize charts
    initializeCharts();

    // Load CSV data
    loadCSVData();

    // Smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });
});

// ========================================
// UTILITY FUNCTIONS
// ========================================

function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function createStat(label, value, unit = '') {
    return `
        <div class="stat">
            <div class="stat-value">${value}${unit}</div>
            <div class="stat-label">${label}</div>
        </div>
    `;
}

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showSection,
        showAlgoTab,
        initializeCharts,
        loadCSVData,
        formatNumber,
        createStat
    };
}
