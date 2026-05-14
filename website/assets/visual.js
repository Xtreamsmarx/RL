/**
 * RL Visual Showcase - Interactive Visualizations
 * Comprehensive algorithm, environment, and training visualizations
 */

// ========================================
// CONFIGURATION
// ========================================

const CONFIG = {
    gridSize: 4,
    cellSize: 100,
    frozenLake: {
        start: 0,
        holes: [5, 7, 11, 12],
        goal: 15
    },
    colors: {
        start: '#4CAF50',
        frozen: '#F0F0F0',
        hole: '#FF6B6B',
        goal: '#2196F3',
        agent: '#FFD700',
        path: '#FF6B6B'
    }
};

// ========================================
// UTILITIES
// ========================================

function getStatePosition(state) {
    return {
        row: Math.floor(state / CONFIG.gridSize),
        col: state % CONFIG.gridSize
    };
}

function drawGrid(canvasId, policy = null) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / CONFIG.gridSize;

    // Clear
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid cells
    for (let i = 0; i < CONFIG.gridSize * CONFIG.gridSize; i++) {
        const pos = getStatePosition(i);
        const x = pos.col * cellSize;
        const y = pos.row * cellSize;

        // Fill cell
        if (i === CONFIG.frozenLake.start) {
            ctx.fillStyle = CONFIG.colors.start;
        } else if (CONFIG.frozenLake.holes.includes(i)) {
            ctx.fillStyle = CONFIG.colors.hole;
        } else if (i === CONFIG.frozenLake.goal) {
            ctx.fillStyle = CONFIG.colors.goal;
        } else {
            ctx.fillStyle = CONFIG.colors.frozen;
        }
        ctx.fillRect(x, y, cellSize, cellSize);

        // Draw grid lines
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, cellSize, cellSize);

        // Draw state number
        ctx.fillStyle = '#333';
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(i, x + cellSize - 5, y + 15);

        // Draw policy arrow if provided
        if (policy) {
            const action = policy[i];
            drawArrow(ctx, x + cellSize/2, y + cellSize/2, action, cellSize * 0.3);
        }
    }
}

function drawArrow(ctx, x, y, action, size) {
    ctx.strokeStyle = '#333';
    ctx.fillStyle = '#333';
    ctx.lineWidth = 2;

    const angles = {
        0: -Math.PI / 2,  // Up
        1: Math.PI / 2,   // Down
        2: Math.PI,        // Left
        3: 0              // Right
    };

    const angle = angles[action] || 0;
    const arrowLen = size;
    const headSize = size / 3;

    const endX = x + arrowLen * Math.cos(angle);
    const endY = y + arrowLen * Math.sin(angle);

    // Draw line
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw arrowhead
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - headSize * Math.cos(angle - Math.PI / 6), endY - headSize * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(endX - headSize * Math.cos(angle + Math.PI / 6), endY - headSize * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
}

function drawHeatmap(canvasId, values) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / CONFIG.gridSize;

    // Clear
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw heatmap
    for (let i = 0; i < CONFIG.gridSize * CONFIG.gridSize; i++) {
        const pos = getStatePosition(i);
        const x = pos.col * cellSize;
        const y = pos.row * cellSize;
        const value = Math.min(1.0, Math.max(0, values[i] || 0));

        // Interpolate color (dark blue to red)
        const hue = (1 - value) * 240; // 240 = blue, 0 = red
        ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
        ctx.fillRect(x, y, cellSize, cellSize);

        // Grid lines
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellSize, cellSize);

        // Value text
        ctx.fillStyle = value > 0.5 ? 'white' : 'black';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(value.toFixed(2), x + cellSize/2, y + cellSize/2);
    }
}

// ========================================
// POLICY GENERATION (Simulated)
// ========================================

function generatePolicy(algorithmType) {
    // Simulate learned policies
    const policies = {
        'ql': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3, 0],
        'sarsa': [0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 0, 3, 0],
        'td': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3, 0],
        'pi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3, 0]
    };
    return policies[algorithmType] || policies.ql;
}

function generateValues(algorithmType) {
    // Simulate learned value functions
    const values = {
        'ql': [0.9, 0.8, 0.7, 0.6, 0.85, 0, 0.75, 0, 0.92, 0.95, 0.98, 0, 0, 0, 0.99, 1.0],
        'sarsa': [0.8, 0.7, 0.6, 0.5, 0.75, 0, 0.65, 0, 0.82, 0.85, 0.88, 0, 0, 0, 0.89, 1.0],
        'td': [0.88, 0.78, 0.68, 0.58, 0.83, 0, 0.73, 0, 0.9, 0.93, 0.96, 0, 0, 0, 0.97, 1.0],
        'vi': [1.0, 0.99, 0.98, 0.97, 0.98, 0, 0.97, 0, 0.99, 1.0, 1.0, 0, 0, 0, 1.0, 1.0]
    };
    return values[algorithmType] || values.ql;
}

// ========================================
// INITIALIZATION
// ========================================

function initializeVisualizations() {
    // Environment grid
    drawGrid('gridCanvas');

    // Policies
    drawGrid('policyQL', generatePolicy('ql'));
    drawGrid('policySARSA', generatePolicy('sarsa'));
    drawGrid('policyTD', generatePolicy('td'));
    drawGrid('policyPI', generatePolicy('pi'));

    // Heatmaps
    drawHeatmap('heatmapQL', generateValues('ql'));
    drawHeatmap('heatmapSARSA', generateValues('sarsa'));
    drawHeatmap('heatmapTD', generateValues('td'));
    drawHeatmap('heatmapVI', generateValues('vi'));

    // Charts
    initializeCharts();

    // Comparison table is already in HTML
}

// ========================================
// CHARTS
// ========================================

let convergenceChart = null;
let returnsChart = null;
let lossChart = null;
let epsilonChart = null;

function initializeCharts() {
    // Convergence Comparison
    const convCtx = document.getElementById('convergenceChart');
    if (convCtx && !convergenceChart) {
        convergenceChart = new Chart(convCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 100}, (_, i) => (i+1)*200),
                datasets: [
                    {
                        label: 'Q-Learning',
                        data: generateConvergenceData('ql'),
                        borderColor: '#FF6B6B',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 0
                    },
                    {
                        label: 'SARSA',
                        data: generateConvergenceData('sarsa'),
                        borderColor: '#4CAF50',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 0
                    },
                    {
                        label: 'TD(λ)',
                        data: generateConvergenceData('td'),
                        borderColor: '#2196F3',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 0
                    },
                    {
                        label: 'Value Iteration',
                        data: generateConvergenceData('vi'),
                        borderColor: '#9C27B0',
                        tension: 0.4,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'top' }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Mean Episode Return' },
                        max: 1.0
                    },
                    x: {
                        title: { display: true, text: 'Episodes' }
                    }
                }
            }
        });
    }

    // Returns Chart
    const retCtx = document.getElementById('returnsChart');
    if (retCtx && !returnsChart) {
        returnsChart = new Chart(retCtx, {
            type: 'bar',
            data: {
                labels: ['QL', 'SARSA', 'TD(λ)', 'MC', 'Policy Iter', 'Value Iter'],
                datasets: [{
                    label: 'Mean Return',
                    data: [0.88, 0.80, 0.85, 0.75, 0.99, 0.98],
                    backgroundColor: ['#FF6B6B', '#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#00BCD4']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { max: 1.0 } }
            }
        });
    }

    // Loss Chart
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx && !lossChart) {
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 50}, (_, i) => (i+1)*30),
                datasets: [{
                    label: 'DQN Loss',
                    data: generateLossData(),
                    borderColor: '#FF6B6B',
                    fill: true,
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { title: { display: true, text: 'MSE Loss' } }
                }
            }
        });
    }

    // Epsilon Decay Chart
    const epsCtx = document.getElementById('epsilonChart');
    if (epsCtx && !epsilonChart) {
        epsilonChart = new Chart(epsCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 100}, (_, i) => (i+1)*150),
                datasets: [{
                    label: 'ε (Exploration Rate)',
                    data: generateEpsilonData(),
                    borderColor: '#2196F3',
                    fill: true,
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 1.0,
                        title: { display: true, text: 'Epsilon' }
                    }
                }
            }
        });
    }
}

function generateConvergenceData(type) {
    const data = [];
    for (let i = 0; i < 100; i++) {
        const progress = i / 100;
        let value;
        switch(type) {
            case 'ql': value = 0.88 * (1 - Math.exp(-5 * progress)); break;
            case 'sarsa': value = 0.80 * (1 - Math.exp(-4 * progress)); break;
            case 'td': value = 0.85 * (1 - Math.exp(-4.5 * progress)); break;
            case 'vi': value = 0.99; break;
            default: value = 0;
        }
        data.push(value + (Math.random() - 0.5) * 0.05);
    }
    return data;
}

function generateLossData() {
    const data = [];
    for (let i = 0; i < 50; i++) {
        const decay = Math.exp(-i / 20);
        data.push(0.5 * decay + Math.random() * 0.01);
    }
    return data;
}

function generateEpsilonData() {
    const data = [];
    for (let i = 0; i < 100; i++) {
        data.push(Math.max(0.05, 1.0 - (i / 100) * 0.95));
    }
    return data;
}

// ========================================
// AGENT SIMULATION
// ========================================

function runSimulation(algorithmType) {
    const policy = generatePolicy(algorithmType);
    simulateAgent(policy);

    // Update active button
    document.querySelectorAll('.sim-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.includes(algorithmType === 'ql' ? 'Q-Learning' : 
            algorithmType === 'sarsa' ? 'SARSA' : 
            algorithmType === 'td' ? 'TD(λ)' : 
            algorithmType === 'pi' ? 'Policy' : '')) {
            btn.classList.add('active');
        }
    });
    
    // Also handle event if it exists
    if (event && event.target) {
        event.target.classList.add('active');
    }
}

function simulateAgent(policy) {
    const canvas = document.getElementById('simCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / CONFIG.gridSize;

    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid('simCanvas', policy);

    // Simulate agent movement
    let state = CONFIG.frozenLake.start;
    let episode = 0;
    let stepCount = 0;
    let totalReturn = 0;
    let successCount = 0;

    const simulate = () => {
        // Draw episode
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawGrid('simCanvas', policy);

        // Animate agent movement
        let currentState = CONFIG.frozenLake.start;
        let steps = 0;
        const maxSteps = 200;

        const moveAgent = () => {
            const action = policy[currentState];
            const newState = getNextState(currentState, action);

            // Draw agent
            const pos = getStatePosition(currentState);
            const x = pos.col * cellSize + cellSize / 2;
            const y = pos.row * cellSize + cellSize / 2;

            ctx.fillStyle = CONFIG.colors.agent;
            ctx.beginPath();
            ctx.arc(x, y, cellSize * 0.3, 0, 2 * Math.PI);
            ctx.fill();

            if (newState === CONFIG.frozenLake.goal || CONFIG.frozenLake.holes.includes(newState) || steps >= maxSteps) {
                // Episode done
                if (newState === CONFIG.frozenLake.goal) {
                    totalReturn += 1;
                    successCount += 1;
                }
                episode++;
                stepCount += steps;

                // Update stats
                document.getElementById('simEpisode').textContent = episode;
                document.getElementById('simReturn').textContent = (totalReturn / Math.max(1, episode)).toFixed(2);
                document.getElementById('simSteps').textContent = stepCount;
                document.getElementById('simSuccess').textContent = ((successCount / Math.max(1, episode)) * 100).toFixed(1) + '%';

                // Draw trajectory
                drawTrajectory(currentState, policy);

                if (episode < 10) {
                    setTimeout(simulate, 500);
                }
            } else {
                currentState = newState;
                steps++;
                setTimeout(moveAgent, 100);
            }
        };

        moveAgent();
    };

    simulate();
}

function getNextState(state, action) {
    const pos = getStatePosition(state);
    let newRow = pos.row;
    let newCol = pos.col;

    if (action === 0 && newRow > 0) newRow--; // Up
    else if (action === 1 && newRow < CONFIG.gridSize - 1) newRow++; // Down
    else if (action === 2 && newCol > 0) newCol--; // Left
    else if (action === 3 && newCol < CONFIG.gridSize - 1) newCol++; // Right

    return newRow * CONFIG.gridSize + newCol;
}

function drawTrajectory(finalState, policy) {
    const canvas = document.getElementById('trajCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / CONFIG.gridSize;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid('trajCanvas', policy);

    // Draw path from start to current
    let state = CONFIG.frozenLake.start;
    ctx.strokeStyle = CONFIG.colors.path;
    ctx.lineWidth = 3;
    ctx.beginPath();

    const startPos = getStatePosition(state);
    ctx.moveTo(startPos.col * cellSize + cellSize / 2, startPos.row * cellSize + cellSize / 2);

    for (let step = 0; step < 50; step++) {
        const action = policy[state];
        state = getNextState(state, action);

        const pos = getStatePosition(state);
        ctx.lineTo(pos.col * cellSize + cellSize / 2, pos.row * cellSize + cellSize / 2);

        if (state === CONFIG.frozenLake.goal || CONFIG.frozenLake.holes.includes(state)) break;
    }
    ctx.stroke();
}

// ========================================
// NAVIGATION
// ========================================

function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

    // Show selected tab
    const activeTab = document.getElementById(tabId);
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // Update sidebar
    document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
    event.target.parentElement.classList.add('active');
}

// ========================================
// INITIALIZATION ON PAGE LOAD
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    initializeVisualizations();

    // Set first tab as active
    const firstTab = document.querySelector('.tab-content');
    if (firstTab) {
        firstTab.classList.add('active');
    }

    // Set first nav item as active
    const firstNavItem = document.querySelector('.sidebar-nav li');
    if (firstNavItem) {
        firstNavItem.classList.add('active');
    }

    // Initialize first simulation
    setTimeout(() => runSimulation('ql'), 500);
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        drawGrid,
        drawHeatmap,
        generatePolicy,
        generateValues,
        runSimulation
    };
}
