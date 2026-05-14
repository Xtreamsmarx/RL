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
        // Action mapping: 0=up, 1=down, 2=left, 3=right
        // Policies below intentionally encode safe routes to goal state 15.
        'ql':    [1, 3, 1, 2, 1, 0, 1, 0, 3, 3, 1, 0, 0, 3, 3, 0],
        'sarsa': [1, 2, 1, 2, 1, 0, 1, 0, 3, 3, 1, 0, 0, 3, 3, 0],
        'td':    [1, 3, 1, 2, 1, 0, 1, 0, 3, 3, 1, 0, 0, 3, 3, 0],
        'pi':    [1, 3, 1, 2, 1, 0, 1, 0, 3, 3, 1, 0, 0, 3, 3, 0]
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
    renderEnvironmentCanvas();
    renderPolicyCanvases();
    renderValueCanvases();

    // Charts
    initializeCharts();

    // Comparison table is already in HTML
}

function renderEnvironmentCanvas() {
    drawGrid('gridCanvas');
}

function renderPolicyCanvases() {
    drawGrid('policyQL', generatePolicy('ql'));
    drawGrid('policySARSA', generatePolicy('sarsa'));
    drawGrid('policyTD', generatePolicy('td'));
    drawGrid('policyPI', generatePolicy('pi'));
}

function renderValueCanvases() {
    drawHeatmap('heatmapQL', generateValues('ql'));
    drawHeatmap('heatmapSARSA', generateValues('sarsa'));
    drawHeatmap('heatmapTD', generateValues('td'));
    drawHeatmap('heatmapVI', generateValues('vi'));
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

const SIM_STATE = {
    algorithm: 'ql',
    policy: [],
    episodes: [],
    currentEpisodeIndex: 0,
    currentFrame: 0,
    speed: 1,
    isPlaying: false,
    timerId: null,
    totalEpisodes: 150,
    maxEpisodeSteps: 28,
    chart: null
};

function runSimulation(algorithmType, triggerElement = null) {
    SIM_STATE.algorithm = algorithmType;
    SIM_STATE.policy = generatePolicy(algorithmType);
    SIM_STATE.episodes = generateEpisodeBatch(SIM_STATE.policy, SIM_STATE.totalEpisodes, algorithmType);
    SIM_STATE.currentEpisodeIndex = 0;
    SIM_STATE.currentFrame = 0;
    SIM_STATE.isPlaying = false;
    stopPlaybackTimer();

    updateAlgorithmButtons(triggerElement, algorithmType);
    renderCurrentEpisodeFrame();
    updateSimulationStats();
    updateEpisodeNarrative();
    updateTimelineChart();
    updatePlaybackButton();
}

function generateEpisodeBatch(policy, count, algorithmType) {
    const profiles = {
        ql: { initialEps: 0.22, decay: 0.987 },
        sarsa: { initialEps: 0.18, decay: 0.989 },
        td: { initialEps: 0.20, decay: 0.988 },
        pi: { initialEps: 0.04, decay: 0.996 }
    };
    const profile = profiles[algorithmType] || profiles.ql;
    const episodes = [];

    for (let ep = 0; ep < count; ep++) {
        const epsRate = Math.max(0.01, profile.initialEps * Math.pow(profile.decay, ep));
        episodes.push(generateSingleEpisode(policy, epsRate, ep));
    }

    return episodes;
}

function generateSingleEpisode(policy, explorationRate, episodeIndex) {
    let currentState = CONFIG.frozenLake.start;
    const path = [currentState];
    const decisions = [];
    let success = false;
    let terminated = false;

    for (let step = 0; step < SIM_STATE.maxEpisodeSteps; step++) {
        const greedyAction = policy[currentState] ?? 0;
        const explore = Math.random() < explorationRate;
        const action = explore ? Math.floor(Math.random() * 4) : greedyAction;
        const nextState = getNextState(currentState, action);

        decisions.push({
            step: step + 1,
            state: currentState,
            action,
            greedyAction,
            explore,
            nextState
        });

        path.push(nextState);
        currentState = nextState;

        if (nextState === CONFIG.frozenLake.goal || CONFIG.frozenLake.holes.includes(nextState)) {
            success = nextState === CONFIG.frozenLake.goal;
            terminated = true;
            break;
        }
    }

    return {
        episode: episodeIndex + 1,
        path,
        decisions,
        explorationRate,
        success,
        reward: success ? 1 : 0,
        terminalState: path[path.length - 1],
        terminated,
        steps: path.length - 1
    };
}

function togglePlayback() {
    if (!SIM_STATE.episodes.length) {
        return;
    }

    SIM_STATE.isPlaying = !SIM_STATE.isPlaying;
    updatePlaybackButton();

    if (SIM_STATE.isPlaying) {
        playFromCurrentPosition();
    } else {
        stopPlaybackTimer();
    }
}

function playFromCurrentPosition() {
    stopPlaybackTimer();

    SIM_STATE.timerId = setTimeout(() => {
        const episode = SIM_STATE.episodes[SIM_STATE.currentEpisodeIndex];

        if (!episode) {
            SIM_STATE.isPlaying = false;
            updatePlaybackButton();
            return;
        }

        SIM_STATE.currentFrame += 1;
        if (SIM_STATE.currentFrame >= episode.path.length) {
            if (SIM_STATE.currentEpisodeIndex >= SIM_STATE.episodes.length - 1) {
                SIM_STATE.currentFrame = episode.path.length - 1;
                SIM_STATE.isPlaying = false;
                updatePlaybackButton();
            } else {
                SIM_STATE.currentEpisodeIndex += 1;
                SIM_STATE.currentFrame = 0;
            }
        }

        renderCurrentEpisodeFrame();
        updateSimulationStats();
        updateEpisodeNarrative();
        updateTimelineChart();

        if (SIM_STATE.isPlaying) {
            playFromCurrentPosition();
        }
    }, Math.max(70, Math.floor(280 / SIM_STATE.speed)));
}

function resetSimulation() {
    SIM_STATE.currentEpisodeIndex = 0;
    SIM_STATE.currentFrame = 0;
    SIM_STATE.isPlaying = false;
    stopPlaybackTimer();
    renderCurrentEpisodeFrame();
    updateSimulationStats();
    updateEpisodeNarrative();
    updateTimelineChart();
    updatePlaybackButton();
}

function setSimulationSpeed(speed) {
    SIM_STATE.speed = Number(speed) || 1;
    if (SIM_STATE.isPlaying) {
        playFromCurrentPosition();
    }
}

function jumpToEpisode(value) {
    const target = Math.max(1, Math.min(SIM_STATE.totalEpisodes, Number(value))) - 1;
    SIM_STATE.currentEpisodeIndex = target;
    SIM_STATE.currentFrame = 0;
    renderCurrentEpisodeFrame();
    updateSimulationStats();
    updateEpisodeNarrative();
    updateTimelineChart();
}

function stopPlaybackTimer() {
    if (SIM_STATE.timerId) {
        clearTimeout(SIM_STATE.timerId);
        SIM_STATE.timerId = null;
    }
}

function updatePlaybackButton() {
    const btn = document.getElementById('simPlayPause');
    if (!btn) {
        return;
    }
    btn.textContent = SIM_STATE.isPlaying ? '⏸ Pause' : '▶ Play';
}

function updateAlgorithmButtons(triggerElement, algorithmType) {
    document.querySelectorAll('.sim-btn[data-algo]').forEach((btn) => {
        btn.classList.remove('active');
        if ((triggerElement && btn === triggerElement) || btn.getAttribute('data-algo') === algorithmType) {
            btn.classList.add('active');
        }
    });
}

function renderCurrentEpisodeFrame() {
    const episode = SIM_STATE.episodes[SIM_STATE.currentEpisodeIndex];
    if (!episode) {
        return;
    }

    drawSimulationGrid(episode, SIM_STATE.currentFrame);
    drawTrajectory(episode);

    const slider = document.getElementById('simEpisodeSlider');
    if (slider) {
        slider.value = String(SIM_STATE.currentEpisodeIndex + 1);
    }

    const currentEpisode = document.getElementById('simCurrentEpisode');
    const totalEpisodes = document.getElementById('simTotalEpisodes');
    if (currentEpisode) {
        currentEpisode.textContent = String(SIM_STATE.currentEpisodeIndex + 1);
    }
    if (totalEpisodes) {
        totalEpisodes.textContent = String(SIM_STATE.totalEpisodes);
    }
}

function drawSimulationGrid(episode, frame) {
    const canvas = document.getElementById('simCanvas');
    if (!canvas) {
        return;
    }

    drawGrid('simCanvas', SIM_STATE.policy);
    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / CONFIG.gridSize;
    const safeFrame = Math.max(0, Math.min(frame, episode.path.length - 1));

    // Animate traversed path until this frame.
    ctx.strokeStyle = '#ff8a3d';
    ctx.lineWidth = 4;
    ctx.beginPath();
    for (let i = 0; i <= safeFrame; i++) {
        const point = getStatePosition(episode.path[i]);
        const px = point.col * cellSize + cellSize / 2;
        const py = point.row * cellSize + cellSize / 2;
        if (i === 0) {
            ctx.moveTo(px, py);
        } else {
            ctx.lineTo(px, py);
        }
    }
    ctx.stroke();

    // Current agent marker.
    const state = episode.path[safeFrame];
    const pos = getStatePosition(state);
    const x = pos.col * cellSize + cellSize / 2;
    const y = pos.row * cellSize + cellSize / 2;

    const pulseRadius = cellSize * (0.23 + 0.05 * Math.sin(Date.now() / 140));
    ctx.fillStyle = 'rgba(2, 214, 193, 0.35)';
    ctx.beginPath();
    ctx.arc(x, y, pulseRadius + 8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = CONFIG.colors.agent;
    ctx.beginPath();
    ctx.arc(x, y, pulseRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#173b63';
    ctx.font = 'bold 13px "IBM Plex Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('A', x, y);
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

function drawTrajectory(episode) {
    const canvas = document.getElementById('trajCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / CONFIG.gridSize;

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid('trajCanvas', SIM_STATE.policy);

    ctx.strokeStyle = CONFIG.colors.path;
    ctx.lineWidth = 3;
    ctx.beginPath();

    episode.path.forEach((state, idx) => {
        const pos = getStatePosition(state);
        const x = pos.col * cellSize + cellSize / 2;
        const y = pos.row * cellSize + cellSize / 2;
        if (idx === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    const terminal = getStatePosition(episode.path[episode.path.length - 1]);
    ctx.fillStyle = episode.success ? '#10b981' : '#ef4444';
    ctx.beginPath();
    ctx.arc(terminal.col * cellSize + cellSize / 2, terminal.row * cellSize + cellSize / 2, 10, 0, Math.PI * 2);
    ctx.fill();
}

function updateSimulationStats() {
    if (!SIM_STATE.episodes.length) {
        return;
    }

    const upto = SIM_STATE.currentEpisodeIndex + 1;
    const observed = SIM_STATE.episodes.slice(0, upto);
    const totalReward = observed.reduce((sum, ep) => sum + ep.reward, 0);
    const totalSteps = observed.reduce((sum, ep) => sum + ep.steps, 0);
    const successCount = observed.reduce((sum, ep) => sum + (ep.success ? 1 : 0), 0);
    const current = SIM_STATE.episodes[SIM_STATE.currentEpisodeIndex];

    setText('simEpisode', String(upto));
    setText('simReturn', (totalReward / Math.max(1, upto)).toFixed(2));
    setText('simSteps', String(totalSteps));
    setText('simSuccess', `${((successCount / Math.max(1, upto)) * 100).toFixed(1)}%`);
    setText('simOutcome', current.success ? 'Reached goal' : current.terminalState === CONFIG.frozenLake.goal ? 'Reached goal' : 'Fell or timed out');
    setText('simPathLen', String(current.steps));
    setText('simExploration', current.explorationRate.toFixed(2));
}

function updateEpisodeNarrative() {
    const episode = SIM_STATE.episodes[SIM_STATE.currentEpisodeIndex];
    if (!episode) {
        return;
    }

    const stepIndex = Math.max(0, Math.min(SIM_STATE.currentFrame - 1, episode.decisions.length - 1));
    const step = episode.decisions[stepIndex];
    const actionNames = ['Up', 'Down', 'Left', 'Right'];

    let message = `Episode ${episode.episode}: agent starts at state 0 and follows ${SIM_STATE.algorithm.toUpperCase()} policy.`;
    if (step) {
        message = `Episode ${episode.episode}, step ${step.step}: state ${step.state} -> ${step.nextState} via ${actionNames[step.action]} (${step.explore ? 'explore' : 'greedy'}).`;
    }

    setText('simDecision', message);

    const notes = document.getElementById('simEpisodeNotes');
    if (!notes) {
        return;
    }

    const explorationShare = episode.decisions.length
        ? episode.decisions.filter((d) => d.explore).length / episode.decisions.length
        : 0;

    notes.innerHTML = '';
    [
        `Terminal state: ${episode.terminalState}`,
        `Reward: ${episode.reward.toFixed(2)}`,
        `Steps: ${episode.steps}`,
        `Exploratory actions: ${(explorationShare * 100).toFixed(1)}%`
    ].forEach((item) => {
        const li = document.createElement('li');
        li.textContent = item;
        notes.appendChild(li);
    });
}

function buildTimelineChart() {
    const ctx = document.getElementById('simTimelineChart');
    if (!ctx || SIM_STATE.chart) {
        return;
    }

    const labels = Array.from({ length: SIM_STATE.totalEpisodes }, (_, i) => i + 1);
    SIM_STATE.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Episode Return',
                    data: [],
                    borderColor: '#0c3f8f',
                    backgroundColor: 'rgba(12, 63, 143, 0.12)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Running Success %',
                    data: [],
                    borderColor: '#02d6c1',
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y1'
                },
                {
                    label: 'Current Episode',
                    data: [],
                    borderColor: '#ff8a3d',
                    borderWidth: 0,
                    pointBackgroundColor: '#ff8a3d',
                    pointRadius: 4,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Return' }
                },
                y1: {
                    min: 0,
                    max: 100,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Success %' }
                },
                x: {
                    title: { display: true, text: 'Episode' }
                }
            }
        }
    });
}

function updateTimelineChart() {
    if (!SIM_STATE.chart || !SIM_STATE.episodes.length) {
        return;
    }

    const returns = SIM_STATE.episodes.map((ep) => ep.reward);
    let successSoFar = 0;
    const runningSuccess = SIM_STATE.episodes.map((ep, idx) => {
        successSoFar += ep.success ? 1 : 0;
        return (successSoFar / (idx + 1)) * 100;
    });
    const markerData = SIM_STATE.episodes.map((_, idx) => idx === SIM_STATE.currentEpisodeIndex ? SIM_STATE.episodes[idx].reward : null);

    SIM_STATE.chart.data.datasets[0].data = returns;
    SIM_STATE.chart.data.datasets[1].data = runningSuccess;
    SIM_STATE.chart.data.datasets[2].data = markerData;
    SIM_STATE.chart.update('none');
}

function setText(id, text) {
    const node = document.getElementById(id);
    if (node) {
        node.textContent = text;
    }
}

function refreshVisibleCharts(tabId = '') {
    const activeTabId = tabId || (document.querySelector('.tab-content.active')?.id ?? '');

    // Charts initialized in hidden tabs can render at zero width.
    // Resize only currently relevant charts to keep layout stable.
    if (activeTabId === 'training') {
        [convergenceChart, returnsChart, lossChart, epsilonChart].forEach((chart) => {
            if (chart) {
                chart.resize();
                chart.update('none');
            }
        });
    }

    if (activeTabId === 'agent-sim' && SIM_STATE.chart) {
        SIM_STATE.chart.resize();
        SIM_STATE.chart.update('none');
        updateTimelineChart();
    }
}

// ========================================
// NAVIGATION
// ========================================

function switchTab(tabId, linkElement = null) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

    // Show selected tab
    const activeTab = document.getElementById(tabId);
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // Update sidebar
    document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
    if (linkElement && linkElement.parentElement) {
        linkElement.parentElement.classList.add('active');
    }

    // Re-render canvases when a tab becomes visible to avoid hidden-tab blank canvas bugs.
    if (tabId === 'environment') {
        setTimeout(() => renderEnvironmentCanvas(), 25);
    } else if (tabId === 'policies') {
        setTimeout(() => renderPolicyCanvases(), 25);
    } else if (tabId === 'values') {
        setTimeout(() => renderValueCanvases(), 25);
    }

    // Delay ensures layout is committed before Chart.js resize.
    setTimeout(() => refreshVisibleCharts(tabId), 50);

    return false;
}

// ========================================
// INITIALIZATION ON PAGE LOAD
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    initializeVisualizations();
    buildTimelineChart();

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

    const slider = document.getElementById('simEpisodeSlider');
    if (slider) {
        slider.max = String(SIM_STATE.totalEpisodes);
    }

    // Initialize first simulation with full-episode timeline.
    runSimulation('ql');
    togglePlayback();

    window.addEventListener('resize', () => {
        refreshVisibleCharts();
    });
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
