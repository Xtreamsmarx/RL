(() => {
    const GRID_SIZE = 6;
    const CELL_SPACING = 1.55;
    const START_STATE = 0;
    const GOAL_STATE = 35;
    const HOLES = new Set([7, 12, 18, 22, 29]);

    const MODELS = {
        q_learning: {
            name: 'Q-Learning',
            color: 0x24e6c3,
            success: '86%',
            avgReturn: '0.81',
            avgSteps: '9.1',
            episodes: [
                [0, 1, 2, 8, 14, 20, 26, 27, 33, 34, 35],
                [0, 1, 2, 3, 9, 15, 21, 27, 33, 34, 35],
                [0, 1, 2, 8, 9, 15, 21, 27, 28, 34, 35]
            ]
        },
        sarsa: {
            name: 'SARSA',
            color: 0x5fdb8f,
            success: '78%',
            avgReturn: '0.74',
            avgSteps: '10.4',
            episodes: [
                [0, 1, 2, 8, 9, 15, 16, 17, 23, 29],
                [0, 1, 2, 3, 9, 15, 16, 22],
                [0, 1, 2, 8, 14, 20, 21, 27, 33, 34, 35]
            ]
        },
        dqn: {
            name: 'DQN',
            color: 0x35a7ff,
            success: '89%',
            avgReturn: '0.84',
            avgSteps: '8.7',
            episodes: [
                [0, 1, 2, 3, 9, 15, 21, 27, 33, 34, 35],
                [0, 1, 2, 8, 14, 20, 26, 27, 28, 34, 35],
                [0, 1, 2, 8, 9, 15, 21, 27, 33, 34, 35]
            ]
        },
        policy_iteration: {
            name: 'Policy Iteration',
            color: 0xfaa45f,
            success: '95%',
            avgReturn: '0.93',
            avgSteps: '7.3',
            episodes: [
                [0, 1, 2, 3, 9, 15, 21, 27, 33, 34, 35],
                [0, 1, 2, 8, 14, 20, 26, 32, 33, 34, 35],
                [0, 1, 2, 3, 4, 10, 16, 22, 28, 34, 35]
            ]
        },
        td_lambda: {
            name: 'TD(lambda)',
            color: 0xd178ff,
            success: '80%',
            avgReturn: '0.77',
            avgSteps: '9.9',
            episodes: [
                [0, 1, 2, 8, 14, 15, 16, 22],
                [0, 1, 2, 3, 9, 10, 16, 22],
                [0, 1, 2, 8, 14, 20, 26, 27, 33, 34, 35]
            ]
        }
    };

    const state = {
        activeModelKey: 'q_learning',
        activeEpisodeIndex: 0,
        playhead: 0,
        speed: 1.2,
        isPlaying: false,
        compareMode: false,
        comparePlayheads: {},
        selectedLine: null,
        ghostAgents: [],
        agent: null,
        environmentGroup: null,
        scene: null,
        camera: null,
        renderer: null,
        clock: null
    };

    const refs = {
        modelButtons: document.getElementById('modelButtons'),
        tableBody: document.getElementById('modelTableBody'),
        trajectoryStates: document.getElementById('trajectoryStates'),
        episodeMetrics: document.getElementById('episodeMetrics'),
        playPauseBtn: document.getElementById('playPauseBtn'),
        nextEpisodeBtn: document.getElementById('nextEpisodeBtn'),
        resetBtn: document.getElementById('resetBtn'),
        speedRange: document.getElementById('speedRange'),
        speedValue: document.getElementById('speedValue'),
        compareToggle: document.getElementById('compareToggle'),
        sceneContainer: document.getElementById('sceneContainer')
    };

    function toXZ(index) {
        const row = Math.floor(index / GRID_SIZE);
        const col = index % GRID_SIZE;
        return {
            x: (col - (GRID_SIZE - 1) / 2) * CELL_SPACING,
            z: (row - (GRID_SIZE - 1) / 2) * CELL_SPACING
        };
    }

    function stateType(index) {
        if (index === START_STATE) return 'start';
        if (index === GOAL_STATE) return 'goal';
        if (HOLES.has(index)) return 'hole';
        return 'frozen';
    }

    function setupScene() {
        state.scene = new THREE.Scene();
        state.scene.fog = new THREE.Fog(0x081024, 10, 38);

        state.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
        state.camera.position.set(0, 14.5, 13.5);
        state.camera.lookAt(0, 0, 0);

        state.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        state.renderer.setSize(refs.sceneContainer.clientWidth, refs.sceneContainer.clientHeight);
        refs.sceneContainer.appendChild(state.renderer.domElement);

        state.clock = new THREE.Clock();

        const hemi = new THREE.HemisphereLight(0xb4d7ff, 0x0c1326, 1.2);
        state.scene.add(hemi);

        const key = new THREE.DirectionalLight(0xffffff, 1.05);
        key.position.set(8, 12, 5);
        state.scene.add(key);

        const rim = new THREE.PointLight(0x2de2c3, 0.9, 36);
        rim.position.set(-9, 4, -6);
        state.scene.add(rim);

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(28, 28),
            new THREE.MeshStandardMaterial({ color: 0x0b1733, roughness: 0.95, metalness: 0.1 })
        );
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -1.03;
        state.scene.add(floor);

        const grid = new THREE.GridHelper(26, 26, 0x2b4f8a, 0x17335d);
        grid.position.y = -1;
        state.scene.add(grid);

        buildEnvironment();
        createAgent();
        refreshSelectedPathLine();

        window.addEventListener('resize', onResize);
        onResize();
    }

    function buildEnvironment() {
        const group = new THREE.Group();

        for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            const type = stateType(i);
            const pos = toXZ(i);
            let color = 0x9fb4d1;
            let height = 0.55;

            if (type === 'start') {
                color = 0x38c172;
                height = 0.8;
            } else if (type === 'goal') {
                color = 0x2bb4ff;
                height = 0.9;
            } else if (type === 'hole') {
                color = 0xd85b6b;
                height = 0.24;
            }

            const box = new THREE.Mesh(
                new THREE.BoxGeometry(1.22, height, 1.22),
                new THREE.MeshStandardMaterial({ color, roughness: 0.68, metalness: 0.14 })
            );
            box.position.set(pos.x, -0.95 + height / 2, pos.z);
            group.add(box);
        }

        state.environmentGroup = group;
        state.scene.add(group);
    }

    function createAgent() {
        const model = MODELS[state.activeModelKey];
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.38, 26, 26),
            new THREE.MeshStandardMaterial({ color: model.color, emissive: model.color, emissiveIntensity: 0.22 })
        );
        state.agent = sphere;
        state.scene.add(sphere);
        resetAgentToStart();
    }

    function resetAgentToStart() {
        const start = toXZ(START_STATE);
        state.agent.position.set(start.x, 0.52, start.z);
    }

    function activeEpisode() {
        return MODELS[state.activeModelKey].episodes[state.activeEpisodeIndex];
    }

    function updateAgent(dt) {
        if (!state.isPlaying || state.compareMode) {
            return;
        }

        const episode = activeEpisode();
        if (episode.length < 2) {
            return;
        }

        state.playhead += dt * state.speed;
        const maxIndex = episode.length - 1;
        if (state.playhead >= maxIndex) {
            state.playhead = maxIndex;
            state.isPlaying = false;
            refs.playPauseBtn.textContent = 'Play';
        }

        const i0 = Math.floor(state.playhead);
        const i1 = Math.min(i0 + 1, maxIndex);
        const t = state.playhead - i0;
        const p0 = toXZ(episode[i0]);
        const p1 = toXZ(episode[i1]);

        state.agent.position.x = p0.x + (p1.x - p0.x) * t;
        state.agent.position.z = p0.z + (p1.z - p0.z) * t;
    }

    function createOrUpdateGhostAgents() {
        state.ghostAgents.forEach((mesh) => state.scene.remove(mesh));
        state.ghostAgents = [];
        state.comparePlayheads = {};

        if (!state.compareMode) {
            return;
        }

        Object.entries(MODELS).forEach(([key, model]) => {
            const ghost = new THREE.Mesh(
                new THREE.SphereGeometry(0.24, 18, 18),
                new THREE.MeshStandardMaterial({ color: model.color, transparent: true, opacity: 0.88 })
            );
            const start = toXZ(model.episodes[0][0]);
            ghost.position.set(start.x, 0.26, start.z);
            state.scene.add(ghost);
            state.ghostAgents.push(ghost);
            state.comparePlayheads[key] = 0;
        });
    }

    function updateGhostAgents(dt) {
        if (!state.compareMode) {
            return;
        }

        const keys = Object.keys(MODELS);
        keys.forEach((key, idx) => {
            const episode = MODELS[key].episodes[0];
            const maxIndex = episode.length - 1;
            let playhead = state.comparePlayheads[key] + dt * state.speed * 0.85;
            if (playhead >= maxIndex) {
                playhead = 0;
            }
            state.comparePlayheads[key] = playhead;

            const i0 = Math.floor(playhead);
            const i1 = Math.min(i0 + 1, maxIndex);
            const t = playhead - i0;
            const p0 = toXZ(episode[i0]);
            const p1 = toXZ(episode[i1]);

            const ghost = state.ghostAgents[idx];
            ghost.position.x = p0.x + (p1.x - p0.x) * t;
            ghost.position.z = p0.z + (p1.z - p0.z) * t;
        });
    }

    function refreshSelectedPathLine() {
        if (state.selectedLine) {
            state.scene.remove(state.selectedLine);
            state.selectedLine.geometry.dispose();
            state.selectedLine.material.dispose();
        }

        const episode = activeEpisode();
        const pts = episode.map((s) => {
            const p = toXZ(s);
            return new THREE.Vector3(p.x, 0.12, p.z);
        });

        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({ color: MODELS[state.activeModelKey].color });
        state.selectedLine = new THREE.Line(geo, mat);
        state.scene.add(state.selectedLine);

        updateTrajectoryChips(episode);
    }

    function updateTrajectoryChips(episode) {
        refs.trajectoryStates.innerHTML = '';
        episode.forEach((stateIndex) => {
            const chip = document.createElement('span');
            chip.className = 'state-chip';
            chip.textContent = `s${stateIndex}`;
            refs.trajectoryStates.appendChild(chip);
        });
    }

    function renderModelButtons() {
        refs.modelButtons.innerHTML = '';
        Object.entries(MODELS).forEach(([key, model]) => {
            const btn = document.createElement('button');
            btn.className = `btn ${state.activeModelKey === key ? 'active' : ''}`;
            btn.textContent = model.name;
            btn.addEventListener('click', () => {
                state.activeModelKey = key;
                state.activeEpisodeIndex = 0;
                state.playhead = 0;
                state.agent.material.color.setHex(model.color);
                state.agent.material.emissive.setHex(model.color);
                state.isPlaying = false;
                refs.playPauseBtn.textContent = 'Play';
                refreshSelectedPathLine();
                updateOverview();
                renderModelButtons();
                resetAgentToStart();
            });
            refs.modelButtons.appendChild(btn);
        });
    }

    function updateOverview() {
        refs.tableBody.innerHTML = '';
        Object.values(MODELS).forEach((model) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${model.name}</td><td>${model.success}</td><td>${model.avgReturn}</td><td>${model.avgSteps}</td>`;
            refs.tableBody.appendChild(tr);
        });

        const model = MODELS[state.activeModelKey];
        const episode = activeEpisode();
        const terminal = episode[episode.length - 1];
        const reachedGoal = terminal === GOAL_STATE;

        refs.episodeMetrics.innerHTML = '';
        [
            `Model: <strong>${model.name}</strong>`,
            `Episode: <strong>${state.activeEpisodeIndex + 1}/${model.episodes.length}</strong>`,
            `Path length: <strong>${episode.length - 1} steps</strong>`,
            `Terminal state: <strong>s${terminal}</strong>`,
            `Outcome: <strong>${reachedGoal ? 'Reached goal' : 'Failed / hole'}</strong>`
        ].forEach((line) => {
            const li = document.createElement('li');
            li.innerHTML = line;
            refs.episodeMetrics.appendChild(li);
        });
    }

    function bindControls() {
        refs.playPauseBtn.addEventListener('click', () => {
            state.isPlaying = !state.isPlaying;
            refs.playPauseBtn.textContent = state.isPlaying ? 'Pause' : 'Play';
        });

        refs.nextEpisodeBtn.addEventListener('click', () => {
            const model = MODELS[state.activeModelKey];
            state.activeEpisodeIndex = (state.activeEpisodeIndex + 1) % model.episodes.length;
            state.playhead = 0;
            state.isPlaying = false;
            refs.playPauseBtn.textContent = 'Play';
            refreshSelectedPathLine();
            updateOverview();
            resetAgentToStart();
        });

        refs.resetBtn.addEventListener('click', () => {
            state.playhead = 0;
            state.isPlaying = false;
            refs.playPauseBtn.textContent = 'Play';
            resetAgentToStart();
        });

        refs.speedRange.addEventListener('input', () => {
            state.speed = Number(refs.speedRange.value);
            refs.speedValue.textContent = `${state.speed.toFixed(1)}x`;
        });

        refs.compareToggle.addEventListener('click', () => {
            state.compareMode = !state.compareMode;
            refs.compareToggle.textContent = state.compareMode
                ? 'Stop Compare Mode'
                : 'Compare All 5 Models';
            createOrUpdateGhostAgents();
            state.isPlaying = !state.compareMode && state.isPlaying;
            refs.playPauseBtn.disabled = state.compareMode;
            refs.nextEpisodeBtn.disabled = state.compareMode;
        });
    }

    function onResize() {
        const w = refs.sceneContainer.clientWidth;
        const h = refs.sceneContainer.clientHeight;
        if (w === 0 || h === 0) {
            return;
        }
        state.camera.aspect = w / h;
        state.camera.updateProjectionMatrix();
        state.renderer.setSize(w, h);
    }

    function animate() {
        requestAnimationFrame(animate);
        const dt = state.clock.getDelta();

        if (state.environmentGroup) {
            state.environmentGroup.rotation.y += dt * 0.08;
        }

        updateAgent(dt);
        updateGhostAgents(dt);

        state.renderer.render(state.scene, state.camera);
    }

    function init() {
        renderModelButtons();
        updateOverview();
        bindControls();
        setupScene();
        animate();
    }

    init();
})();
