/* ======================================================
   CliffWalking 3D RL Arena  –  app.js
   Environment: 4 rows × 12 cols, 48 states.
     Start = 36 (row 3, col 0)   Goal = 47 (row 3, col 11)
     Cliff = states 37–46 (row 3, cols 1–10)  → −100 reward
   Plateau (rows 0–2) is the elevated safe land.
   Row 3 is the lower ledge; the cliff gap is empty / lava.
   ====================================================== */
(() => {
    'use strict';

    // ── Constants ────────────────────────────────────────────────
    const ROWS = 4, COLS = 12;
    const CELL  = 1.5;          // world units per grid cell
    const CLIFF = new Set([37,38,39,40,41,42,43,44,45,46]);
    const START = 36, GOAL = 47;
    const PLATEAU_Y = 0.0;       // y-level for rows 0-2
    const LEDGE_Y   = -1.5;      // y-level for row 3 (start/goal)
    const LAVA_Y    = -3.8;      // y-level for lava abyss
    const AGENT_H   = 0.36;      // agent hover height above tile surface

    // ── Model definitions ────────────────────────────────────────
    // Each model = { name, color, desc, successRate, avgReturn, avgSteps, episodes[] }
    // Paths follow canonical CliffWalking results (Sutton & Barto, Ch.6).
    // Q-Learning (off-policy) → greedy near-cliff path (row 2)
    // SARSA (on-policy)       → safe path (row 1)
    // DQN                     → near-optimal (row 2 most episodes)
    // Policy Iteration        → always optimal (row 2)
    // TD(λ)                   → very safe (row 0 / top row)
    const MODELS = {
        q_learning: {
            name: 'Q-Learning', color: 0x34d399, hex: '#34d399',
            desc: 'Q-Learning — Off-policy TD. Greedy → near-cliff path. Occasionally falls during early training.',
            successRate: '83%', avgReturn: '-13.5', avgSteps: '14',
            episodes: [
                // ep 1: falls off cliff (early training behaviour)
                [36, 37],
                // ep 2: optimal near-cliff path via row 2
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47],
                // ep 3: same optimal path
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]
            ]
        },
        sarsa: {
            name: 'SARSA', color: 0x60a5fa, hex: '#60a5fa',
            desc: 'SARSA — On-policy TD. Conservative: takes safer route via row 1 to avoid cliff risk.',
            successRate: '91%', avgReturn: '-17.2', avgSteps: '18',
            episodes: [
                [36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47],
                [36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47],
                [36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47]
            ]
        },
        dqn: {
            name: 'DQN', color: 0xf472b6, hex: '#f472b6',
            desc: 'DQN — Deep Q-Network. Neural policy finds near-optimal path; occasionally takes safer detour.',
            successRate: '88%', avgReturn: '-14.1', avgSteps: '15',
            episodes: [
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47],
                [36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47],
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]
            ]
        },
        policy_iter: {
            name: 'Policy Iteration', color: 0xfbbf24, hex: '#fbbf24',
            desc: 'Policy Iteration — Dynamic programming. Computes optimal policy exactly; always takes shortest path.',
            successRate: '96%', avgReturn: '-13.0', avgSteps: '13',
            episodes: [
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47],
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47],
                [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]
            ]
        },
        td_lambda: {
            name: 'TD(λ)', color: 0xc084fc, hex: '#c084fc',
            desc: 'TD(λ) — Eligibility traces. Very cautious: routes via top row (row 0) to maximise safety.',
            successRate: '89%', avgReturn: '-19.0', avgSteps: '21',
            episodes: [
                [36, 24, 12,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 35, 47],
                [36, 24, 12,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23, 35, 47],
                [36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47]
            ]
        }
    };

    // ── Simulation state ─────────────────────────────────────────
    const sim = {
        modelKey: 'q_learning',
        epIdx: 0,
        step: 0,
        tween: 1,       // 0..1 lerp between steps
        prevPos: new THREE.Vector3(),
        nextPos: new THREE.Vector3(),
        playing: false,
        compareMode: false,
        speed: 1.0,
        fell: false,
        fallProgress: 0
    };

    // ── Three.js references ───────────────────────────────────────
    let renderer, camera, scene, controls;
    let agentMesh, agentLight, trailLine, trailBuf;
    const TRAIL_LEN = 30;
    const ghostAgents = [];  // compare mode agents

    // ─────────────────────────────────────────────────────────────
    // World helpers
    // ─────────────────────────────────────────────────────────────
    function stateToWorld(s) {
        const r = Math.floor(s / COLS), c = s % COLS;
        const x = c * CELL;
        const z = r * CELL;
        const tileY = (r < 3) ? PLATEAU_Y : LEDGE_Y;
        return new THREE.Vector3(x, tileY + AGENT_H, z);
    }

    function isCliff(s) { return CLIFF.has(s); }
    function isGoal(s)  { return s === GOAL; }

    // ─────────────────────────────────────────────────────────────
    // Scene construction
    // ─────────────────────────────────────────────────────────────
    function buildScene() {
        const canvas = document.getElementById('three-canvas');
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.1;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x090d16);
        scene.fog = new THREE.FogExp2(0x090d16, 0.022);

        const cx = (COLS - 1) * CELL / 2;   // 8.25
        const cz = (ROWS - 1) * CELL / 2;   // 2.25

        camera = new THREE.PerspectiveCamera(52, 1, 0.1, 300);
        camera.position.set(cx, 11, -2);
        camera.lookAt(cx, -0.5, cz);

        controls = new THREE.OrbitControls(camera, canvas);
        controls.target.set(cx, -0.5, cz);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 5;
        controls.maxDistance = 40;
        controls.update();

        // Double-click resets camera
        canvas.addEventListener('dblclick', () => {
            camera.position.set(cx, 11, -2);
            controls.target.set(cx, -0.5, cz);
            controls.update();
        });

        buildLighting();
        buildPlateau();
        buildCliffZone();
        buildAgent();
        buildTrail();
        buildGhostAgents();

        window.addEventListener('resize', onResize);
        onResize();
    }

    // ── Lighting ─────────────────────────────────────────────────
    function buildLighting() {
        scene.add(new THREE.AmbientLight(0x1a2030, 2.0));

        const sun = new THREE.DirectionalLight(0xfff3d0, 3.0);
        sun.position.set(6, 16, -5);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        Object.assign(sun.shadow.camera, { left:-22, right:22, top:15, bottom:-10, near:1, far:60 });
        scene.add(sun);

        const fill = new THREE.DirectionalLight(0x4466aa, 0.6);
        fill.position.set(-8, 4, 8);
        scene.add(fill);

        // Lava glow (flickering – updated each frame)
        const lavaPt = new THREE.PointLight(0xff3300, 5, 18);
        lavaPt.position.set((COLS - 1) * CELL / 2, LAVA_Y + 1.5, 3 * CELL);
        scene.add(lavaPt);
        scene._lavaLight = lavaPt;

        // Goal glow
        const goalPt = new THREE.PointLight(0xffd700, 2.5, 6);
        goalPt.position.set(11 * CELL, LEDGE_Y + 1.0, 3 * CELL);
        scene.add(goalPt);

        // Start glow
        const startPt = new THREE.PointLight(0x44ff88, 2.0, 5);
        startPt.position.set(0, LEDGE_Y + 1.0, 3 * CELL);
        scene.add(startPt);
    }

    // ── Plateau (rows 0-2) + start/goal tiles ────────────────────
    const TILE_W = 1.3, TILE_H = 0.28;
    const plateauColors = [0x2b4466, 0x2d4a6e, 0x304f72, 0x2a4060, 0x2c4568];

    function buildPlateau() {
        const tileGeom = new THREE.BoxGeometry(TILE_W, TILE_H, TILE_W);

        for (let s = 0; s < ROWS * COLS; s++) {
            const r = Math.floor(s / COLS), c = s % COLS;

            // Skip cliff states (no tile)
            if (isCliff(s)) continue;

            const tileY = (r < 3) ? PLATEAU_Y : LEDGE_Y;
            let color, emissive = 0x000000, emissInt = 0, roughness = 0.72;

            if (s === START) {
                color = 0x1a6636; emissive = 0x11ff55; emissInt = 0.25; roughness = 0.45;
            } else if (s === GOAL) {
                color = 0xb07a08; emissive = 0xffd700; emissInt = 0.55; roughness = 0.35;
            } else {
                color = plateauColors[(r * 5 + c * 3) % plateauColors.length];
            }

            const mat = new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.12, emissive, emissiveIntensity: emissInt });
            const mesh = new THREE.Mesh(tileGeom, mat);
            mesh.position.set(c * CELL, tileY + TILE_H / 2, r * CELL);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);

            // Edge highlight strip on top of each plateau tile
            if (r < 3 && s !== START && s !== GOAL) {
                const strip = new THREE.Mesh(
                    new THREE.BoxGeometry(TILE_W - 0.04, 0.02, TILE_W - 0.04),
                    new THREE.MeshStandardMaterial({ color: 0x3d6494, emissive: 0x2255aa, emissiveIntensity: 0.2 })
                );
                strip.position.set(c * CELL, tileY + TILE_H + 0.01, r * CELL);
                scene.add(strip);
            }
        }

        // Decorative rock pillars on some plateau tiles
        const pillarSpots = [2, 5, 8, 14, 21, 28, 33];
        for (const s of pillarSpots) {
            const r = Math.floor(s / COLS), c = s % COLS;
            const h = 0.3 + Math.sin(s * 1.7) * 0.1;
            const rock = new THREE.Mesh(
                new THREE.CylinderGeometry(0.12, 0.16, h, 6),
                new THREE.MeshStandardMaterial({ color: 0x1e3050, roughness: 0.9 })
            );
            rock.position.set(c * CELL, PLATEAU_Y + TILE_H + h / 2, r * CELL);
            scene.add(rock);
        }
    }

    // ── Cliff zone: face wall, lava, side walls ──────────────────
    function buildCliffZone() {
        const cliffFaceW = COLS * CELL + 1;
        const cliffFaceH = Math.abs(LEDGE_Y - PLATEAU_Y) + 0.3;
        const cliffFaceZ = 3 * CELL - CELL / 2 + 0.75;  // south face of row 2

        // Main cliff face wall (south side of plateau row 2)
        const faceGeom = new THREE.BoxGeometry(cliffFaceW, cliffFaceH, 0.18);
        const faceMat = new THREE.MeshStandardMaterial({ color: 0x131d2e, roughness: 0.95, metalness: 0.05 });
        const faceWall = new THREE.Mesh(faceGeom, faceMat);
        faceWall.position.set((COLS - 1) * CELL / 2, PLATEAU_Y - cliffFaceH / 2, cliffFaceZ);
        faceWall.castShadow = true;
        faceWall.receiveShadow = true;
        scene.add(faceWall);

        // Horizontal ledge connecting cliff face to lower ledge level (row 3)
        const ledgeConnW = COLS * CELL + 1;
        const ledgeConn = new THREE.Mesh(
            new THREE.BoxGeometry(ledgeConnW, 0.14, CELL),
            new THREE.MeshStandardMaterial({ color: 0x0e1628, roughness: 0.9 })
        );
        ledgeConn.position.set((COLS - 1) * CELL / 2, LEDGE_Y + 0.07, 3 * CELL);
        ledgeConn.receiveShadow = true;
        scene.add(ledgeConn);

        // Lava abyss floor under cliff zone
        const lavaMat = new THREE.MeshStandardMaterial({
            color: 0x1c0400, emissive: 0xff2200, emissiveIntensity: 0.7,
            roughness: 0.85, metalness: 0.0
        });
        const lavaFloor = new THREE.Mesh(
            new THREE.PlaneGeometry(COLS * CELL + 1, CELL * 2),
            lavaMat
        );
        lavaFloor.rotation.x = -Math.PI / 2;
        lavaFloor.position.set((COLS - 1) * CELL / 2, LAVA_Y, 3 * CELL);
        scene.add(lavaFloor);
        scene._lavaMesh = lavaFloor;

        // Lava surface ripple rings (3 flat torus)
        for (let i = 0; i < 4; i++) {
            const rx = 1 + i * 2.5, rz = 0.6 + i * 1.5;
            const ring = new THREE.Mesh(
                new THREE.TorusGeometry(rx, 0.06, 6, 32),
                new THREE.MeshStandardMaterial({ color: 0xff6600, emissive: 0xff4400, emissiveIntensity: 0.9, transparent: true, opacity: 0.6 })
            );
            ring.rotation.x = -Math.PI / 2;
            ring.position.set((COLS - 1) * CELL / 2 + (i - 1.5) * 1.8, LAVA_Y + 0.05, 3 * CELL + (i % 2) * 0.5);
            scene.add(ring);
            scene._lavaRings = scene._lavaRings || [];
            scene._lavaRings.push(ring);
        }

        // Warning border cracks along the edge of the cliff
        const crackMat = new THREE.MeshStandardMaterial({ color: 0xff2200, emissive: 0xff2200, emissiveIntensity: 0.8 });
        for (let c = 1; c <= 10; c++) {
            const crack = new THREE.Mesh(new THREE.BoxGeometry(TILE_W + 0.05, 0.04, 0.06), crackMat);
            crack.position.set(c * CELL, LEDGE_Y + 0.14, 3 * CELL - 0.75);
            scene.add(crack);
        }

        // Ground plane (far background)
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(60, 60),
            new THREE.MeshStandardMaterial({ color: 0x080c14, roughness: 1.0 })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.set((COLS - 1) * CELL / 2, LAVA_Y - 0.1, (ROWS - 1) * CELL / 2);
        ground.receiveShadow = true;
        scene.add(ground);
    }

    // ── Agent ─────────────────────────────────────────────────────
    function buildAgent() {
        const col = MODELS[sim.modelKey].color;
        agentMesh = new THREE.Mesh(
            new THREE.SphereGeometry(0.26, 24, 24),
            new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.85, roughness: 0.2, metalness: 0.3 })
        );
        agentMesh.castShadow = true;
        scene.add(agentMesh);

        agentLight = new THREE.PointLight(col, 3.5, 5);
        agentMesh.add(agentLight);

        const startPos = stateToWorld(START);
        agentMesh.position.copy(startPos);
        sim.prevPos.copy(startPos);
        sim.nextPos.copy(startPos);
    }

    // ── Trail ─────────────────────────────────────────────────────
    const trailHistory = [];
    function buildTrail() {
        trailBuf = new Float32Array(TRAIL_LEN * 3);
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(trailBuf, 3));
        geom.setDrawRange(0, 0);
        const col = MODELS[sim.modelKey].color;
        trailLine = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.55 }));
        scene.add(trailLine);
    }

    function pushTrail(pos) {
        trailHistory.push(pos.clone());
        if (trailHistory.length > TRAIL_LEN) trailHistory.shift();
        const n = trailHistory.length;
        for (let i = 0; i < n; i++) {
            trailBuf[i * 3]     = trailHistory[i].x;
            trailBuf[i * 3 + 1] = trailHistory[i].y;
            trailBuf[i * 3 + 2] = trailHistory[i].z;
        }
        trailLine.geometry.attributes.position.needsUpdate = true;
        trailLine.geometry.setDrawRange(0, n);
    }

    function clearTrail() {
        trailHistory.length = 0;
        trailLine.geometry.setDrawRange(0, 0);
    }

    // ── Ghost agents (compare mode) ──────────────────────────────
    function buildGhostAgents() {
        for (const key of Object.keys(MODELS)) {
            const m = MODELS[key];
            const g = new THREE.Mesh(
                new THREE.SphereGeometry(0.21, 18, 18),
                new THREE.MeshStandardMaterial({ color: m.color, emissive: m.color, emissiveIntensity: 0.7, transparent: true, opacity: 0.88 })
            );
            const gl = new THREE.PointLight(m.color, 2.0, 4);
            g.add(gl);
            g.visible = false;
            scene.add(g);
            ghostAgents.push({ key, mesh: g, step: 0, tween: 0, fell: false, prevPos: new THREE.Vector3(), nextPos: new THREE.Vector3() });
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Simulation logic
    // ─────────────────────────────────────────────────────────────
    function currentEpisode() {
        return MODELS[sim.modelKey].episodes[sim.epIdx];
    }

    function initSim() {
        const ep = currentEpisode();
        sim.step = 0; sim.tween = 1; sim.fell = false; sim.fallProgress = 0;
        clearTrail();
        const p = stateToWorld(ep[0]);
        agentMesh.position.copy(p);
        sim.prevPos.copy(p);
        sim.nextPos.copy(p);
        agentMesh.visible = true;
        trailLine.visible = true;
        setAgentColor(sim.modelKey);
        updateUI();
    }

    function setAgentColor(key) {
        const col = MODELS[key].color;
        agentMesh.material.color.setHex(col);
        agentMesh.material.emissive.setHex(col);
        agentLight.color.setHex(col);
        trailLine.material.color.setHex(col);
    }

    function stepDuration() { return 0.52 / sim.speed; }  // seconds per step

    function tickMainAgent(dt) {
        if (!sim.playing) return;

        // If agent already fell, animate fall
        if (sim.fell) {
            sim.fallProgress += dt * 2.2;
            agentMesh.position.y = sim.nextPos.y - sim.fallProgress * 2;
            agentMesh.material.emissiveIntensity = Math.max(0, 0.85 - sim.fallProgress * 0.8);
            if (sim.fallProgress > 1.5) {
                sim.playing = false;
                agentMesh.visible = false;
                updateUI();
            }
            return;
        }

        sim.tween += dt / stepDuration();
        if (sim.tween >= 1) {
            sim.tween = 1;
            pushTrail(agentMesh.position);

            const ep = currentEpisode();
            if (sim.step >= ep.length - 1) {
                sim.playing = false;
                updateUI();
                return;
            }
            sim.step++;
            const s = ep[sim.step];
            sim.prevPos.copy(agentMesh.position);
            sim.nextPos.copy(stateToWorld(s));
            sim.tween = 0;

            if (isCliff(s)) {
                sim.fell = true;
                sim.nextPos.copy(stateToWorld(s));  // position of cliff state
            }
            updateUI();
        }

        // Smooth lerp with hop arc
        const t = sim.tween;
        const arc = sim.fell ? 0 : Math.sin(t * Math.PI) * 0.28;
        agentMesh.position.lerpVectors(sim.prevPos, sim.nextPos, t);
        agentMesh.position.y += arc;
    }

    function tickGhosts(dt) {
        for (const g of ghostAgents) {
            if (!g.mesh.visible) continue;
            if (g.fell) {
                g.mesh.position.y -= dt * 3;
                g.mesh.material.opacity -= dt * 0.8;
                if (g.mesh.material.opacity <= 0) { g.mesh.visible = false; }
                continue;
            }

            g.tween += dt / stepDuration();
            if (g.tween >= 1) {
                g.tween = 1;
                const ep = MODELS[g.key].episodes[0];
                if (g.step >= ep.length - 1) {
                    // Loop the ghost
                    g.step = 0;
                    const p = stateToWorld(ep[0]);
                    g.mesh.position.copy(p);
                    g.prevPos.copy(p);
                    g.nextPos.copy(p);
                    g.tween = 0;
                    continue;
                }
                g.step++;
                const s = ep[g.step];
                g.prevPos.copy(g.mesh.position);
                g.nextPos.copy(stateToWorld(s));
                g.tween = 0;
                if (isCliff(s)) g.fell = true;
            }

            const t = g.tween;
            const arc = Math.sin(t * Math.PI) * 0.22;
            g.mesh.position.lerpVectors(g.prevPos, g.nextPos, t);
            g.mesh.position.y += arc;
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Animation loop
    // ─────────────────────────────────────────────────────────────
    let lastTime = 0;

    function animate(ts) {
        requestAnimationFrame(animate);
        const dt = Math.min((ts - lastTime) / 1000, 0.06);
        lastTime = ts;

        // Lava flicker
        if (scene._lavaLight) {
            scene._lavaLight.intensity = 4 + Math.sin(ts * 0.0023) * 1.2 + Math.sin(ts * 0.0071) * 0.6;
        }
        if (scene._lavaMesh) {
            scene._lavaMesh.material.emissiveIntensity = 0.55 + Math.sin(ts * 0.0019) * 0.15;
        }
        if (scene._lavaRings) {
            scene._lavaRings.forEach((r, i) => {
                r.scale.setScalar(1 + Math.sin(ts * 0.001 + i * 1.1) * 0.08);
            });
        }

        controls.update();

        if (sim.compareMode) {
            tickGhosts(dt);
        } else {
            tickMainAgent(dt);
        }

        renderer.render(scene, camera);
    }

    // ─────────────────────────────────────────────────────────────
    // UI
    // ─────────────────────────────────────────────────────────────
    function updateUI() {
        const m   = MODELS[sim.modelKey];
        const ep  = currentEpisode();
        const cur = ep[Math.min(sim.step, ep.length - 1)];
        const total = ep.length - 1;

        document.getElementById('ui-model').textContent   = m.name;
        document.getElementById('ui-episode').textContent = `${sim.epIdx + 1} / ${m.episodes.length}`;
        document.getElementById('ui-step').textContent    = `${sim.step} / ${total}`;
        document.getElementById('ui-state').textContent   = cur;
        document.getElementById('ui-success').textContent = m.successRate;
        document.getElementById('ui-return').textContent  = m.avgReturn;
        document.getElementById('model-desc').textContent = m.desc;

        let statusText;
        if (sim.fell)                                       statusText = '💀 Fell off cliff!';
        else if (!sim.playing && sim.step >= total)         statusText = '🏁 Reached goal!';
        else if (sim.playing)                               statusText = '▶ Running…';
        else                                                statusText = '⏸ Paused';
        document.getElementById('ui-status').textContent = statusText;
        document.getElementById('btn-play').textContent  = sim.playing ? '⏸ Pause' : '▶ Play';
    }

    // ─────────────────────────────────────────────────────────────
    // Controls
    // ─────────────────────────────────────────────────────────────
    function bindControls() {
        // Model buttons
        document.querySelectorAll('.model-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (sim.compareMode) return;
                document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                sim.modelKey  = btn.dataset.model;
                sim.epIdx     = 0;
                sim.playing   = false;
                initSim();
            });
        });

        document.getElementById('btn-play').addEventListener('click', () => {
            if (sim.compareMode) return;
            const ep = currentEpisode();
            if (sim.step >= ep.length - 1 || sim.fell) { initSim(); }
            sim.playing = !sim.playing;
            updateUI();
        });

        document.getElementById('btn-next').addEventListener('click', () => {
            if (sim.compareMode) return;
            sim.playing = false;
            sim.epIdx = (sim.epIdx + 1) % MODELS[sim.modelKey].episodes.length;
            initSim();
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            sim.playing   = false;
            sim.compareMode = false;
            document.getElementById('btn-compare').classList.remove('active');
            hideGhosts();
            agentMesh.visible = true;
            trailLine.visible = true;
            initSim();
        });

        document.getElementById('speed-slider').addEventListener('input', e => {
            sim.speed = parseFloat(e.target.value);
            document.getElementById('speed-label').textContent = sim.speed.toFixed(1) + '×';
        });

        document.getElementById('btn-compare').addEventListener('click', () => {
            sim.compareMode = !sim.compareMode;
            document.getElementById('btn-compare').classList.toggle('active', sim.compareMode);
            if (sim.compareMode) {
                sim.playing = false;
                agentMesh.visible = false;
                trailLine.visible = false;
                startCompare();
            } else {
                hideGhosts();
                agentMesh.visible = true;
                trailLine.visible = true;
                initSim();
            }
            updateUI();
        });
    }

    function startCompare() {
        for (const g of ghostAgents) {
            const ep = MODELS[g.key].episodes[0];
            const p  = stateToWorld(ep[0]);
            g.step = 0; g.tween = 0; g.fell = false;
            g.mesh.material.opacity = 0.88;
            g.mesh.visible = true;
            g.mesh.position.copy(p);
            g.prevPos.copy(p);
            g.nextPos.copy(p);
        }
    }

    function hideGhosts() {
        for (const g of ghostAgents) g.mesh.visible = false;
    }

    // ─────────────────────────────────────────────────────────────
    // Resize
    // ─────────────────────────────────────────────────────────────
    function onResize() {
        const wrap = document.getElementById('arena-wrap');
        const w = wrap.clientWidth, h = wrap.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    // ─────────────────────────────────────────────────────────────
    // Bootstrap
    // ─────────────────────────────────────────────────────────────
    buildScene();
    bindControls();
    initSim();
    requestAnimationFrame(animate);
})();

