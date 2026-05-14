# V(extra) - 3D RL Model Arena

This folder contains a standalone website that visualizes a 3D RL environment with five models:

- Q-Learning
- SARSA
- DQN
- Policy Iteration
- TD(lambda)

## Files

- index.html: Main dashboard page
- assets/style.css: Visual styling
- assets/app.js: Three.js environment, model playback, and controls

## Run

From repository root:

```powershell
python -m http.server 8000
```

Open:

- http://localhost:8000/V(extra)/

## Features

- Interactive 3D GridWorld with start/goal/hole states
- Per-model episode playback with speed control
- Next episode and reset actions
- Compare mode to animate all five models simultaneously
- Live performance overview table and trajectory chips
