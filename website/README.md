# RL Capstone V3 - Interactive Website

A professional, interactive website to showcase your Reinforcement Learning capstone project with beautiful visualizations and comprehensive algorithm documentation.

## 📂 Structure

```
website/
├── index.html          # Main dashboard (open in browser)
├── assets/
│   ├── style.css       # Responsive styling
│   └── app.js          # Interactive functionality & charts
└── README.md           # This file
```

## 🚀 Quick Start

### Option 1: Open Directly in Browser
1. Navigate to this folder in your file explorer
2. Double-click `index.html` to open in your default browser

### Option 2: Use a Local Server (Recommended)

**Python 3:**
```bash
cd website
python -m http.server 8000
```
Then open: http://localhost:8000

**Node.js (http-server):**
```bash
npx http-server website -p 8000
```
Then open: http://localhost:8000

## 📊 What's Included

### 🎯 Overview Section
- High-level project statistics
- V3 highlights and key accomplishments
- Quick links to detailed documentation

### 🧠 Algorithms Section
Comprehensive documentation of all 15 algorithms organized by category:

**Dynamic Programming (2)**
- Policy Iteration
- Value Iteration

**Monte Carlo (2)**
- MC Exploring-Starts
- MC ε-Greedy

**Temporal Difference (5)**
- TD(0) Prediction
- n-Step TD Control
- TD(λ) Forward-View
- TD(λ) Backward-View
- Q-Learning

**SARSA (4)**
- SARSA(0)
- n-Step SARSA
- SARSA(λ) Forward-View
- SARSA(λ) Backward-View

**Deep RL (1)**
- Deep Q-Network (DQN)

### 📈 Results Section
- **Classical RL Comparison**: Bar chart showing performance across algorithms
- **Category Distribution**: Pie chart showing algorithm breakdown
- **DQN Training Progress**: Dual-axis chart showing returns and loss over episodes
- **Visualization Gallery**: Links to saved result files

### 🎯 Decisions Section
- **What Was Implemented**: Detailed rationale for each algorithm choice
- **What Was NOT Implemented**: Clear explanations for deliberate non-implementations
- **Environmental Fit**: Table justifying FrozenLake-v1 selection

## 🎨 Features

✨ **Responsive Design**
- Works on desktop, tablet, and mobile
- Adaptive layouts that reflow based on screen size
- Touch-friendly navigation

⚡ **Interactive Charts**
- Built with Chart.js for smooth animations
- Hover tooltips with detailed metrics
- Responsive to window resizing

🎭 **Professional Styling**
- Modern gradient header
- Smooth section transitions
- Hover effects and animations
- Accessible color scheme

📱 **Navigation**
- Sticky header with quick navigation
- Smooth scrolling between sections
- Tab-based algorithm exploration

## 🔧 Customization

### Update Algorithm Cards
Edit `index.html` to modify algorithm descriptions in the `#algorithms` section.

### Change Colors
Edit `:root` variables in `assets/style.css`:
```css
--primary-color: #1e3c72;
--accent-color: #00d4ff;
```

### Add Real Data
Modify `assets/app.js` to load CSV data from `../result/csv/`:
```javascript
async function loadCSVData() {
    // Load and parse CSV files
    // Update chart data dynamically
}
```

### Add Visualizations
The `#results` section has a visualization gallery. Add links to:
- `../result/figures/` - Per-run training curves and saliency maps
- `../visualization/` - Overview plots and consolidated visualizations
- `../result/csv/` - Raw result data

## 🔗 Integration with Capstone

This website seamlessly integrates with your main RL project:

```
RL_Course_V1/
├── DECISIONS.md          ← Strategic justifications
├── README.md             ← Project overview
├── website/              ← This folder
│   ├── index.html        ← Dashboard
│   └── assets/           ← CSS & JS
├── result/
│   ├── csv/              ← Training metrics
│   └── figures/          ← Run-specific plots
├── visualization/        ← Overview plots
└── src/rl_course/        ← Algorithm implementations
```

## 📚 Viewing Your Results

Once you've generated results from training, they automatically appear in:

**CSV Data:**
- `result/csv/classical_runs.csv` - Summary of all classical training runs
- `result/csv/dqn_runs.csv` - Summary of all DQN runs
- `result/csv/classical_run_TIMESTAMP.csv` - Individual run details

**Figures:**
- `result/figures/` - Training curves and evaluation plots
- `visualization/` - Overview plots across all runs
- `reports/figures/` - Saliency maps and special visualizations

The website provides a gallery interface to browse these results.

## 🎓 Educational Value

This website demonstrates:
- ✅ Full-stack web development (HTML, CSS, JavaScript)
- ✅ Data visualization techniques (Chart.js)
- ✅ Responsive UI design principles
- ✅ Professional documentation presentation
- ✅ Integration of complex technical content with user-friendly interfaces

Perfect for showcasing to employers or academic audiences!

## 📝 Notes

- The website works **offline** - no internet connection required
- Charts use placeholder data; update `assets/app.js` to load your actual CSV results
- All links are relative paths, so keep the folder structure intact
- Tested on Chrome, Firefox, Safari, and Edge

## 🚀 Deployment

To publish this website:

1. **GitHub Pages** (Free):
   ```bash
   git add website/
   git commit -m "Add interactive capstone website"
   git push origin main
   ```
   Then enable Pages in GitHub settings

2. **Netlify** (Free):
   - Drag & drop the `website` folder
   - Instant deployment

3. **Your own server**:
   - Copy the `website` folder to your web server
   - No backend required - pure static files

## 📞 Support

For questions about the website:
- Check HTML comments for structure
- See `assets/style.css` for styling options
- Review `assets/app.js` for interactivity

For RL algorithm questions:
- See `DECISIONS.md` for implementation rationale
- Check `docs/technical-challenges.md` for debugging insights
- Review `README.md` for usage examples

---

**Built for RL Course V3 Capstone** | May 2026
