# Website Documentation

## Quick Access

### View the Website
The interactive website dashboard is located in the `website/` folder.

**To open:**
1. **Direct:** Double-click `website/index.html` in your file explorer
2. **Local Server (Recommended):**
   ```bash
   cd website
   python -m http.server 8000
   # Then visit: http://localhost:8000
   ```

## What's in the Website

### 🎯 Main Dashboard (`index.html`)
- **Overview**: High-level project statistics and V3 highlights
- **Algorithms**: Complete documentation of all 15 algorithms with features and rationale
- **Results**: Visual charts comparing algorithm performance and DQN training progress
- **Decisions**: Detailed explanation of implementation choices and strategic decisions

### 📊 Interactive Features
- **Algorithm Tabs**: Browse algorithms by category (DP, MC, TD, SARSA, Deep RL)
- **Interactive Charts**: 
  - Classical algorithm performance comparison
  - Algorithm category distribution
  - DQN training metrics (returns + loss)
- **Responsive Design**: Works on desktop, tablet, and mobile

### 🎨 Styling & Technology
- **Framework**: Pure HTML5 + CSS3 + JavaScript (no dependencies except Chart.js)
- **Charts**: Chart.js 4.4.0 for professional visualizations
- **Design**: Modern gradient header, smooth animations, accessible colors
- **Responsive**: Mobile-first approach with breakpoints at 768px and 480px

## Integration with Project

The website integrates with your capstone:

```
website/
├── index.html          # Main dashboard
├── assets/
│   ├── style.css       # All styling (professional & responsive)
│   └── app.js          # Interactive charts & navigation
└── README.md           # Website usage guide
```

Links to external content:
- `../DECISIONS.md` - Strategic justifications
- `../README.md` - Project overview
- `../result/csv/` - Training metrics (can be integrated)
- `../result/figures/` - Run-specific visualizations
- `../visualization/` - Overview plots

## Customization

### Add Real Data
Edit `website/assets/app.js` to load your CSV results:
```javascript
async function loadCSVData() {
    // Load ../result/csv/classical_runs.csv
    // Load ../result/csv/dqn_runs.csv
    // Update chart data
}
```

### Update Algorithm Descriptions
Edit algorithm cards in `website/index.html` to reflect your implementation details.

### Change Colors
Modify CSS variables in `website/assets/style.css`:
```css
--primary-color: #1e3c72;
--accent-color: #00d4ff;
```

## Deployment

### Option 1: GitHub Pages (Free)
```bash
# Your repo is already at: https://github.com/Xtreamsmarx/RL
# Website automatically available at: https://xtreamsmarx.github.io/RL/website/
```

### Option 2: Netlify (Free)
- Drag & drop the `website` folder to netlify.com

### Option 3: Your Own Server
- Copy `website/` folder to web server
- No backend required - pure static files

## File Sizes & Performance

| File | Size | Purpose |
|------|------|---------|
| `index.html` | ~25KB | Dashboard structure |
| `assets/style.css` | ~15KB | All styling (responsive) |
| `assets/app.js` | ~12KB | Charts & interactivity |
| **Total** | **~52KB** | Loads in <1s |

No external dependencies except Chart.js CDN (loaded from CDN).

## Browser Support

✅ Tested on:
- Chrome 120+
- Firefox 121+
- Safari 17+
- Edge 120+

✅ Features:
- CSS Grid & Flexbox
- ES6 JavaScript
- Chart.js 4.x
- Responsive Design

## Next Steps

1. **View the website:**
   ```bash
   cd website
   python -m http.server 8000
   # Visit http://localhost:8000
   ```

2. **Customize algorithm cards** with your implementation details

3. **Add real data** by loading CSV files from `result/csv/`

4. **Deploy to GitHub Pages**, Netlify, or your own server

5. **Share with employers/academics** as a portfolio piece!

---

**Created for RL Course V3 Capstone** | Interactive Dashboard for Algorithm Showcase
