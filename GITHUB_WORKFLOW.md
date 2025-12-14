# GitHub + RunPod Workflow (Recommended)

**This is the easiest method!** Push code to GitHub, then clone on RunPod.

## Setup (One-time, on Mac)

### 1. Initialize Git (if not already done)

```bash
cd experiment_B_lora_decision_engine

# If not already a git repo
git init

# Add files
git add .
git commit -m "Experiment B: LoRA training setup"
```

### 2. Create GitHub Repository

1. Go to GitHub.com
2. Create new repository (e.g., `trauma-triage-experiment-b`)
3. **Don't** initialize with README (we already have files)

### 3. Push to GitHub

```bash
# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/trauma-triage-experiment-b.git

# Or if using SSH
git remote add origin git@github.com:YOUR_USERNAME/trauma-triage-experiment-b.git

# Push
git branch -M main
git push -u origin main
```

## On RunPod (After Connecting GitHub)

### 1. Clone Repository

RunPod has a "Git" connection option. If enabled:

1. Open RunPod pod terminal
2. Navigate to workspace:
   ```bash
   cd /workspace
   ```
3. Clone your repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/trauma-triage-experiment-b.git
   cd trauma-triage-experiment-b/experiment_B_lora_decision_engine
   ```

**Or use RunPod's built-in Git integration** (if available in their UI).

### 2. Verify Setup

```bash
python3 scripts/verify_setup.py
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train

```bash
python3 scripts/train_lora.py
```

### 5. Download Model (Still needed)

Models are too large for git. Download via:
- RunPod web UI (file browser)
- Or push model to a separate branch/tag if you want versioning

## Benefits of Git Workflow

✅ **Easy updates**: Push changes from Mac, pull on RunPod  
✅ **Version control**: Track changes to scripts  
✅ **No manual upload**: Just `git pull` on cloud  
✅ **Collaboration**: Share with team easily  

## Workflow Diagram

```
Local Mac:
  1. Make changes
  2. git add .
  3. git commit -m "Update script"
  4. git push

RunPod:
  1. git pull              # Get latest changes
  2. python3 scripts/train_lora.py
  3. Download models/      # Via web UI or rsync
```

## What Gets Pushed to GitHub

**Included** (tracked):
- All scripts (`scripts/*.py`)
- Configuration (`requirements.txt`, `.gitignore`)
- Documentation (all `.md` files)
- Rules (`rules/*.json`)
- Data structure (`.gitkeep` files)

**Excluded** (by `.gitignore`):
- `models/lora_adapter/` - Too large, download separately
- `data/*.xlsx` - Large data files (upload separately or use git-lfs)
- `outputs/*.jsonl` - Generated files (can regenerate)
- `__pycache__/` - Python cache

## Updating Scripts During Development

If you need to update scripts while training:

**On Mac:**
```bash
# Make changes
git add scripts/train_lora.py
git commit -m "Fix training script"
git push
```

**On RunPod:**
```bash
git pull
# Continue training or restart
```

## Large Files (Optional: Git LFS)

If you want to track large files (datasets, models), use Git LFS:

```bash
# Install git-lfs (one-time)
git lfs install

# Track large files
git lfs track "*.xlsx"
git lfs track "models/lora_adapter/**"

git add .gitattributes
git commit -m "Add LFS tracking"
```

But for this project, manual upload/download is fine since:
- Models are generated (not source code)
- Datasets are large (better to upload separately)

## Troubleshooting

**"Repository not found"**
- Check GitHub repo URL
- Verify RunPod has GitHub access configured
- Use HTTPS if SSH doesn't work

**"Permission denied"**
- For private repos, RunPod needs GitHub token/SSH key configured
- Or use public repo for code (keep data private)

**Large files won't push**
- Use `.gitignore` to exclude them
- Or use Git LFS for large files you need to track
