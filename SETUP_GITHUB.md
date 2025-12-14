# Quick Setup: Push to GitHub

## Step 1: Initialize Git (One-time)

Run these commands in your terminal:

```bash
cd experiment_B_lora_decision_engine

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Experiment B LoRA training setup"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `trauma-triage-experiment-b` (or your choice)
3. **Make it Private** (recommended for research data)
4. **Don't** check "Initialize with README" (we already have files)
5. Click "Create repository"

## Step 3: Push to GitHub

GitHub will show you commands. Run these (replace YOUR_USERNAME and REPO_NAME):

```bash
# Add remote (use the URL GitHub shows you)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Or if you prefer SSH (if you have SSH keys set up):
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

If GitHub asks for authentication:
- Use a **Personal Access Token** (not password)
- Generate one: GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
- Give it `repo` permissions

## Step 4: Verify on GitHub

Go to your GitHub repo URL. You should see all your files!

## Step 5: On RunPod

Once RunPod is connected to your GitHub:

1. In RunPod, use the Git integration or:
   ```bash
   cd /workspace
   git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
   cd REPO_NAME/experiment_B_lora_decision_engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify setup:
   ```bash
   python3 scripts/verify_setup.py
   ```

4. Train:
   ```bash
   python3 scripts/train_lora.py
   ```

## Updating Code Later

If you make changes on Mac:

```bash
cd experiment_B_lora_decision_engine
git add .
git commit -m "Description of changes"
git push
```

Then on RunPod:
```bash
git pull
```

## What Gets Pushed

✅ **Pushed to GitHub:**
- All Python scripts
- Documentation files
- Configuration files (requirements.txt, .gitignore)
- Rules JSON
- Small data files

❌ **NOT pushed** (too large or generated):
- `models/lora_adapter/` - Download separately after training
- `data/*.xlsx` - Upload manually to cloud or use Git LFS
- `outputs/*.jsonl` - Can regenerate on cloud
- Python cache files

The `.gitignore` file handles this automatically.

## Troubleshooting

**"remote origin already exists"**
- Remove it: `git remote remove origin`
- Then add again with correct URL

**"Permission denied"**
- Use Personal Access Token instead of password
- Or set up SSH keys for GitHub

**"Large files rejected"**
- Check `.gitignore` is working
- Files should be excluded automatically
