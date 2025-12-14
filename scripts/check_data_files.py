#!/usr/bin/env python3
"""
Quick diagnostic script to check what data files are present.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

print("="*70)
print("DATA FILES CHECK")
print("="*70)

print(f"\nLooking in: {DATA_DIR}")
print(f"Directory exists: {DATA_DIR.exists()}")

if DATA_DIR.exists():
    print(f"\nFiles in data/ directory:")
    files = list(DATA_DIR.glob("*"))
    if files:
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            file_type = "DIR" if f.is_dir() else "FILE"
            print(f"  [{file_type}] {f.name} ({size_mb:.2f} MB)")
    else:
        print("  (empty)")

# Check for required files
required_excel = DATA_DIR / "OCH_RCH_2023_2025_Combined_Master_V11_EXP_B_COPY.xlsx"
required_json = DATA_DIR / "V11_cleaned_transcripts_gpt5nano.json"

print(f"\nRequired files:")
print(f"  Excel: {required_excel.name}")
print(f"    Exists: {'✅ YES' if required_excel.exists() else '❌ NO'}")
if required_excel.exists():
    size_mb = required_excel.stat().st_size / (1024 * 1024)
    print(f"    Size: {size_mb:.2f} MB")

print(f"  JSON: {required_json.name}")
print(f"    Exists: {'✅ YES' if required_json.exists() else '❌ NO'}")
if required_json.exists():
    size_mb = required_json.stat().st_size / (1024 * 1024)
    print(f"    Size: {size_mb:.2f} MB")

# Check for alternative file names
print(f"\nAlternative file names found:")
for pattern in ["*.xlsx", "*.xls", "*Master*.xlsx", "*V11*.xlsx", "*.json"]:
    matches = list(DATA_DIR.glob(pattern))
    if matches:
        for m in matches:
            if m.name not in [required_excel.name, required_json.name]:
                size_mb = m.stat().st_size / (1024 * 1024) if m.is_file() else 0
                print(f"  {m.name} ({size_mb:.2f} MB)")

# Check parent directory too
parent_data = PROJECT_ROOT.parent / "CombinedData"
if parent_data.exists():
    print(f"\nAlso checking parent CombinedData/ directory:")
    excel_files = list(parent_data.rglob("*.xlsx"))
    json_files = list(parent_data.rglob("*gpt5nano.json"))
    if excel_files:
        print(f"  Found {len(excel_files)} Excel files in parent")
        for f in excel_files[:5]:  # Show first 5
            print(f"    {f.relative_to(PROJECT_ROOT.parent)}")
    if json_files:
        print(f"  Found {len(json_files)} JSON files in parent")
        for f in json_files[:5]:  # Show first 5
            print(f"    {f.relative_to(PROJECT_ROOT.parent)}")

print("\n" + "="*70)
print("RECOMMENDATION:")
if not required_excel.exists():
    print("  ❌ Excel file missing - Upload via RunPod Web UI")
    print(f"     Expected: {required_excel}")
if not required_json.exists():
    print("  ❌ JSON file missing - Upload via RunPod Web UI")
    print(f"     Expected: {required_json}")
if required_excel.exists() and required_json.exists():
    print("  ✅ All required files present!")
print("="*70)
