# Setup Guide — EcoTrust-DT

This guide explains how to set up the Python environment, install dependencies, and compile the IEEE manuscript locally or on Overleaf.

## 1. Prerequisites
- **Python:** 3.10+ (tested on Python 3.14.4, Windows 11, Intel Core i7-8665U, 15.82 GiB RAM)
- **Git:** 2.x+

## 2. Clone and Install Dependencies
```bash
git clone https://github.com/waseem087/Sustainable-AI-Agent-Scheduling.git
cd Sustainable-AI-Agent-Scheduling
```

Create and activate a virtual environment:

**Windows (Command Prompt):**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Verify Installation
Run the automated unit and invariant tests:
```bash
python -m unittest discover -s tests -v
```
You should see `Ran 3 tests ... OK`.

## 4. Compile the IEEE Manuscript (`main.tex`)
- **Overleaf (Recommended):** Upload `main.tex` and the three figure PDFs from `results/figures/` (`fig1_waterwise_baseline.pdf`, `fig2_ecotrust_pareto.pdf`, `fig3_policy_and_overhead.pdf`), then compile with **Normal** mode (not Fast/Draft).
- **Local TeX Live / MiKTeX:**
  ```bash
  pdflatex main.tex
  pdflatex main.tex
  ```

## 5. Notes on `baseline/waterwise/`
- `baseline/waterwise/` contains the unmodified `PPoPP25_WaterWise` artifact (`Apache-2.0`).
- `baseline/waterwise/data/transfer.json` is loaded relative to `baseline/waterwise/`. When running `baseline/waterwise/main.py` directly, execute it from inside `baseline/waterwise/` (or use `experiments/reproduce_waterwise.py`, which handles working directories automatically).
