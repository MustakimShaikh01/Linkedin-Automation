"""
QLoRA Fine-tuner for Apple Silicon (Mac M2/M3)
Uses Apple MLX framework - native Metal GPU, no CUDA needed.

Speed-optimised for M2: fewer iters, smaller sequences, gradient checkpointing.

Install:
    pip install mlx-lm huggingface_hub

Workflow (run in order):
    python llm/mlx_qlora_trainer.py --prepare       # 1. convert dataset
    python llm/mlx_qlora_trainer.py --estimate      # 2. see time estimates
    python llm/mlx_qlora_trainer.py --train         # 3. train (~25-45 min)
    python llm/mlx_qlora_trainer.py --train --fast  # 3b. ultra-fast test (8 min)
    python llm/mlx_qlora_trainer.py --test          # 4. run inference
    python llm/mlx_qlora_trainer.py --fuse          # 5. fuse into model
    python llm/mlx_qlora_trainer.py --push REPO     # 6. push to HuggingFace
"""

import json
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ─── Model Config ─────────────────────────────────────────────────────────────
MODELS = {
    "phi3-mini":   "microsoft/Phi-3-mini-4k-instruct",   # 3.8B  best for M2
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.2", # 7B    high quality
    "phi3-medium": "microsoft/Phi-3-medium-4k-instruct", # 14B   16 GB+ only
}

TARGET_MODEL = "phi3-mini"
MODEL_ID     = MODELS[TARGET_MODEL]

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR  = PROJECT_ROOT / "llm" / "qlora_dataset"
MLX_DATA_DIR = PROJECT_ROOT / "llm" / "mlx_data"
ADAPTER_DIR  = PROJECT_ROOT / "llm" / "mlx_adapters"
FUSED_DIR    = PROJECT_ROOT / "llm" / "mlx_fused_model"
RESUME_PATH  = PROJECT_ROOT / "resume" / "resume.json"

# ─── Speed Presets ────────────────────────────────────────────────────────────
# Tune these to trade quality vs speed

PRESETS = {
    # Mode         iters  batch  seq_len  lora_layers  lora_rank  est_time
    "fast":       (200,   2,     512,     4,           4,         "8-12 min"),
    "balanced":   (500,   2,     512,     8,           8,         "20-30 min"),
    "standard":   (1000,  2,     1024,    8,           8,         "45-75 min"),
    "quality":    (2000,  2,     1024,    16,          16,        "90-150 min"),
}

DEFAULT_PRESET = "balanced"   # Change to "fast" for quick experiments

# ─── Chat Template Tokens ─────────────────────────────────────────────────────
# Built dynamically to avoid XML parser issues in tooling

def _tok(name: str) -> str:
    """Build a Phi-3 special token like <|name|>."""
    return chr(60) + chr(124) + name + chr(124) + chr(62)


# Phi-3 tokens
PHI3 = {
    "sys_s":  _tok("system"),
    "sys_e":  _tok("end"),
    "usr_s":  _tok("user"),
    "usr_e":  _tok("end"),
    "ast_s":  _tok("assistant"),
    "ast_e":  _tok("end"),
}

# Mistral tokens
MISTRAL = {"inst_s": "[INST]", "inst_e": "[/INST]"}

SYSTEM_MSG = (
    "You are an expert ATS resume optimizer for Mustakim Shaikh. "
    "Rewrite resumes using ONLY real candidate information. "
    "Never invent skills, experience, or achievements."
)


# ─── Formatting ───────────────────────────────────────────────────────────────

def _user_text(instruction: str, user_input: str) -> str:
    return instruction + ("\n\n" + user_input if user_input.strip() else "")


def fmt_phi3(instruction: str, user_input: str, output: str) -> str:
    p = PHI3
    return (
        p["sys_s"] + "\n" + SYSTEM_MSG + "\n" + p["sys_e"] + "\n"
        + p["usr_s"] + "\n" + _user_text(instruction, user_input) + "\n" + p["usr_e"] + "\n"
        + p["ast_s"] + "\n" + output + "\n" + p["ast_e"]
    )


def fmt_mistral(instruction: str, user_input: str, output: str) -> str:
    m = MISTRAL
    user = _user_text(instruction, user_input)
    return f"{m['inst_s']} {user} {m['inst_e']} {output}</s>"


def get_formatter():
    """Return the correct formatter for the current model."""
    return fmt_phi3 if "phi" in TARGET_MODEL.lower() else fmt_mistral


# ─── Dataset Preparation ─────────────────────────────────────────────────────

def prepare_dataset():
    """Convert QLoRA JSONL dataset into MLX-LM chat format."""
    print("\n📦 Preparing MLX dataset...")
    print("=" * 50)

    fmt = get_formatter()
    style = "phi3" if "phi" in TARGET_MODEL.lower() else "mistral"
    print(f"  Model:   {MODEL_ID}")
    print(f"  Format:  {style}")

    train_file = DATASET_DIR / "train.jsonl"
    if not train_file.exists():
        print("\n  ❌ train.jsonl not found!")
        print("     Run first: python llm/qlora_dataset_generator.py")
        return False

    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, fname in [("train", "train.jsonl"), ("valid", "val.jsonl")]:
        src = DATASET_DIR / fname
        if not src.exists():
            print(f"  ⚠️  {fname} missing — skipping {split}")
            continue

        raw = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        out = MLX_DATA_DIR / f"{split}.jsonl"

        with open(out, "w") as f:
            for s in raw:
                text = fmt(
                    s.get("instruction", ""),
                    s.get("input", ""),
                    s.get("output", ""),
                )
                f.write(json.dumps({"text": text}) + "\n")

        # Show token-length statistics
        lengths = [len(s.get("output", "").split()) for s in raw]
        avg_len = sum(lengths) // max(len(lengths), 1)
        print(f"  ✅ {split:6s}: {len(raw):5d} samples | avg output ~{avg_len} words → {out.name}")

    print(f"\n✅ Dataset ready: {MLX_DATA_DIR}")
    return True


# ─── Training ─────────────────────────────────────────────────────────────────

def train(fast: bool = False):
    """
    Run MLX-LM LoRA fine-tuning.

    Speed optimisations applied:
      - Small batch size (RAM-friendly on M2)
      - Gradient checkpointing (halves peak RAM)
      - Shorter sequences (512 tokens by default)
      - Only 4-8 LoRA layers (not all 32)
      - Low LoRA rank (4-8 vs 64)
      - 'fast' preset = 200 iters, usable model in 8-12 min
    """
    preset = "fast" if fast else DEFAULT_PRESET
    iters, batch, seq_len, lora_layers, lora_rank, est_time = PRESETS[preset]

    print(f"\n🏋️  MLX LoRA Training — preset: {preset.upper()}")
    print("=" * 50)
    print(f"  Model:        {MODEL_ID}")
    print(f"  Iters:        {iters}")
    print(f"  Batch size:   {batch}")
    print(f"  Seq length:   {seq_len} tokens")
    print(f"  LoRA layers:  {lora_layers}")
    print(f"  LoRA rank:    {lora_rank}")
    print(f"  Est. time:    {est_time} on Mac M2")
    print()

    if not (MLX_DATA_DIR / "train.jsonl").exists():
        print("  ❌ MLX dataset missing. Run --prepare first.")
        return False

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model",            MODEL_ID,
        "--train",
        "--data",             str(MLX_DATA_DIR),
        "--adapter-path",     str(ADAPTER_DIR),
        "--batch-size",       str(batch),
        "--iters",            str(iters),
        "--learning-rate",    "3e-4",       # Slightly higher LR = faster convergence
        "--lora-layers",      str(lora_layers),
        "--lora-rank",        str(lora_rank),
        "--lora-scale",       "20.0",
        "--steps-per-eval",   str(max(50, iters // 10)),
        "--steps-per-report", "10",
        "--val-batches",      "10",          # Fewer val batches = faster eval
        "--grad-checkpoint",                 # CRITICAL: halves peak RAM on M2
        "--max-seq-length",   str(seq_len),
        "--seed",             "42",
    ]

    print(f"  ▶ Starting training...")
    print(f"  {'━' * 44}")

    start = datetime.now()
    result = subprocess.run(cmd)
    elapsed = (datetime.now() - start).total_seconds() / 60

    if result.returncode == 0:
        print(f"\n  ✅ Training complete in {elapsed:.1f} minutes!")
        print(f"     Adapters saved: {ADAPTER_DIR}")
        print(f"\n  Next steps:")
        print(f"     python llm/mlx_qlora_trainer.py --test")
        print(f"     python llm/mlx_qlora_trainer.py --fuse")
    else:
        print(f"\n  ❌ Training failed (exit code {result.returncode})")
        print("     Ensure mlx-lm is installed: pip install mlx-lm")

    return result.returncode == 0


# ─── Inference Test ───────────────────────────────────────────────────────────

def test_model():
    """Quick inference with the trained LoRA adapters."""
    print("\n🧪 Testing trained adapters...")
    print("=" * 50)

    with open(RESUME_PATH) as f:
        resume = json.load(f)

    summary = resume.get("summary", "")

    # Three test prompts
    test_cases = [
        {
            "role": "AI Engineer (LLM + RAG)",
            "jd":   "Build LLM-powered apps using LangChain, FastAPI, Ollama. RAG experience essential.",
        },
        {
            "role": "ML Engineer (Data Pipelines)",
            "jd":   "Build analytics pipelines, automate reporting, PostgreSQL, Python, Docker.",
        },
        {
            "role": "Backend AI Developer",
            "jd":   "FastAPI backend for AI services, REST APIs, vector DBs, sub-400ms latency.",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] Role: {case['role']}")
        print(f"  JD: {case['jd']}")
        print("  " + "─" * 46)

        prompt = (
            "Tailor this resume summary to match the job description. "
            "Use ONLY real information from the resume.\n\n"
            f"Resume summary: {summary}\n\n"
            f"Job: {case['role']}\nDescription: {case['jd']}"
        )

        cmd = [
            sys.executable, "-m", "mlx_lm.generate",
            "--model",        MODEL_ID,
            "--adapter-path", str(ADAPTER_DIR),
            "--prompt",       prompt,
            "--max-tokens",   "200",
            "--temp",         "0.2",   # Low temp = more deterministic, less hallucination
        ]
        subprocess.run(cmd)
        print()


# ─── Fuse Weights ─────────────────────────────────────────────────────────────

def fuse_model():
    """Fuse LoRA adapters into the base model for standalone use."""
    print("\n🔧 Fusing LoRA into base model...")
    print("=" * 50)

    if not ADAPTER_DIR.exists():
        print("  ❌ No adapters found. Run --train first.")
        return False

    FUSED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model",        MODEL_ID,
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path",    str(FUSED_DIR),
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n  ✅ Fused model: {FUSED_DIR}")

        # Write Ollama Modelfile for local use
        mf = FUSED_DIR / "Modelfile"
        mf.write_text(
            f"FROM {FUSED_DIR}\n"
            f"SYSTEM {SYSTEM_MSG}\n"
        )
        print(f"\n  📦 To use in Ollama (locally):")
        print(f"     ollama create resume-ai -f {mf}")
        print(f"     ollama run resume-ai")
    else:
        print(f"\n  ❌ Fusion failed.")

    return result.returncode == 0


# ─── HuggingFace Hub ─────────────────────────────────────────────────────────

def push_to_hub(repo_id: str):
    """Push fused model to HuggingFace Hub."""
    print(f"\n🚀 Pushing to HuggingFace: {repo_id}")
    print("=" * 50)

    if not FUSED_DIR.exists():
        print("  ❌ No fused model. Run --fuse first.")
        return

    try:
        from huggingface_hub import HfApi, login

        print("  🔑 Logging in to HuggingFace...")
        login()  # Uses HF_TOKEN env var or browser login

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        print("  📤 Uploading...")
        api.upload_folder(
            folder_path=str(FUSED_DIR),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"\n  ✅ Live: https://huggingface.co/{repo_id}")

    except ImportError:
        print("  ❌ Install: pip install huggingface_hub")
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ─── Estimate Training Time ───────────────────────────────────────────────────

def estimate():
    """Print training time and RAM estimates for each preset."""
    print("\n⏱️  Training Estimates for Mac M2")
    print("=" * 60)
    print(f"  {'Preset':<12} {'Iters':<8} {'SeqLen':<8} {'LoRA':<8} {'RAM':<10} {'Time'}")
    print("  " + "─" * 54)

    ram = {
        "fast":     "~4 GB",
        "balanced": "~5 GB",
        "standard": "~7 GB",
        "quality":  "~10 GB",
    }

    for name, (iters, batch, seq_len, lora_layers, lora_rank, est_time) in PRESETS.items():
        marker = " ← default" if name == DEFAULT_PRESET else ""
        print(
            f"  {name:<12} {iters:<8} {seq_len:<8} "
            f"r{lora_rank}/{lora_layers}L   {ram[name]:<10} {est_time}{marker}"
        )

    print()
    print("  To use a preset:")
    print("    python llm/mlx_qlora_trainer.py --train          # balanced (default)")
    print("    python llm/mlx_qlora_trainer.py --train --fast   # 8-12 min test")
    print()
    print("  To change default preset, edit DEFAULT_PRESET in this file.")
    print()
    print("  Speed tips:")
    print("    • --fast flag uses 200 iters, seq=512 — good enough for testing")
    print("    • Close other apps to free unified memory")
    print("    • phi3-mini is 2x faster than mistral-7b")
    print("    • balanced preset gives 90%+ of quality at 40% of the time")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or "--help" in args:
        print("""
MLX QLoRA Trainer — Mac M2 Optimised
======================================
Commands:
  --prepare            Convert dataset to MLX format (run first!)
  --estimate           Show time/RAM estimates for all presets
  --train              Train with balanced preset (~20-30 min)
  --train --fast       Ultra-fast training run (~8-12 min)
  --test               Run inference with trained adapters
  --fuse               Fuse adapters into standalone model
  --push USERNAME/REPO Push fused model to HuggingFace Hub

Full workflow:
  1. python llm/qlora_dataset_generator.py        # generate 2000 samples
  2. python llm/mlx_qlora_trainer.py --prepare    # convert to MLX format
  3. python llm/mlx_qlora_trainer.py --estimate   # check time estimates
  4. python llm/mlx_qlora_trainer.py --train --fast  # quick test first
  5. python llm/mlx_qlora_trainer.py --train      # full quality run
  6. python llm/mlx_qlora_trainer.py --test       # verify output
  7. python llm/mlx_qlora_trainer.py --fuse       # create standalone model
  8. python llm/mlx_qlora_trainer.py --push MustakimShaikh01/resume-optimizer
""")

    elif "--prepare"  in args: prepare_dataset()
    elif "--estimate" in args: estimate()
    elif "--train"    in args: train(fast="--fast" in args)
    elif "--test"     in args: test_model()
    elif "--fuse"     in args: fuse_model()
    elif "--push"     in args:
        idx = args.index("--push")
        repo = args[idx + 1] if idx + 1 < len(args) else None
        if repo:
            push_to_hub(repo)
        else:
            print("Usage: python mlx_qlora_trainer.py --push username/repo-name")
    else:
        print(f"Unknown command: {' '.join(args)}")
        print("Run with --help to see all commands.")


if __name__ == "__main__":
    main()