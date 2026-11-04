#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preflight check for a live adaptive-orchestrator run.

Run this ONCE after Ollama is installed and up. It verifies, without launching a
full attack run:
  1. Ollama is reachable.
  2. The models each role needs are pulled (role models + the COMET pipeline
     models from config/.env + the image model if an Ollama image backend is set).
  3. target/judge/refiner are known to the model registry (else clients fail).
  4. The target looks vision-capable (images are sent to it).
  5. (optional --smoke) a single live chat call to the target actually responds.

Exit code 0 = ready to run; 1 = something needs fixing (it tells you what).

Usage:
  python preflight.py
  python preflight.py --target qwen3-vl:30b-a3b-instruct --judge qwen3-vl:32b \
      --refiner qwen3-vl:30b-a3b-instruct --smoke
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import requests
import config as CFG

OK = "[ OK ]"
MISS = "[MISS]"
WARN = "[WARN]"


def _norm(name: str) -> str:
    return (name or "").strip()


def _installed_models(host: str):
    """Return the set of model tags Ollama has, or None if unreachable."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    names = set()
    for m in data.get("models", []):
        n = m.get("name") or m.get("model")
        if n:
            names.add(n)
    return names


def _present(name: str, installed: set) -> bool:
    name = _norm(name)
    if not name:
        return True  # nothing required
    if name in installed:
        return True
    # lenient :latest matching both directions
    base = name.split(":")[0]
    for inst in installed:
        if inst == name or inst.split(":")[0] == base and (
                name.endswith(":latest") or inst.endswith(":latest")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Preflight for a live adaptive run.")
    ap.add_argument("--target", default="qwen3-vl:30b-a3b-instruct")
    ap.add_argument("--judge", default="qwen3-vl:32b")
    ap.add_argument("--refiner", default="qwen3-vl:30b-a3b-instruct")
    ap.add_argument("--smoke", action="store_true",
                    help="Do one live chat call to the target to confirm it responds.")
    args = ap.parse_args()

    host = CFG.OLLAMA_HOST
    problems = 0

    print(f"\nPreflight -- Ollama at {host}\n" + "-" * 60)

    # 1) connectivity
    installed = _installed_models(host)
    if installed is None:
        print(f"{MISS} Ollama not reachable at {host}")
        print("       Start it (launch the Ollama app, or run `ollama serve`),")
        print("       then re-run:  python preflight.py")
        sys.exit(1)
    print(f"{OK} Ollama reachable -- {len(installed)} model(s) installed")

    # 2) required models by role + pipeline + image
    transmuter_primary = os.getenv("TRANSMUTER_PRIMARY_MODEL", "")
    transmuter_fallback = os.getenv("TRANSMUTER_FALLBACK_MODEL", "")
    required = {
        "target (vision)": args.target,
        "judge": args.judge,
        "refiner": args.refiner,
        "pipeline TEXT_MODEL": CFG.DEFAULT_TEXT_MODEL,
        "pipeline VISION_MODEL": CFG.DEFAULT_VISION_MODEL,
        "pipeline ADVERSARIAL_MODEL": CFG.DEFAULT_ADVERSARIAL_MODEL,
    }
    if transmuter_primary:
        required["transmuter primary"] = transmuter_primary
    if transmuter_fallback:
        required["transmuter fallback"] = transmuter_fallback
    backend = getattr(CFG, "IMAGE_GEN_BACKEND", "local_pillow").lower()
    if backend in ("zimage", "ollama_image", "ollama_zimage"):
        required[f"image model ({backend})"] = CFG.IMAGE_GEN_MODEL

    print("\nRequired models:")
    to_pull = set()
    for role, name in required.items():
        if not _norm(name):
            continue
        if _present(name, installed):
            print(f"  {OK} {role:<28} {name}")
        else:
            print(f"  {MISS} {role:<28} {name}")
            to_pull.add(name)
            problems += 1

    if to_pull:
        print("\nPull the missing models:")
        for n in sorted(to_pull):
            print(f"  ollama pull {n}")
        print("  (Or edit .env / pass flags to use models you already have. If")
        print("   the image model is the only gap, set IMAGE_GEN_BACKEND=local_pillow.)")

    # 3) registry membership for the client-creating roles
    reg = set(CFG.MODEL_REGISTRY.get("ollama", []))
    print("\nModel-registry check (target/judge/refiner must be registered):")
    for role, name in (("target", args.target), ("judge", args.judge), ("refiner", args.refiner)):
        if _norm(name) in reg:
            print(f"  {OK} {role:<8} {name}")
        else:
            print(f"  {WARN} {role:<8} {name} not in MODEL_REGISTRY['ollama']")
            print(f"         add it to LOCAL_MODELS in config.py, or use a registered tag")
            problems += 1

    # 4) vision heuristic for the target
    if "vl" not in args.target.lower():
        print(f"\n{WARN} target '{args.target}' may not be vision-capable; images are sent "
              f"to it.\n       Prefer a *-vl* model (qwen3-vl:30b-a3b-instruct / qwen3-vl:32b).")

    # 5) optional live smoke call
    if args.smoke and _present(args.target, installed):
        print("\nLive smoke call to target ...")
        try:
            from process_utils.strongreject_evaluator import create_model_client
            client = create_model_client(args.target)
            import time
            t0 = time.time()
            resp = client.chat.completions.create(
                model=args.target,
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                temperature=0.0, max_tokens=16,
            )
            txt = (resp.choices[0].message.content or "").strip()
            print(f"  {OK} target responded in {time.time()-t0:.1f}s: {txt[:40]!r}")
        except Exception as e:
            print(f"  {MISS} target call failed: {type(e).__name__}: {e}")
            problems += 1

    print("\n" + "-" * 60)
    if problems == 0:
        print(f"{OK} Preflight passed -- ready to run.\n")
        sys.exit(0)
    print(f"{WARN} {problems} issue(s) above. Fix them, then re-run preflight.\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
