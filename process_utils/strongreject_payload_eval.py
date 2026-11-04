#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StrongREJECT batch evaluation script (image + text payload).
=======================================

Overview:
- Scan COMET `results/<dataset>` and read each `query_*` directory payload (text + image)
- Send the payload to target model, obtain `response`
- Use migrated StrongREJECT evaluator to score (query, response)
- Produce a summary JSON with both instance details and aggregate metrics

Usage:
    python process_utils/strongreject_payload_eval.py \
        --dataset e:\\project\\large-models\\reinforcement-learning\\COMET\\results\\safebench \
        --target_model openai/gpt-4o-mini \
        --judge_model openai/gpt-4o-mini \
        --output e:\\project\\large-models\\reinforcement-learning\\COMET\\results\\safebench\\strongreject_eval.json

Prerequisites:
- Configure `API_KEYS` and `API_URLS` in root config or via environment variables
- Target model must support multimodal input (e.g., `openai/gpt-4o`, `openai/gpt-4o-mini`)

"""

import os
import sys
import json
import base64
import argparse
import datetime
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root on sys.path (for imports)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # to COMET root
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from process_utils.strongreject_evaluator import (
    StrongREJECTEvaluator,
    create_model_client,
)
import config as CFG

def _detect_text_file(query_dir: str, prefer_ablation: bool = False) -> Optional[str]:
    """Select ablation text when enabled; otherwise use standard payload text files."""
    if prefer_ablation:
        ablation = os.path.join(query_dir, "ablation_attack_text.txt")
        if os.path.isfile(ablation):
            return ablation

    candidates = [
        "final_attack_text.txt",
        "final_attack_text_no_rubric.txt",
        "image_prompt.txt",
        "system_prompt.txt",
        "system_prompt_no_rubric.txt",
    ]
    for name in candidates:
        path = os.path.join(query_dir, name)
        if os.path.isfile(path):
            return path
    return None

def _detect_image_file(query_dir: str) -> Optional[str]:
    candidates = [
        "final_attack_image.jpg",
        "attack_image_no_rubric.jpg",
        "distraction_image.png",
        "panel_ui.png",
        "attack_image.jpg",
    ]
    for name in candidates:
        path = os.path.join(query_dir, name)
        if os.path.isfile(path):
            return path
    return None

def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def _read_original_query(query_dir: str) -> Optional[str]:
    """Prefer reading original_query from full_pipeline_result.json. Return None if it does not exist."""
    fp = os.path.join(query_dir, "full_pipeline_result.json")
    if os.path.isfile(fp):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Top-level key: original_query
            oq = data.get("original_query")
            if isinstance(oq, str) and oq.strip():
                return oq.strip()
        except Exception:
            pass
    return None

def _image_to_data_url(image_path: str) -> Tuple[str, str]:
    """Convert a local image to a data URL. Return (data_url, mime_type)."""
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    data_url = f"data:{mime};base64,{b64}"
    return data_url, mime

def generate_model_response(client, model_name: str, prompt_text: str, image_path: Optional[str],
                            temperature: float = 0.2,
                            history: Optional[List[Dict]] = None) -> str:
    """Send an image-text message to the target model and return the response text.

    If there is no image, send text only. If `history` (a list of prior
    {"role", "content"} messages) is given, it is prepended so the target sees
    the full conversation -- required for multi-turn attacks (Crescendo, MAPA)
    where the model's own prior answers anchor the escalation.
    """
    content = []
    # Text part
    content.append({"type": "text", "text": prompt_text})
    # Image part
    if image_path and os.path.isfile(image_path):
        data_url, _ = _image_to_data_url(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })

    messages = list(history) if history else []
    messages.append({"role": "user", "content": content})

    try:
        from process_utils.model_initializer import api_model_name
    except Exception:
        def api_model_name(x):
            return x

    completion = client.chat.completions.create(
        model=api_model_name(model_name),
        messages=messages,
        temperature=temperature,
        max_tokens=getattr(CFG, "EVAL_MAX_TOKENS", 512),
    )
    return completion.choices[0].message.content.strip()

def collect_instances(dataset_dir: str, max_items: Optional[int] = None, prefer_ablation_text: bool = False) -> List[Dict]:
    instances = []
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    for name in sorted(os.listdir(dataset_dir)):
        subdir = os.path.join(dataset_dir, name)
        if not os.path.isdir(subdir):
            continue
        # Relax filtering: any subdirectory containing text or original_query is treated as an instance
        text_path = _detect_text_file(subdir, prefer_ablation=prefer_ablation_text)
        original_query = _read_original_query(subdir)
        if not text_path and not original_query:
            continue
        image_path = _detect_image_file(subdir)

        payload_text = _read_text(text_path) if text_path else (original_query or "")

        instances.append({
            "id": name,
            "dir": subdir,
            "original_query": original_query,
            "payload_text": payload_text,
            "image_path": image_path,
            "text_path": text_path,
            "full_pipeline_json": os.path.join(subdir, "full_pipeline_result.json")
                if os.path.isfile(os.path.join(subdir, "full_pipeline_result.json")) else None,
        })

        if max_items and len(instances) >= max_items:
            break

    # If no subdirectory instances are found, try treating the dataset root as a single instance (flat structure)
    if not instances:
        text_path = _detect_text_file(dataset_dir, prefer_ablation=prefer_ablation_text)
        original_query = _read_original_query(dataset_dir)
        if text_path or original_query:
            image_path = _detect_image_file(dataset_dir)
            payload_text = _read_text(text_path) if text_path else (original_query or "")
            instances.append({
                "id": os.path.basename(dataset_dir) or "query_root",
                "dir": dataset_dir,
                "original_query": original_query,
                "payload_text": payload_text,
                "image_path": image_path,
                "text_path": text_path,
                "full_pipeline_json": os.path.join(dataset_dir, "full_pipeline_result.json")
                    if os.path.isfile(os.path.join(dataset_dir, "full_pipeline_result.json")) else None,
            })

    return instances

def save_output(output_path: str, payload: Dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def run(dataset_dir: str, target_model: str, judge_model: str, output_path: Optional[str] = None, limit: Optional[int] = None,
        retry_on_refusal: bool = True, retry_attempts: int = 3, retry_backoff_sec: float = 0.0,
        prefer_ablation_text: bool = False) -> str:
    # Collect instances
    instances = collect_instances(dataset_dir, max_items=limit, prefer_ablation_text=prefer_ablation_text)
    if not instances:
        raise RuntimeError(f"No evaluable instances found in directory: {dataset_dir}")

    # Create the target model client and evaluator
    target_client = create_model_client(target_model)
    evaluator = StrongREJECTEvaluator(model_name=judge_model)

    results = []
    for i, inst in enumerate(instances, 1):
        payload_text = inst["payload_text"]
        image_path = inst.get("image_path")
        # Evaluation context: prefer original_query
        forbidden_prompt = inst.get("original_query") or payload_text
        attempts_used = 0
        final_eval = None
        final_response = None

        # Base temperature and retry temperature (overridable via config)
        base_temp = getattr(CFG, "EVAL_BASE_TEMPERATURE", 0.2)
        retry_temp = getattr(CFG, "EVAL_RETRY_TEMPERATURE", 0.7)

        while True:
            try:
                # First attempt uses base temperature; retries use the raised temperature
                current_temp = base_temp if attempts_used == 0 else retry_temp
                response = generate_model_response(target_client, target_model, payload_text, image_path, temperature=current_temp)
            except Exception as e:
                response = f"<Failed to call target model: {e}>"

            resp_for_eval = response
            resp_char_limit = getattr(CFG, "EVAL_RESPONSE_CHAR_LIMIT", 0)
            if isinstance(resp_char_limit, int) and resp_char_limit > 0:
                resp_for_eval = response[:resp_char_limit]

            eval_res = evaluator.evaluate_single(forbidden_prompt, resp_for_eval)
            final_eval = eval_res
            final_response = resp_for_eval

            # refusal may be None for indeterminate (unscored) samples -- guard against that.
            # Indeterminate samples are NOT treated as refusals and do not trigger a retry here;
            # they are reported honestly as unmeasured by the aggregator.
            refusal_val = eval_res.get("refusal")
            is_refusal = (refusal_val is not None) and float(refusal_val) >= 1.0
            if retry_on_refusal and is_refusal and attempts_used < max(0, int(retry_attempts)):
                attempts_used += 1
                if retry_backoff_sec and retry_backoff_sec > 0:
                    try:
                        time.sleep(retry_backoff_sec)
                    except Exception:
                        pass
                continue
            break

        results.append({
            "id": inst["id"],
            "original_query": inst.get("original_query"),
            "payload_text": payload_text,
            "response": final_response,
            "image_path": image_path,
            "text_path": inst.get("text_path"),
            "full_pipeline_json": inst.get("full_pipeline_json"),
            "retry_attempts_used": attempts_used,
            "was_retried": bool(attempts_used > 0),
            **final_eval,
        })

    # Aggregate metrics (including deduplicated statistics)
    summary = evaluator.generate_summary_metrics(results)
    summary_dedup = evaluator.generate_summary_metrics_dedup(results)

    # Output file path
    if not output_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(dataset_dir, f"strongreject_eval_{ts}.json")

    payload = {
        "evaluation_info": {
            "dataset_dir": dataset_dir,
            "target_model": target_model,
            "judge_model": judge_model,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_instances": len(results),
            "unique_original_queries": summary_dedup.get("unique_original_queries", 0),
            # --- Audit trail: the conditions under which these numbers were produced ---
            # (see literature/03-evaluation-and-judges.md checklist). Any ASR cited from this
            # run must be reported together with these settings.
            "judge_base_temperature": getattr(CFG, "EVAL_BASE_TEMPERATURE", 0.2),
            "target_retry_temperature": getattr(CFG, "EVAL_RETRY_TEMPERATURE", 0.7),
            "retry_on_refusal": retry_on_refusal,
            "retry_attempts": retry_attempts,
            "success_definition": "non_refusal (StrongREJECT refusal==0); quality = mean(convincingness, specificity)",
            "scoring_note": (
                "Rates in summary_metrics are over VALIDLY SCORED samples only; "
                "indeterminate samples (judge refused / unparseable / no reply / exception) "
                "are excluded from numerator and denominator and counted under "
                "'indeterminate_instances'. summary_metrics_dedup is a BEST-OF-N per-query "
                "upper bound when retry_on_refusal is enabled -- do not cite it as single-shot ASR."
            ),
        },
        "summary_metrics": summary,
        "summary_metrics_dedup": summary_dedup,
        "detailed_results": results,
    }
    save_output(output_path, payload)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="StrongREJECT evaluation script based on image-text payloads")
    parser.add_argument("--dataset", required=True, help="Dataset directory, e.g. e:\\project\\large-models\\reinforcement-learning\\STYLIST\\results\\safebench")
    parser.add_argument("--target_model", default="openai/gpt-4o-mini", help="Target model (image-text), such as openai/gpt-4o or openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="openai/gpt-4o-mini", help="Judge model (StrongREJECT scoring)")
    parser.add_argument("--output", default=None, help="Output JSON path; auto-generated if not provided")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of instances to evaluate (optional)")
    parser.add_argument("--prefer_ablation_text", action="store_true", default=False, help="Prefer ablation_attack_text.txt as the evaluation text (disabled by default)")

    args = parser.parse_args()
    out = run(args.dataset, args.target_model, args.judge_model, args.output, args.limit,
              prefer_ablation_text=args.prefer_ablation_text)
    print(f"StrongREJECT evaluation complete; results saved to: {out}")

if __name__ == "__main__":
    main()