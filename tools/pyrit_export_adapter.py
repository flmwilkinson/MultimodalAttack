#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyRIT -> Robustness-Harness Export Adapter
==========================================

Converts conversations produced by Microsoft PyRIT's multi-turn adaptive orchestrators
(PAIR, TAP / TreeOfAttacksWithPruning, Crescendo, RedTeamingOrchestrator, ...) into the
JSONL transcript schema consumed by `process_utils/robustness_eval.py`.

Scope / stance
--------------
This is a DATA ADAPTER only. It reads attack conversations that PyRIT already ran under YOUR
authorization and reshapes them for scoring. It does not run PyRIT, does not generate attacks,
and does not implement any attack algorithm. The human sets the objective inside PyRIT; this
tool carries that objective and the captured turns across into the harness.

Two input modes
---------------
1. `--from-json pieces.json`  (recommended, no PyRIT import needed)
   A JSON array of PyRIT "prompt request piece" records exported from PyRIT memory, e.g.:
       memory = CentralMemory.get_memory_instance()
       pieces = memory.get_prompt_request_pieces()          # your PyRIT version's accessor
       json.dump([p.to_dict() for p in pieces], open("pieces.json","w"))
   Each record is expected to carry at least: conversation_id, role, a value, a data type,
   and a sequence index. Field names vary across PyRIT versions -- see FIELD MAP below and
   adjust `--field-*` options if yours differ.

2. `--from-duckdb path.db`  (optional, requires `pyrit` installed)
   Reads directly from a PyRIT DuckDB memory file. This path uses defensive attribute access
   because PyRIT's API has changed across 0.x releases; if it fails, export to JSON (mode 1).

Output
------
JSONL, one conversation per line, matching robustness_eval's loader:
    {"id","objective","original_query","attack_family","turns":[{"text","image_path"}],
     "attacker_metadata":{"orchestrator","captured_responses","pyrit_scores","num_attacker_turns"}}

FIELD MAP (defaults; override with --field-* if your PyRIT version differs)
---------------------------------------------------------------------------
    role              <- "role"                      ("user" = attacker, "assistant" = target)
    value             <- "converted_value" or "original_value"
    data_type         <- "converted_value_data_type" or "original_value_data_type"
    conversation_id   <- "conversation_id"
    sequence          <- "sequence"
    orchestrator      <- "orchestrator_identifier"   (dict; __type__ becomes attack_family)
    labels            <- "labels"                    (objective read from labels["objective"])
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Any


def _first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _family_from_orchestrator(orch: Any) -> str:
    """Map a PyRIT orchestrator identifier to an attack_family label."""
    name = None
    if isinstance(orch, dict):
        name = orch.get("__type__") or orch.get("type") or orch.get("name")
    elif isinstance(orch, str):
        name = orch
    if not name:
        return "pyrit_multiturn"
    low = name.lower()
    if "crescendo" in low:
        return "pyrit_crescendo"
    if "pair" in low:
        return "pyrit_pair"
    if "treeofattack" in low or "tap" in low:
        return "pyrit_tap"
    if "redteam" in low:
        return "pyrit_redteam"
    # Fall back to a normalized form of the class name
    return "pyrit_" + "".join(c if c.isalnum() else "_" for c in low).strip("_")


def _group_pieces(pieces: List[Dict[str, Any]], fmap: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    convos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pieces:
        cid = _first_present(p, fmap["conversation_id"], "unknown_conversation")
        convos[str(cid)].append(p)
    # Sort each conversation by sequence (fall back to original order)
    for cid, items in convos.items():
        items.sort(key=lambda x: _first_present(x, fmap["sequence"], 0) or 0)
    return convos


def _piece_value(p: Dict[str, Any], fmap: Dict[str, Any]) -> str:
    return str(_first_present(p, fmap["value"], "") or "")


def _piece_datatype(p: Dict[str, Any], fmap: Dict[str, Any]) -> str:
    return str(_first_present(p, fmap["data_type"], "text") or "text")


def _piece_role(p: Dict[str, Any], fmap: Dict[str, Any]) -> str:
    return str(_first_present(p, fmap["role"], "user") or "user").lower()


def convert_conversations(
    pieces: List[Dict[str, Any]],
    fmap: Dict[str, Any],
    default_objective: Optional[str],
    scores_by_convo: Optional[Dict[str, Any]] = None,
    image_base: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Reshape a flat list of PyRIT pieces into harness transcript records."""
    convos = _group_pieces(pieces, fmap)
    out: List[Dict[str, Any]] = []

    for cid, items in convos.items():
        # Collect attacker (user-role) turns, merging text + image pieces at the same sequence.
        turns_by_seq: Dict[Any, Dict[str, Any]] = {}
        captured_responses: List[str] = []
        orchestrator = None
        # A per-conversation objective recorded in PyRIT labels takes priority; the CLI
        # --objective is only a fallback when no label is present.
        objective = None

        for p in items:
            role = _piece_role(p, fmap)
            if orchestrator is None:
                orchestrator = _first_present(p, fmap["orchestrator"], None)
            labels = _first_present(p, fmap["labels"], {}) or {}
            if isinstance(labels, dict) and labels.get("objective") and not objective:
                objective = labels.get("objective")

            if role == "user":
                seq = _first_present(p, fmap["sequence"], len(turns_by_seq))
                turn = turns_by_seq.setdefault(seq, {"text": "", "image_path": None})
                dtype = _piece_datatype(p, fmap)
                val = _piece_value(p, fmap)
                if dtype == "image_path":
                    path = val
                    if image_base and path and not os.path.isabs(path):
                        path = os.path.join(image_base, path)
                    turn["image_path"] = path
                else:
                    turn["text"] = (turn["text"] + ("\n" if turn["text"] else "") + val).strip()
            elif role == "assistant":
                captured_responses.append(_piece_value(p, fmap))
            # system-role pieces are ignored (the harness applies the DEFENDED system prompt itself)

        # Fall back to the CLI-provided objective only if no per-conversation label was found.
        objective = objective or default_objective

        turns = [turns_by_seq[k] for k in sorted(turns_by_seq.keys(), key=lambda x: (str(type(x)), x))]
        # Drop empty turns (no text and no image)
        turns = [t for t in turns if t["text"] or t["image_path"]]
        if not turns:
            print(f"[warn] conversation {cid}: no attacker turns found; skipping.")
            continue

        record = {
            "id": cid,
            "objective": objective or "(objective not recorded; set via --objective or PyRIT labels)",
            "original_query": objective or "",
            "attack_family": _family_from_orchestrator(orchestrator),
            "turns": turns,
            "attacker_metadata": {
                "orchestrator": orchestrator,
                "num_attacker_turns": len(turns),
                "captured_responses": captured_responses,
                "pyrit_scores": (scores_by_convo or {}).get(str(cid)),
            },
        }
        out.append(record)

    return out


def load_pieces_from_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "pieces" in data:
        data = data["pieces"]
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of PyRIT piece records (or {'pieces': [...]}).")
    return data


def load_pieces_from_duckdb(path: str) -> List[Dict[str, Any]]:
    """Best-effort direct read from a PyRIT DuckDB memory file.

    PyRIT's API varies by version; this uses defensive access. If it fails, export your memory
    to JSON and use --from-json instead.
    """
    try:
        from pyrit.memory import DuckDBMemory  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import PyRIT (pyrit.memory.DuckDBMemory). Install pyrit, or export your "
            f"memory to JSON and use --from-json. Underlying error: {e}"
        )
    mem = DuckDBMemory(db_path=path)
    try:
        pieces = mem.get_prompt_request_pieces()  # accessor name may differ by version
    except Exception as e:
        raise RuntimeError(
            "PyRIT memory loaded but get_prompt_request_pieces() failed (API differs by "
            f"version). Export to JSON and use --from-json. Underlying error: {e}"
        )
    out = []
    for p in pieces:
        if hasattr(p, "to_dict"):
            out.append(p.to_dict())
        else:
            out.append({k: getattr(p, k, None) for k in (
                "role", "original_value", "converted_value",
                "original_value_data_type", "converted_value_data_type",
                "conversation_id", "sequence", "orchestrator_identifier", "labels")})
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert PyRIT conversations to harness transcript JSONL (data adapter only).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-json", help="JSON array of exported PyRIT piece records.")
    src.add_argument("--from-duckdb", help="Path to a PyRIT DuckDB memory file (requires pyrit).")
    ap.add_argument("--output", required=True, help="Output JSONL path for robustness_eval --transcripts.")
    ap.add_argument("--objective", default=None, help="Default human-set objective if not in PyRIT labels.")
    ap.add_argument("--image-base", default=None, help="Base dir to resolve relative image paths against.")
    # Field-map overrides (defaults match common PyRIT exports)
    ap.add_argument("--field-role", default="role")
    ap.add_argument("--field-value", default="converted_value,original_value,value")
    ap.add_argument("--field-datatype", default="converted_value_data_type,original_value_data_type,data_type")
    ap.add_argument("--field-conversation", default="conversation_id")
    ap.add_argument("--field-sequence", default="sequence")
    ap.add_argument("--field-orchestrator", default="orchestrator_identifier,orchestrator")
    ap.add_argument("--field-labels", default="labels")
    args = ap.parse_args()

    fmap = {
        "role": [args.field_role],
        "value": args.field_value.split(","),
        "data_type": args.field_datatype.split(","),
        "conversation_id": [args.field_conversation],
        "sequence": [args.field_sequence],
        "orchestrator": args.field_orchestrator.split(","),
        "labels": [args.field_labels],
    }

    if args.from_json:
        pieces = load_pieces_from_json(args.from_json)
    else:
        pieces = load_pieces_from_duckdb(args.from_duckdb)

    records = convert_conversations(pieces, fmap, args.objective, image_base=args.image_base)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    families = sorted({r["attack_family"] for r in records})
    print(f"[done] wrote {len(records)} conversations to {args.output}")
    print(f"[info] attack families present: {families}")
    print(f"[next] feed to the harness:\n"
          f"       python process_utils/robustness_eval.py --targets targets.example.json "
          f"--transcripts {args.output} --judge-model <judge> --output reports/robustness_report")


if __name__ == "__main__":
    main()
