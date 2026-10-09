#!/usr/bin/env python3
"""BFCL V4 offline tool-search benchmark via OpenAI Python SDK.

Downloads public BFCL files when absent; no tool execution. Three architectures.
The generated arguments are retained for inspection but are NOT scored as
search results or against execution ground truth.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import time
import tempfile
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Any

import numpy as np
try:
    from openai import OpenAI
except ImportError:  # Allow --download-only / --help without the OpenAI SDK.
    OpenAI = None

DOC_CLASS = {
    "gorilla_file_system.json": "GorillaFileSystem",
    "math_api.json": "MathAPI",
    "message_api.json": "MessageAPI",
    "posting_api.json": "TwitterAPI",
    "ticket_api.json": "TicketAPI",
    "trading_bot.json": "TradingBot",
    "travel_booking.json": "TravelAPI",
    "vehicle_control.json": "VehicleControlAPI",
    "web_search.json": "WebSearchAPI",
    "memory_kv.json": "MemoryAPI_kv",
    "memory_vector.json": "MemoryAPI_vector",
    "memory_rec_sum.json": "MemoryAPI_rec_sum",
}

BFCL_RAW_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)
BFCL_TASK_FILES = (
    "BFCL_v4_parallel_multiple.json",
    "BFCL_v4_multi_turn_base.json",
    "possible_answer/BFCL_v4_parallel_multiple.json",
    "possible_answer/BFCL_v4_multi_turn_base.json",
)
BFCL_DOC_FILES = (
    "gorilla_file_system.json", "math_api.json", "message_api.json",
    "posting_api.json", "ticket_api.json", "trading_bot.json",
    "travel_booking.json", "vehicle_control.json",
)
BFCL_DOWNLOAD_FILES = BFCL_TASK_FILES + tuple(
    "multi_turn_func_doc/" + name for name in BFCL_DOC_FILES
)


def download_bfcl_dataset(root: Path, base_url: str = BFCL_RAW_BASE,
                          force: bool = False, timeout: float = 30.0,
                          retries: int = 3) -> list[dict]:
    """Download only missing/invalid files, atomically, from the official repo.

    Existing valid files are preserved unless force=True. JSON/JSONL is checked
    before replacing a file, so a GitHub error page cannot poison the dataset.
    """
    manifest = []
    root.mkdir(parents=True, exist_ok=True)
    for rel in BFCL_DOWNLOAD_FILES:
        dest = root / rel
        url = base_url.rstrip("/") + "/" + rel
        valid = False
        if dest.is_file() and dest.stat().st_size > 0:
            try:
                valid = len(read_records(dest)) > 0
            except (ValueError, UnicodeError, json.JSONDecodeError):
                print(f"Invalid local JSONL; downloading again: {rel}", flush=True)
        if valid and not force:
            manifest.append({"file": rel, "action": "cached", "size": dest.stat().st_size})
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, retries + 1):
            tmp_path = None
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "bfcl-router-benchmark/2.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    with tempfile.NamedTemporaryFile(
                            dir=dest.parent, prefix=".bfcl_", delete=False) as f:
                        tmp_path = Path(f.name)
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                if not read_records(tmp_path):
                    raise ValueError(f"Downloaded file was empty: {url}")
                tmp_path.replace(dest)
                manifest.append({"file": rel, "action": "downloaded", "size": dest.stat().st_size})
                print(f"Downloaded {rel} ({dest.stat().st_size:,} bytes)", flush=True)
                break
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                last_error = exc
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                if attempt < retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        else:
            raise RuntimeError(
                f"BFCL automatic download failed for {url} after {retries} attempts. "
                f"Check network access or use --no-download with locally cached data. "
                f"Original error: {last_error}") from last_error
    return manifest


def read_records(file: Path) -> list[dict]:
    if not file.exists():
        raise FileNotFoundError(f"Missing local file: {file}")
    content = file.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    if content[0] == "[":
        result = json.loads(content)
    else:
        result = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not isinstance(result, list) or not all(isinstance(x, dict) for x in result):
        raise ValueError(f"Expected JSON/JSONL objects in {file}")
    return result


def parse_gt_call(call: Any) -> str | None:
    if isinstance(call, dict):
        if "name" in call:
            return str(call["name"])
        if len(call) == 1:
            return str(next(iter(call)))
    if isinstance(call, str):
        m = re.match(r"\s*([\w.]+)\s*\(", call)
        if m:
            return m.group(1)
    return None


def turns_of(question: Any) -> list[str]:
    """BFCL question: list[list[chat_message]]; plain messages also supported."""
    if isinstance(question, str):
        return [question]
    if not isinstance(question, list):
        return []
    output = []
    for turn in question:
        messages = turn if isinstance(turn, list) else [turn]
        parts = []
        for item in messages:
            if isinstance(item, dict) and item.get("role", "user") == "user":
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        output.append("\n".join(parts))
    return output


@dataclass
class Example:
    id: str
    split: str
    query: str
    history: list[str]
    gold: list[str]


def tidy_params(p: Any) -> dict:
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except (TypeError, ValueError):
            p = {}
    return p if isinstance(p, dict) else {}


def normalize_tool_definition(doc: dict, cls: str | None = None) -> dict | None:
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        return None
    if cls and "." not in name:
        name = cls + "." + name
    return {
        "name": name,
        "description": re.sub(r"\s+", " ", str(doc.get("description", ""))).strip(),
        "parameters": tidy_params(doc.get("parameters", {})),
    }


def catalog_from_sources(single_records: list[dict], docs_dir: Path) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for row in single_records:
        for t in row.get("function", []):
            if isinstance(t, dict):
                tool = normalize_tool_definition(t)
                if tool:
                    old = catalog.get(tool["name"])
                    if not old or len(tool["description"]) > len(old["description"]):
                        catalog[tool["name"]] = tool
    if not docs_dir.exists() or not list(docs_dir.glob("*.json")):
        raise FileNotFoundError(f"Download the multi_turn_func_doc/*.json files into {docs_dir}")
    for path in sorted(docs_dir.glob("*.json")):
        cls = DOC_CLASS.get(path.name)
        if not cls:
            print(f"WARNING: unknown tool-document filename skipped: {path.name}")
            continue
        for t in read_records(path):
            tool = normalize_tool_definition(t, cls)
            if tool:
                catalog[tool["name"]] = tool
    return dict(sorted(catalog.items()))


def gt_multi_to_name(name: str, involved: list[str], catalog: dict[str, dict]) -> str | None:
    # Multi-turn BFCL answers are often unqualified method names. Prefer
    # class-qualified definitions even if a single-turn API shares the name.
    if "." in name and name in catalog:
        return name
    valid = [f"{cls}.{name}" for cls in involved if f"{cls}.{name}" in catalog]
    if len(valid) == 1:
        return valid[0]
    if name in catalog and not valid:
        return name
    # Important: Never disambiguate using the gold `path` because that could
    # leak target information into the catalog-selection mechanism.
    return None


def create_examples(root: Path, seed: int, n_single: int, n_multi: int,
                    catalog: dict[str, dict], multi_only_multilabel: bool = False
                    ) -> tuple[list[Example], list[dict]]:
    sp = root / "BFCL_v4_parallel_multiple.json"
    mp = root / "BFCL_v4_multi_turn_base.json"
    sa = root / "possible_answer" / sp.name
    ma = root / "possible_answer" / mp.name
    single = read_records(sp)
    multi = read_records(mp)
    single_ans = {x["id"]: x.get("ground_truth", []) for x in read_records(sa)}
    multi_ans = {x["id"]: x.get("ground_truth", []) for x in read_records(ma)}
    rng = random.Random(seed)
    rng.shuffle(single)
    rng.shuffle(multi)
    examples = []
    skips = []
    for row in single:
        ans = single_ans.get(row["id"])
        if ans is None:
            skips.append({"id": row["id"], "reason": "missing answer"})
            continue
        gold = sorted(set(filter(None, (parse_gt_call(x) for x in ans))))
        if len(gold) < 2:  # Multi-label cases only, no same-tool repeats
            continue
        if not set(gold) <= catalog.keys():
            skips.append({"id": row["id"], "reason": "missing tool definitions", "gold": gold})
            continue
        q = turns_of(row.get("question", []))
        if q and q[0].strip():
            examples.append(Example(row["id"], "single", q[0], [], gold))
        if sum(x.split == "single" for x in examples) >= n_single:
            break
    multi_added = 0
    for row in multi:
        seq = multi_ans.get(row["id"])
        qs = turns_of(row.get("question", []))
        if not isinstance(seq, list) or len(qs) != len(seq):
            skips.append({"id": row["id"], "reason": "multi length/missing answer mismatch"})
            continue
        involved = row.get("involved_classes", [])
        candidates = []
        for i in range(1, len(qs)):  # use at least one preceding user turn
            calls = seq[i]
            if not isinstance(calls, list):
                continue
            raw_names = set(filter(None, (parse_gt_call(c) for c in calls)))
            resolved = [gt_multi_to_name(x, involved, catalog) for x in raw_names]
            if not raw_names or any(x is None for x in resolved):
                continue
            tools = sorted(set(resolved))
            if multi_only_multilabel and len(tools) < 2:
                continue
            if not qs[i].strip():
                continue
            candidates.append((i, tools))
        if not candidates:
            skips.append({"id": row["id"], "reason": "no resolvable multi-turn target"})
            continue
        # Prefer turns requiring multiple DISTINCT tools; tie prefers later turn.
        i, gold = max(candidates, key=lambda v: (len(v[1]), v[0]))
        examples.append(Example(f"{row['id']}:turn{i}", "multi", qs[i], qs[:i], gold))
        multi_added += 1
        if multi_added >= n_multi:
            break
    return examples, skips


def compact_schema(tool: dict, max_desc: int = 160, max_param_desc: int = 60) -> dict:
    pars = tool.get("parameters", {})
    props = pars.get("properties", {}) if isinstance(pars, dict) else {}
    if not isinstance(props, dict):
        props = {}
    return {
        "name": tool["name"],
        "description": tool["description"][:max_desc],
        "parameters": {
            key: {
                "type": (val.get("type", "string") if isinstance(val, dict) else "string"),
                "description": (str(val.get("description", ""))[:max_param_desc]
                                if isinstance(val, dict) else ""),
            }
            for key, val in props.items()
        },
        "required": pars.get("required", []) if isinstance(pars, dict) else [],
    }


def menu(catalog: dict[str, dict], names: list[str]) -> str:
    return json.dumps([compact_schema(catalog[x]) for x in names], ensure_ascii=False)


def parse_json_object(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        l, r = s.find("{"), s.rfind("}")
        if l == -1 or r <= l:
            raise
        parsed = json.loads(s[l:r + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Response must be JSON object")
    return parsed


class LLM:
    def __init__(self, args: argparse.Namespace):
        self.client = OpenAI(base_url=args.chat_url, api_key=args.chat_key,
                             timeout=args.timeout, max_retries=0)
        self.model = args.chat_model
        self.temperature = args.temperature
        self.reasoning_effort = args.reasoning_effort
        self.max_tokens = args.max_tokens
        self.calls = 0

    def json_call(self, system: str, payload: dict) -> dict:
        prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        last_err = None
        for retry in range(2):
            self.calls += 1
            try:
                options: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system + "\nReturn only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                }
                if self.model.startswith("gpt-6"):
                    # GPT-6 Luna supports reasoning_effort=none. When the effort
                    # is non-none, temperature is not supported by this model.
                    options.update({
                        "max_completion_tokens": self.max_tokens,
                        "reasoning_effort": self.reasoning_effort,
                        "response_format": {"type": "json_object"},
                    })
                    if self.reasoning_effort == "none" and self.temperature is not None:
                        options["temperature"] = self.temperature
                else:
                    options["max_tokens"] = self.max_tokens
                    if self.temperature is not None:
                        options["temperature"] = self.temperature
                out = self.client.chat.completions.create(**options)
                return parse_json_object(out.choices[0].message.content or "")
            except (ValueError, json.JSONDecodeError, IndexError, AttributeError) as exc:
                last_err = exc
                if retry == 0:
                    system += "\nIMPORTANT: Your previous response was malformed. Output a compact JSON object only."
            # API/network/server errors deliberately propagate instead of silently scoring zero.
        raise RuntimeError(f"LLM JSON parsing failed: {last_err}")


class EmbeddingIndex:
    def __init__(self, args: argparse.Namespace, catalog: dict[str, dict], outdir: Path):
        self.client = OpenAI(base_url=args.embed_url, api_key=args.embed_key,
                             timeout=args.timeout, max_retries=0)
        self.model = args.embed_model
        self.names = list(catalog)
        self.docs = [self.text(catalog[n]) for n in self.names]
        self.batch_size = args.embed_batch
        self.calls = 0
        key = hashlib.sha256(json.dumps(
            {"model": self.model, "url": args.embed_url, "documents": self.docs},
            ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:18]
        cached = outdir / f"catalog_embeddings_{key}.npz"
        if cached.exists():
            self.vectors = np.load(cached)["vectors"]
            print(f"Loaded cached catalog embeddings: {cached}")
        else:
            print(f"Embedding {len(self.docs)} catalog tools...", flush=True)
            self.vectors = self.embed(self.docs)
            np.savez_compressed(cached, vectors=self.vectors)
        self.vectors = self.normalize(self.vectors)

    @staticmethod
    def text(tool: dict) -> str:
        props = tool.get("parameters", {}).get("properties", {})
        if not isinstance(props, dict):
            props = {}
        return f"{tool['name']}. {tool['description'][:550]}. Parameters: {', '.join(props)}"

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            self.calls += 1
            response = self.client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return np.asarray(vectors, dtype=np.float32)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        query_vector = self.normalize(self.embed([query]))[0]
        scores = self.vectors @ query_vector
        top = np.argsort(-scores)[:min(k, len(self.names))]
        return [(self.names[i], float(scores[i])) for i in top]


def sanitize_calls(response: dict, allowed: set[str]) -> tuple[list[str], list[dict], list[str]]:
    raw = response.get("tool_calls", [])
    if not isinstance(raw, list):
        return [], [], ["<malformed tool_calls>"]
    names: list[str] = []
    calls = []
    invalid = []
    for obj in raw:
        if not isinstance(obj, dict):
            invalid.append("<malformed call>")
            continue
        name = obj.get("tool") or obj.get("name")
        if not isinstance(name, str) or name not in allowed:
            invalid.append(str(name))
            continue
        arg = obj.get("arguments", {})
        if not isinstance(arg, dict):
            arg = {}
        q = obj.get("search_query", "")
        if not isinstance(q, str):
            q = ""
        calls.append({"tool": name, "arguments": arg, "search_query": q})
        names.append(name)
    return sorted(set(names)), calls, invalid


def selection_system() -> str:
    return ("You are a multi-label tool router. Select ALL necessary distinct tools, "
            "and NO unnecessary tools. Tool names must exactly match catalog names. "
            "Do not call tools. Never use hidden or outside tools. Output {\"tools\": [\"...\"]}.")


def joint_system() -> str:
    return ("You are a joint multi-tool router and tool-specific argument generator. "
            "Choose ALL needed tools among the provided catalog, and for each create "
            "arguments using the listed parameter schema. Some tools need entity "
            "extraction, others need search query rewriting; follow each tool's description. "
            "Add a search_query string for a tool only when it meaningfully searches. "
            "Use only the given context, don't invent entities; don't execute tools. "
            "Output {\"tool_calls\": [{\"tool\":\"exact_name\",\"arguments\":{},\"search_query\":\"\"}]}. "
            "When no tool is needed output an empty tool_calls array.")


def rewrite_system() -> str:
    return ("You are a joint tool-specific argument and query rewriter. "
            "The selected tool names are FINAL: do not add or remove tools. "
            "Create one entry per tool, respecting tool-specific argument schemas. "
            "An entity extraction API needs structured entities, while search APIs need "
            "search-ready queries. Do not execute tools, do not invent information. "
            "Output {\"tool_calls\": [{\"tool\":\"exact_name\",\"arguments\":{},\"search_query\":\"\"}]}.")


def normalize_system() -> str:
    return ("Convert the CURRENT user message into a standalone request using previous "
            "USER messages. Resolve pronouns, ellipsis, and references. Preserve ONLY "
            "the current-turn intent; do NOT repeat already-completed earlier requests. "
            "Do not infer results of prior tool calls. Do not invent facts. "
            "Output {\"standalone_query\":\"...\"}.")


def router_payload(context: str, allowed_names: list[str], catalog: dict) -> dict:
    return {"user_request": context, "available_tools": json.loads(menu(catalog, allowed_names))}


def selected_payload(context: str, selected: list[str], catalog: dict) -> dict:
    return {
        "user_request": context,
        "selected_tools": [catalog[n] for n in selected],
    }


def route_only(llm: LLM, context: str, allowed: list[str], catalog: dict) -> tuple[list[str], list[str]]:
    reply = llm.json_call(selection_system(), router_payload(context, allowed, catalog))
    proposed = reply.get("tools", [])
    if not isinstance(proposed, list):
        return [], ["<malformed tools>"]
    valid = sorted(set(x for x in proposed if isinstance(x, str) and x in allowed))
    invalid = [str(x) for x in proposed if not (isinstance(x, str) and x in allowed)]
    return valid, invalid


def flatten_context(query: str, history: list[str]) -> str:
    if not history:
        return query
    return "Previous user turns:\n" + "\n".join(
        f"{i+1}. {turn}" for i, turn in enumerate(history)
    ) + "\n\nCurrent user turn:\n" + query


def get_normalized(llm: LLM, ex: Example, cache: dict, cache_path: Path) -> tuple[str, float, int, bool]:
    if ex.id in cache:
        item = cache[ex.id]
        # Older cache format is not timing-comparable. Require a fresh run.
        if not isinstance(item, dict) or "latency_sec" not in item:
            raise ValueError("Old standalone cache format; please start a new output directory")
        return item["query"], item["latency_sec"], item["llm_calls"], True
    t0, n0 = time.perf_counter(), llm.calls
    reply = llm.json_call(normalize_system(), {"history": ex.history, "current_query": ex.query})
    norm = reply.get("standalone_query", "")
    if not isinstance(norm, str) or not norm.strip():
        raise ValueError(f"Invalid standalone query for {ex.id}")
    item = {
        "query": norm.strip(),
        "latency_sec": time.perf_counter() - t0,
        "llm_calls": llm.calls - n0,
    }
    cache[ex.id] = item
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return item["query"], item["latency_sec"], item["llm_calls"], False


def score(pred: list[str], gold: list[str], invalid: list[str]) -> dict:
    # Invalid hallucinated names are false positives, never silently erased.
    p = set(pred) | {f"INVALID:{n}" for n in invalid}
    g = set(gold)
    tp = len(p & g)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else (1.0 if not p else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "exact": int(p == g), "tp": tp, "fp": len(p - g), "fn": len(g - p)}


def run_one(ex: Example, mode: str, arch: str, catalog: dict,
            llm: LLM, emb: EmbeddingIndex | None, cfg: argparse.Namespace,
            norm_cache: dict, norm_cache_path: Path) -> dict:
    normalized_latency, normalized_calls, normalization_cached = 0.0, 0, False
    if mode == "standalone":
        query, normalized_latency, normalized_calls, normalization_cached = get_normalized(
            llm, ex, norm_cache, norm_cache_path)
        context = query
    else:
        query = ex.query
        context = flatten_context(ex.query, ex.history)
    allowed = list(catalog)
    candidate_rank = None
    before_llm, before_embed = llm.calls, emb.calls if emb else 0
    # Normalization must be accounted for in end-to-end latency. Since the
    # normalization is reused among architectures, use wall-clock from the
    # call INCLUDING normalization (see outer runner's measured start).
    if arch == "hybrid":
        if emb is None:
            raise ValueError("hybrid requires an embedding index")
        candidate_rank = emb.search(context, cfg.retrieve_k)
        allowed = [name for name, _ in candidate_rank]
    if arch == "two_stage":
        selected, invalid = route_only(llm, context, allowed, catalog)
        details = []
        rewrite_invalid = []
        if selected:
            result = llm.json_call(rewrite_system(), selected_payload(context, selected, catalog))
            _, details, rewrite_invalid = sanitize_calls(result, set(selected))
        # Routing stage determines selected tools; rewriting is NOT separately
        # scored here. Rewrite errors don't become false-positive tool picks.
        predicted = selected
    else:
        rewrite_invalid = []
        result = llm.json_call(joint_system(), router_payload(context, allowed, catalog))
        predicted, details, invalid = sanitize_calls(result, set(allowed))
    metric = score(predicted, ex.gold, invalid)
    extra = {}
    if candidate_rank is not None:
        found = set(n for n, _ in candidate_rank)
        positions = {n: i + 1 for i, (n, _) in enumerate(candidate_rank)}
        extra = {
            "candidate_recall_at_k": len(found & set(ex.gold)) / len(ex.gold),
            "candidate_all_gold_at_k": int(set(ex.gold) <= found),
            "candidate_mrr": (1.0 / min(positions[g] for g in ex.gold if g in positions)
                              if found & set(ex.gold) else 0.0),
            "candidates": candidate_rank,
        }
    return {
        "id": ex.id, "split": ex.split, "mode": mode, "architecture": arch,
        "query": ex.query, "history": ex.history, "routed_query": query,
        "gold": ex.gold, "predicted": predicted, "invalid": invalid,
        "rewrite_invalid": rewrite_invalid,
        "tool_arguments": details,
        "llm_calls": llm.calls - before_llm,
        "embed_calls": (emb.calls if emb else 0) - before_embed,
        "normalization_latency_sec": normalized_latency,
        "normalization_llm_calls": normalized_calls,
        "normalization_cached": normalization_cached,
        **metric, **extra,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["mode"], row["architecture"])].append(row)
    output = []
    for (split, mode, arch), group in sorted(grouped.items()):
        n = len(group)
        tp = sum(r.get("tp", 0) for r in group)
        fp = sum(r.get("fp", 0) for r in group)
        fn = sum(r.get("fn", 0) for r in group)
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        row = {
            "split": split, "mode": mode, "architecture": arch, "n": n,
            "exact": round(statistics.mean(x["exact"] for x in group), 4),
            "macro_f1": round(statistics.mean(x["f1"] for x in group), 4),
            "macro_recall": round(statistics.mean(x["recall"] for x in group), 4),
            "micro_f1": round(2 * p * r / (p + r) if p + r else 0, 4),
            "p50_sec": round(float(np.percentile([x["latency_sec"] for x in group], 50)), 3),
            "p95_sec": round(float(np.percentile([x["latency_sec"] for x in group], 95)), 3),
            "llm_calls_mean": round(statistics.mean(x["llm_calls"] for x in group), 2),
            "embed_calls_mean": round(statistics.mean(x["embed_calls"] for x in group), 2),
            "errors": sum(bool(x.get("error")) for x in group),
        }
        if arch == "hybrid":
            row["candidate_recall_at_k"] = round(statistics.mean(
                x.get("candidate_recall_at_k", 0) for x in group), 4)
            row["candidate_all_gold_at_k"] = round(statistics.mean(
                x.get("candidate_all_gold_at_k", 0) for x in group), 4)
            row["candidate_mrr"] = round(statistics.mean(
                x.get("candidate_mrr", 0) for x in group), 4)
        output.append(row)
    return output


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--single-n", type=int, default=100)
    ap.add_argument("--multi-n", type=int, default=100)
    ap.add_argument("--multi-only-multilabel", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrieve-k", type=int, default=16)
    ap.add_argument("--chat-url", default="https://api.openai.com/v1",
                    help="OpenAI by default; override for a compatible proxy")
    ap.add_argument("--chat-model", default=os.getenv("CHAT_MODEL", "gpt-6-luna"))
    ap.add_argument("--chat-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--embed-url", default=os.getenv("EMBED_URL", "https://api.openai.com/v1"),
                    help="OpenAI default; use --embed-url http://localhost:8001/v1 for local vLLM")
    ap.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    ap.add_argument("--embed-key", default=os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--embed-batch", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Optional; only sent with GPT-6 if reasoning-effort=none")
    ap.add_argument("--reasoning-effort", default="none",
                    choices=["none", "low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--data-base-url", default=BFCL_RAW_BASE,
                    help="Source base URL for BFCL data, defaults to official GitHub raw")
    ap.add_argument("--no-download", action="store_true",
                    help="Do not access GitHub; require locally cached files")
    ap.add_argument("--force-download", action="store_true",
                    help="Re-download all BFCL data even if cached")
    ap.add_argument("--download-only", action="store_true",
                    help="Download the data and exit without making model API calls")
    ap.add_argument("--max-menu-chars", type=int, default=160000,
                    help="Fail early instead of truncating tool definitions (default 160k chars).")
    ap.add_argument("--resume", action="store_true", help="Skip previously completed experiment rows")
    args = ap.parse_args()
    if args.download_only and args.no_download:
        ap.error("--download-only and --no-download cannot be used together")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        dataset_files = download_bfcl_dataset(args.data_dir, args.data_base_url,
                                              force=args.force_download, timeout=args.timeout)
        (args.output_dir / "download_manifest.json").write_text(
            json.dumps(dataset_files, indent=2), encoding="utf-8")
    else:
        missing = [str(args.data_dir / rel) for rel in BFCL_DOWNLOAD_FILES
                   if not (args.data_dir / rel).is_file()]
        if missing:
            ap.error("Missing local files with --no-download: " + ", ".join(missing))
    if args.download_only:
        print(f"BFCL dataset ready at {args.data_dir}")
        return
    if OpenAI is None:
        ap.error("OpenAI SDK is not installed. Run: pip install -r requirements.txt")
    if not args.chat_key:
        ap.error("Set OPENAI_API_KEY (or pass --chat-key) for gpt-6-luna")
    if not args.embed_key:
        ap.error("Set OPENAI_API_KEY or EMBED_API_KEY (or pass --embed-key)")
    single_records = read_records(args.data_dir / "BFCL_v4_parallel_multiple.json")
    catalog = catalog_from_sources(single_records, args.data_dir / "multi_turn_func_doc")
    examples, skips = create_examples(args.data_dir, args.seed, args.single_n,
                                      args.multi_n, catalog, args.multi_only_multilabel)
    if not examples:
        ap.error("No usable examples; check the paired possible_answer files and multi-turn tool docs.")
    full_menu_chars = len(menu(catalog, list(catalog)))
    print(f"Tools={len(catalog)}, compact menu chars={full_menu_chars}, "
          f"examples={Counter(x.split for x in examples)}, skipped={len(skips)}", flush=True)
    if full_menu_chars > args.max_menu_chars:
        ap.error(f"Catalog larger than --max-menu-chars ({full_menu_chars} chars). "
                 "Increase --max-menu-chars if your model context can accommodate it. "
                 "Never silently trim away gold tools.")
    (args.output_dir / "skipped.json").write_text(json.dumps(skips, indent=2), encoding="utf-8")
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(
        {"data_dir": str(args.data_dir), "seed": args.seed, "n_single": args.single_n,
         "n_multi": args.multi_n, "catalog_size": len(catalog),
         "catalog_chars": full_menu_chars, "examples": [asdict(x) for x in examples]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    llm = LLM(args)
    emb = EmbeddingIndex(args, catalog, args.output_dir)
    print(f"Catalog index ready. Initial embedding calls={emb.calls} (excluded from request latency).")
    normalized_path = args.output_dir / "standalone_queries.json"
    if not args.resume and normalized_path.exists():
        normalized_path.unlink()
    normalized = json.loads(normalized_path.read_text(encoding="utf-8")) if normalized_path.exists() else {}
    detailed_path = args.output_dir / "predictions.jsonl"
    completed = set()
    if args.resume and detailed_path.exists():
        for row in read_records(detailed_path):
            completed.add((row["id"], row["mode"], row["architecture"]))
    else:
        detailed_path.write_text("", encoding="utf-8")
    total = len(examples) * 3 + sum(x.split == "multi" for x in examples) * 3
    number = 0
    for ex in examples:
        modes = ["raw"] if ex.split == "single" else ["raw", "standalone"]
        for mode in modes:
            for arch in ("two_stage", "one_shot", "hybrid"):
                number += 1
                if (ex.id, mode, arch) in completed:
                    continue
                t0 = time.perf_counter()
                lc0, ec0 = llm.calls, emb.calls
                try:
                    result = run_one(ex, mode, arch, catalog, llm, emb, args,
                                     normalized, normalized_path)
                    result["error"] = ""
                except Exception as exc:
                    result = {
                        "id": ex.id, "split": ex.split, "mode": mode,
                        "architecture": arch, "query": ex.query,
                        "history": ex.history, "gold": ex.gold, "predicted": [],
                        "tool_arguments": [], "invalid": [], "error": repr(exc),
                        **score([], ex.gold, []),
                    }
                wall_sec = time.perf_counter() - t0
                # Charge a cached standalone-normalization stage to EVERY
                # architecture, not only the first variant that computed it.
                if result.get("normalization_cached"):
                    wall_sec += result.get("normalization_latency_sec", 0.0)
                result["latency_sec"] = round(wall_sec, 6)
                result["llm_calls"] = (llm.calls - lc0 +
                                       (result.get("normalization_llm_calls", 0)
                                        if result.get("normalization_cached") else 0))
                result["embed_calls"] = emb.calls - ec0
                append_jsonl(detailed_path, result)
                completed.add((ex.id, mode, arch))
                if number % 5 == 0 or result.get("error"):
                    print(f"[{number}/{total}] {ex.split:6} {mode:10} {arch:9} "
                          f"F1={result['f1']:.2f} secs={result['latency_sec']:.2f} "
                          f"{result.get('error', '')[:100]}", flush=True)
    all_rows = read_records(detailed_path)
    # If the output file contains unrelated runs, fail rather than mix them.
    wanted = {(x.id, m, a) for x in examples
              for m in (["raw"] if x.split == "single" else ["raw", "standalone"])
              for a in ("two_stage", "one_shot", "hybrid")}
    relevant = [r for r in all_rows if (r["id"], r["mode"], r["architecture"]) in wanted]
    if len(relevant) != len(wanted) or len(set((r["id"], r["mode"], r["architecture"]) for r in relevant)) != len(wanted):
        raise RuntimeError("Incomplete/duplicate results; delete predictions.jsonl for a fresh run.")
    summary = aggregate(relevant)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== OFFLINE TOOL SEARCH SUMMARY ===")
    cols = ["split", "mode", "architecture", "n", "exact", "macro_f1", "micro_f1", "p50_sec", "p95_sec", "errors"]
    print(" | ".join(cols))
    for r in summary:
        print(" | ".join(str(r[k]) for k in cols))
    print("\nSaved", detailed_path, args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
