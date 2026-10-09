# BFCL V4 Multi-tool Router Benchmark — GPT-6 Luna

The code uses the **OpenAI Python SDK** with **`gpt-6-luna`** for every generative step: routing, joint argument/query rewriting, and multi-turn standalone rewriting. It automatically downloads BFCL V4 data from the **official Gorilla GitHub repository**. No manual data preparation or local LLM server is needed.

## Fix for `Catalog larger than --max-menu-chars`

The global catalog can contain **586 tools** (roughly 241k characters of compact schemas), which is too large to put into every model prompt. As of this version, the 586-tool catalog is **still embedded once**, but the LLM sees a bounded **per-example candidate pool**. All architectures use the **same per-example pool**:

1. BFCL single-turn: functions listed in the question's `function` definitions. Multi-turn: methods in question's `involved_classes`, obtained from the published function docs. These represent tools the *question* makes available, not the answer labels.
2. In the default `--catalog-scope sampled`, sample extra irrelevant candidate tools from the global catalog up to `--pool-size 32`, using a repeatable seed. Do **not** truncate tools provided by BFCL; do **not** add tools from the ground-truth labels.
3. Two-stage and one-shot route over the full candidate pool. Hybrid runs embedding Top-K **inside the same pool** before joint routing.
4. Check the prompt-size guard on each example's menu, not on the total catalog.

```bash
# Works with data/ files already downloaded by the previous version
python3 benchmark.py --single-n 50 --multi-n 50 --retrieve-k 16 \
  --catalog-scope sampled --pool-size 32 --output-dir results_32

# More challenging comparison with 64 candidates, retrieval Top-16
python3 benchmark.py --single-n 50 --multi-n 50 --retrieve-k 16 \
  --catalog-scope sampled --pool-size 64 --output-dir results_64

# BFCL task's original offered tools ONLY: typically a small candidate set
python3 benchmark.py --catalog-scope example --output-dir results_original

# Legacy full 586-tool menu: very large input and potentially high cost
python3 benchmark.py --catalog-scope global --max-menu-chars 300000 \
  --output-dir results_global
```

The artificial distractors make results sensitive to `--pool-size` and `--seed`. **Do not compare numbers from different pool sizes as if they used the same benchmark setup.** BFCL function-tool selection accuracy is an *offline proxy* for tool retrieval; these metrics do not measure downstream document relevance. Previously created `results/` from the old version should not be reused for a different candidate pool; choose a new `--output-dir`.

## Quick start

```bash
unzip bfcl_router_benchmark_gpt6_luna.zip
cd bfcl_router_benchmark
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python benchmark.py --single-n 100 --multi-n 100 --retrieve-k 16
```

**Recommended first run** (small API bill):

```bash
python benchmark.py --single-n 3 --multi-n 3 --output-dir results_smoke
```

The first run downloads all 12 required files to `data/`. Future runs **reuse** the cached files without downloading again. If you only want to fetch the data:

```bash
python benchmark.py --download-only
```

The run fails with a clear error if GitHub cannot be reached and the required files are not already cached. To run fully offline with previously downloaded BFCL data: `--no-download`. To re-fetch files: `--force-download`.

### Experiment matrix

| Architecture | How it works |
| --- | --- |
| `two_stage` | Multi-label LLM router → one joint arguments/query rewriter for selected tools |
| `one_shot` | One joint call selects tools and generates tool-specific arguments/queries |
| `hybrid` | Embedding top-K tool retrieval → one joint selection + argument/query generation call |

Contexts: **Single-turn raw**; **Multi-turn raw** (all prior user turns + current user turn); **Multi-turn standalone** (LLM resolves the current turn into a self-contained request **before routing and retrieval**). That is **9 conditions** across 3 architectures. By default, each dataset is sampled at up to 100 items with `--single-n 100 --multi-n 100` (actual usable counts printed). A strict multi-label subset of multi-turn conversations can be requested via `--multi-only-multilabel`; there may be fewer examples than requested.

### OpenAI and local embeddings

Default settings:

- Generation: `gpt-6-luna`, API `https://api.openai.com/v1`, `--reasoning-effort none`.
- Embeddings: `text-embedding-3-small`, API `https://api.openai.com/v1`.
- JSON responses via `chat.completions.create()`, with `max_completion_tokens` instead of deprecated `max_tokens` for GPT-6.

Embedding and chat access credentials default to `OPENAI_API_KEY`. To use an existing vLLM **embedding** server, for instance:

```bash
python benchmark.py --single-n 100 --multi-n 100 \
  --embed-url http://127.0.0.1:8001/v1 \
  --embed-model BAAI/bge-m3 --embed-key EMPTY
```

Or set `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY`. The chat model remains `gpt-6-luna` on the official OpenAI API unless you **explicitly** change `--chat-url` or `--chat-model`. GPT-6 Luna is a hosted API model, not a model to download and serve inside vLLM.

To test the effect of reasoning effort:

```bash
python benchmark.py --reasoning-effort low --output-dir results_low
```

`--temperature` is omitted by default. It will be passed for GPT-6 only if `--reasoning-effort none`; it is not supported with other GPT-6 reasoning efforts.

### Data download source and layout

Downloaded using Python `urllib` from official raw URLs, with retries, JSON/JSONL validation, and atomic writes. Layout:

```text
data/
  BFCL_v4_parallel_multiple.json
  BFCL_v4_multi_turn_base.json
  possible_answer/
    BFCL_v4_parallel_multiple.json
    BFCL_v4_multi_turn_base.json
  multi_turn_func_doc/
    gorilla_file_system.json
    math_api.json
    message_api.json
    posting_api.json
    ticket_api.json
    trading_bot.json
    travel_booking.json
    vehicle_control.json
```

Official source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data

If GitHub is blocked in your execution environment, copy those same files into `data/` (or supply a mirrored base URL via `--data-base-url`) and run with `--no-download`.

### Outputs and metrics

- `results/summary.json`: Tool-set exact match, macro/micro F1, recall, P50/P95 latency, mean calls, errors; additionally candidate Recall@K / All-Gold@K / MRR for hybrid.
- `results/predictions.jsonl`: each sample's gold, predictions, tool arguments, search-query rewrites, candidate rankings, and latency.
- `results/standalone_queries.json`: separately generated multi-turn standalone queries reused among architectures.
- `results/dataset_manifest.json`, `results/download_manifest.json`, `results/skipped.json`: dataset and download metadata.

Repeated calls to one function are collapsed to **distinct tool names** for these metrics. This evaluates **Tool Retrieval / Tool Selection**, **not** downstream document retrieval or official BFCL function-execution accuracy. The generated queries and parameters are logged but not separately scored. BFCL does not provide document relevance judgments for query rewrite nDCG. Multi-turn is an offline approximation using previous **user turns only**, without replaying assistant responses, earlier tool results, or evolving tool state.

A 100/100 run can trigger **over 1,000 paid LLM calls** because each condition and each stage is evaluated; start with the smoke test. Static catalog embedding construction is cached and excluded from per-query latency. The standalone rewrite is generated once per example, but its recorded latency and LLM calls are **charged to each architecture** for fair comparison. Experiments are processed sequentially. Use `--resume` with the same parameters and output folder to continue interrupted runs; start a new output folder when changing the settings.

## Offline tests

```bash
python -m unittest -v test_benchmark.py
```

These tests mock network/model calls. **They do not prove that your API key, current OpenAI service, or internet access works.**
## Experimental Results (GPT-6 Luna, BFCL V4)

### Experimental Setup

- **LLM:** `gpt-6-luna` (OpenAI Chat Completions API; reasoning effort: `none`)
- **Embedding:** OpenAI-compatible embedding API (default: `text-embedding-3-small`)
- **Dataset:** BFCL V4 `parallel_multiple` (50 single-turn cases) and `multi_turn_base` (50 selected target turns from 50 conversations)
- **Catalog:** 586 distinct tools; all tools embedded once (25 embedding API calls), with initial index construction excluded from per-request latency
- **Candidate setting:** `--catalog-scope sampled --pool-size 32 --retrieve-k 16`; BFCL-provided tools for each question are retained, with randomly sampled distractors added to reach a target pool of 32. Hybrid performs embedding Top-16 *within this pool*, not over the full 586 tools.
- **Conditions:** Three architectures for Single-turn Raw, Multi-turn Raw, and Multi-turn Standalone, giving 9 conditions and 450 evaluations in total
- **Metrics:** Tool-set Exact Match (EM), per-example Macro F1, aggregate Micro F1, P50/P95 end-to-end latency, and execution errors. EM measures exact **tool-set selection**, not the correctness of rewritten arguments or search queries.

### Architecture and Input Mode Definitions (아키텍처 및 입력 모드 차이)

**모든 실험은 같은 샘플별 후보 Tool Pool을 사용**한다. 이번 실행은 `--catalog-scope sampled --pool-size 32`로 BFCL이 해당 문제에 제공한 Tool 목록을 보존하고, 무작위 distractor를 추가해 후보를 구성했다. `Hybrid`만 이 후보 중 Embedding 점수 상위 `--retrieve-k 16`개를 2차 후보로 사용한다.

#### A. Architecture: One-shot vs. Two-stage vs. Hybrid

| Architecture | Processing flow | LLM calls per request (excluding standalone normalization) | Characteristics |
|---|---|---|---|
| **One-shot** (`one_shot`) | User request + Tool descriptions/schemas → **one LLM call** jointly chooses tools and generates tool-specific arguments/search queries | **1** | Tool 선택과 Entity Extraction/Query Rewriting을 동시에 수행. 간단하고 빠르지만 선택과 인자 생성 오류를 분리해서 디버깅하기 어려움 |
| **Two-stage** (`two_stage`) | User request + Tool descriptions/schemas → **Multi-label Router** → **Joint Rewriter** for selected tools | **보통 2** (선택한 Tool이 없으면 1) | 1단계에서 필요한 Tool 집합을 확정하고 2단계에서 Tool별 arguments/query를 생성. Router 선택 결과는 Rewriter에서 추가하거나 삭제하지 않음 |
| **Hybrid** (`hybrid`) | User request → **Embedding Top-16 retrieval from candidate pool** → **one joint LLM call** for routing + arguments/query | **1 + query embedding** | 후보를 먼저 줄여 Prompt를 단축할 수 있지만, 정답 Tool이 Top-K에서 누락되면 이후 LLM이 복구할 수 없음 |

공통으로 Tool Description 및 Parameter Schema가 입력되며, Tool에 따라 정해진 형식의 Entity Extraction (`arguments`) 또는 검색용 Query Rewriting (`search_query`)을 수행한다. 따라서 `One-shot`과 `Hybrid`는 **Joint Routing + Argument/Query Generation**, `Two-stage`는 **Multi-label Routing → Joint Argument/Query Generation**이다. 별도의 독립 Tool별 Rewriter를 병렬 호출하는 방식은 이번 비교에 포함하지 않았다.

```text
One-shot:  Query + (32 candidate tools) --> [LLM: select + rewrite] ------> Tool calls
Two-stage: Query + (32 candidate tools) --> [LLM: select] --> [LLM: rewrite selected tools] --> Tool calls
Hybrid:    Query --> [Embedding: 32 candidates -> Top-16] --> [LLM: select + rewrite] --> Tool calls
```

**채점상의 중요한 차이:** `Two-stage`는 **첫 번째 Router의 Tool 선택 결과**로 Tool-set EM/F1을 계산한다. 두 번째 Joint Rewriter의 argument/query 생성 정확도는 별도 채점하지 않는다. `One-shot`/`Hybrid`는 최종 joint 출력에 포함된 Tool 이름으로 Tool-set EM/F1을 계산한다. 그래서 이번 수치는 **Tool-set 선택 성능**이며, 각 방식의 최종 Tool 호출/검색 성공률을 의미하지 않는다.

#### B. Input mode: Multi Raw vs. Multi Standalone

두 방식 모두 **동일한 Multi-turn 평가 대상 50개**를 사용하지만, Router에 전달되는 입력 문맥이 다르다.

| Input mode | Before routing | Input actually passed to routing / Hybrid embedding retrieval | Extra LLM call |
|---|---|---|---|
| **Multi Raw** (`multi/raw`) | 별도의 질문 정규화 없음 | `Previous user turns` + `Current user turn`을 합쳐 그대로 전달 | 없음 |
| **Multi Standalone** (`multi/standalone`) | 이전 **사용자 발화**를 참고해 현재 질문을 독립적으로 이해 가능한 문장으로 정규화 | **Standalone query만** 전달. Raw History는 뒤의 Router/Retriever에 다시 붙이지 않음 | 정규화용 1회 (세 아키텍처에서 같은 결과를 재사용하며 비용·지연시간은 각 조건에 반영) |
| **Single Raw** (`single/raw`) | 이전 턴이 없음 | 해당 Single-turn 질문 자체 | 없음 |

예시 (설명용 가상 발화이며 BFCL 데이터셋의 실제 사례가 아님):

```text
Previous user turn: "삼성전자 2026년 2분기 실적을 보고 싶어."
Current user turn:  "관련 뉴스도 찾아줘."

Multi Raw — router input:
  Previous user turns:
  1. 삼성전자 2026년 2분기 실적을 보고 싶어.

  Current user turn:
  관련 뉴스도 찾아줘.

Multi Standalone — prior normalization LLM output:
  "삼성전자 2026년 2분기 실적과 관련된 뉴스를 찾아줘."

Multi Standalone — router input:
  삼성전자 2026년 2분기 실적과 관련된 뉴스를 찾아줘.
```

`Multi Standalone`은 이전 요청을 모두 재실행하기 위한 Rewrite가 아니라, **현재 턴의 의도만 유지하면서** 지시어/생략 정보를 복원하는 전처리다. `Multi Raw`는 이 전처리 단계를 건너뛰고 원본 이전 사용자 메시지를 직접 전달한다. **현재 구현은 이전 Assistant 발화·Tool 실행 결과·상태 변경은 포함하지 않는다.**

#### C. 3 × 3 Experiment Matrix

| Split / Input | One-shot | Two-stage | Hybrid |
|---|---|---|---|
| Single Raw | Joint select+rewrite | Route → joint rewrite | Embedding Top-16 → joint select+rewrite |
| Multi Raw | Raw user history → joint | Raw user history → route → joint rewrite | Raw user history → embed → joint |
| Multi Standalone | Normalize → joint | Normalize → route → joint rewrite | Normalize → embed → joint |

Multi Raw와 Multi Standalone의 F1/EM 비교는 **질문 정규화를 라우팅 앞에 두는 것이 유리한가**를 테스트한다. 아키텍처 간 비교는 **Tool Retrieval, Tool Selection, Argument/Query Generation을 어떻게 결합하는가**를 테스트한다. 다만 생성된 `arguments` 및 `search_query`의 내용은 별도 정확도 채점 대상이 아니다.

### Full Results

| Split | Input mode | Architecture | N | Tool-set EM | Macro F1 | Micro F1 | P50 (s) | P95 (s) | Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Single | Raw | **One-shot** | 50 | **1.00** | **1.0000** | **1.0000** | **1.798** | **2.275** | 0 |
| Single | Raw | Two-stage | 50 | 0.98 | 0.9933 | 0.9964 | 3.346 | 4.817 | 0 |
| Single | Raw | Hybrid | 50 | 0.98 | 0.9950 | 0.9928 | 2.037 | 3.131 | 0 |
| Multi | Raw history | **Two-stage** | 50 | **0.40** | **0.7643** | **0.7273** | 3.053 | 4.344 | 0 |
| Multi | Raw history | One-shot | 50 | 0.32 | 0.6485 | 0.6333 | **1.569** | **2.521** | 0 |
| Multi | Raw history | Hybrid | 50 | 0.20 | 0.5081 | 0.5814 | 1.650 | 2.386 | 0 |
| Multi | Standalone | **Two-stage** | 50 | **0.32** | **0.6758** | **0.6278** | 4.542 | 6.472 | 0 |
| Multi | Standalone | One-shot | 50 | 0.22 | 0.5328 | 0.5955 | **3.236** | **4.988** | 0 |
| Multi | Standalone | Hybrid | 50 | 0.22 | 0.5141 | 0.5665 | 3.481 | 5.809 | 0 |

### Key Findings

**1. Single-turn: One-shot is best in this setup.** One-shot selected the exact gold tool set in **50/50 cases** (EM = 1.00; Macro F1 = 1.0000) and had the lowest median latency (1.798 s). Two-stage and Hybrid both achieved 49/50 exact matches, with median latencies of 3.346 s and 2.037 s, respectively. The near-ceiling scores should not be generalized to an unrestricted 586-tool search: all BFCL-provided tools for a question remain in the sampled candidate pool.

**2. Multi-turn Raw: Two-stage yields the highest tool-selection accuracy.** Two-stage reached **Macro F1 = 0.7643, EM = 0.40 (20/50)**, versus One-shot at F1 = 0.6485, EM = 0.32 (16/50), and Hybrid at F1 = 0.5081, EM = 0.20 (10/50). Relative to One-shot, Two-stage improved Macro F1 by **0.1158** and exact match by **8 percentage points**, but increased median latency from 1.569 s to 3.053 s (about **1.95×**). This supports a quality/latency trade-off: Two-stage for selection accuracy, One-shot for speed.

**3. Multi-turn Standalone normalization did not yield a net improvement.** Comparing each architecture on the same sampled multi-turn examples:

| Architecture | Raw F1 | Standalone F1 | Δ F1 | Raw EM | Standalone EM | Δ EM | Raw P50 | Standalone P50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Two-stage | 0.7643 | 0.6758 | **-0.0885** | 0.40 | 0.32 | -8 pp | 3.053 s | 4.542 s |
| One-shot | 0.6485 | 0.5328 | **-0.1157** | 0.32 | 0.22 | -10 pp | 1.569 s | 3.236 s |
| Hybrid | 0.5081 | 0.5141 | +0.0060 | 0.20 | 0.22 | +2 pp | 1.650 s | 3.481 s |

The small Hybrid Macro F1 gain (+0.0060) is not sufficient evidence of a reliable improvement, especially with only 50 examples and higher latency. In this benchmark, the raw mode retains prior **user** turns, whereas standalone normalization condenses them into one request; missing references, omitted constraints, or history information loss are plausible explanations to investigate through per-example logs. Earlier assistant messages, tool outputs, and evolving state are not replayed, so these results do not prove that standalone normalization is harmful in a full conversational agent.

**4. Hybrid retrieval was not beneficial at pool size 32 / Top-16.** One-shot exceeded Hybrid in Macro F1 and had lower P50 latency for every corresponding input mode. This measures embedding filtering within a *sampled pool*, rather than global top-K retrieval from all 586 tools. Inspect `candidate_recall_at_k`, `candidate_all_gold_at_k`, and `candidate_mrr` in `results_fixed/summary.json` to distinguish retrieval misses from LLM selection errors; those values were not present in the console summary.

**5. No execution errors were recorded**, but no downstream tools were executed. Query rewriting outputs and entity/argument generation were saved without being scored. Consequently, this benchmark compares **tool-set retrieval/selection**, not argument accuracy, document relevance, full task success, or official BFCL execution accuracy. Two-stage's selected tool set is scored at the routing stage; a failed second-stage rewrite does not directly reduce its tool-set F1, although its time is included in latency.

### Interpretation and Recommended Configuration

- **Single-turn:** Start with **One-shot** (best observed F1 and latency).
- **Multi-turn, accuracy-sensitive:** Start with **Two-stage + Raw History** (best observed F1 and EM).
- **Multi-turn, latency-sensitive:** Start with **One-shot + Raw History** (fastest observed P50/P95 for multi-turn raw).
- **Standalone normalization:** Do not enable it as a universal preprocessing step based on these results. Evaluate it selectively for referential or elliptical follow-up queries, with the original history retained where possible.
- **Hybrid:** Reassess using larger candidate pools (`--pool-size 64/128`) and different Top-K values, preferably followed by global-catalog retrieval and per-query candidate-recall analysis. Its present ranking is not evidence that embedding retrieval is intrinsically inferior at scale.

### Limitations and Next Experiments

1. **Sample size:** 50 cases per dataset; estimates may be unstable. Repeat with more samples or seeds, and compute paired bootstrap confidence intervals on per-example F1/EM.
2. **Candidate-pool bias:** BFCL-supplied candidate tools are retained, and the remaining pool is padded with random distractors. This is easier than choosing among 586 plausible production tools. Report full-catalog or harder negative results separately.
3. **Multi-label scope:** Single-turn cases are restricted to multiple distinct gold tool names, whereas the multi-turn set can include turns with only one gold tool. Run `--multi-only-multilabel` for a strict multi-label comparison.
4. **Conversation approximation:** Multi-turn Raw contains previous user messages only. To benchmark real agent behavior, restore assistant/tool history and intermediate tool results.
5. **Unscored rewriting:** Evaluate generated tool arguments and search queries separately, using parameter-level accuracy and retrieval Recall@K/nDCG@K against actual searchable content.
6. **Latency/cost accounting:** Standalone rewrite latency is charged to each architecture; the 25 initial catalog-embedding API calls are excluded from per-request latency. Include index-build cost when estimating deployment economics.

**Conclusion:** In this 32-candidate BFCL offline tool-selection setup, **One-shot is preferred for Single-turn**, while **Two-stage with Raw History provides the strongest Multi-turn selection quality at approximately 1.95× the median latency of One-shot**. Pre-routing Standalone rewriting did not consistently improve Multi-turn selection accuracy.
