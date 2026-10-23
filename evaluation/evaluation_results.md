# Evaluation Results Summary

Results of running [`evaluation.py`](evaluation.py) against all cases in [`test_cases.json`](test_cases.json), followed by a manual review of every answer and its citations. Each case in `test_cases.json` has its final `status` and a `review_notes` field explaining the verdict.

## Summary Overview

- **Date of Evaluation:** 2026-10-23
- **Model Evaluated:** `qwen3:8b` (Ollama, temperature 0.0)
- **Indexed Tomes:** D&D 5e Player's Handbook, Call of Cthulhu Investigator Handbook 7th Edition (3,926 chunks)
- **Total Test Cases Executed:** 27
- **Passed:** 14
- **Partial:** 9
- **Failed:** 4
- **Pass Rate:** 51.9% (85.2% pass or partial)

## Performance by Category

| Category | Total Cases | Passed | Partial | Failed | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Successful Cases | 9 | 4 | 3 | 2 | 44.4% |
| Difficult Cases | 12 | 5 | 5 | 2 | 41.7% |
| Failure / Edge Cases | 6 | 5 | 1 | 0 | 83.3% |

## Key Findings & Failure Analysis

### What worked well
- **Guardrails are reliable.** Empty and whitespace input (case-25, 26), the no-tomes guardrail (case-27) and the out-of-scope fallback (case-22, 23) all behaved exactly as specified, with no hallucinated rules.
- **No cross-system bleed.** In every Call of Cthulhu anti-bleed test (case-15, 16) the model refused to import D&D death saves or advantage. In the multi-tome tests it never blended the two systems' mechanics into one rule.
- **D&D 5e core rules are strong.** Grappling, advantage/disadvantage, resting, cover vs. hiding and follow-up questions (case-01, 03, 05, 10, 12) were accurate and cited the correct PHB sections and pages.
- **Edition and injection traps held.** The model didn't invent 3.5e rules (case-18) and stayed in the Arbiter role when told to ignore its instructions (case-24).

### Observed failure modes
- **Retrieval misses on Call of Cthulhu (main weakness).** Sanity loss (case-07) and Bonus/Penalty Dice (case-17) are both in the Investigator Handbook, but top-k retrieval returned unrelated chunks. When the right chunk isn't retrieved, the model either says the rule doesn't exist (case-17) or produces an ungrounded answer (case-07).
- **Misreading tabular chunks.** For success levels (case-09), the retriever returned the reference-sheet table, which OCR had flattened into a jumble of numbers. The model invented fixed ranges with "higher is better", which is the opposite of the real rule.
- **Incomplete answers from partial retrieval.** 0 HP without death saving throws (case-02, 19), the action list without the actual actions (case-04), and the multi-condition question without the poisoned/frightened penalties (case-13). The model answers correctly from what it gets, but four chunks often aren't enough.
- **Small hallucinations around correct answers.** Invented details appear next to otherwise correct rules: advantage on escape for larger targets (case-11), a misstated Sneak Attack condition (case-06), and "stabilize via a CON save" (case-20).
- **Multi-tome retrieval bias.** With both tomes selected, results skew toward D&D (case-19, 21), so the answer silently covers only one system instead of flagging the ambiguity.
- **Model timeout.** "Can I cast a spell as a bonus action and another spell as an action?" (case-14) never completed. It timed out at 180s and again at 600s, which looks like a qwen3:8b reasoning loop.
- **Irrelevant citations on refusal.** The prompt-injection case (case-24) refused correctly but still attached four unrelated Dragonborn citations.

### Process issues found during evaluation
- The first run used a relative `CHROMA_DB_DIR=./data/chroma`. Run from the parent folder, it opened a new, empty database, so every RAG case failed and the guardrail cases fell through to general knowledge. Those results were discarded.
- The Ollama client had no timeout, so one stalled generation (case-14) froze the whole run, and results were only saved at the end.

## Action Items & Improvements
- [x] Use an absolute `CHROMA_DB_DIR` in `.env` so the index is found from any working directory.
- [x] Add a configurable Ollama request timeout (`OLLAMA_TIMEOUT`, default 180s) in `src/models/model_client.py`.
- [x] Save `test_cases.json` after every case in `evaluation.py` so interrupted runs keep their progress.
- [ ] Improve retrieval recall: raise top-k, add hybrid keyword (BM25) search, or re-rank, targeting the Call of Cthulhu misses (case-07, 17).
- [ ] Balance multi-tome retrieval by taking top-k per selected tome, and prompt the model to say which system it is answering for, or ask for clarification (case-19, 21).
- [ ] Exclude or specially parse reference-table pages (e.g. CoC p.267-269) so flattened tables aren't served as rule text (case-09).
- [ ] Add a relevance check before answering so off-topic retrievals (case-24) return the fallback with no citations.
- [ ] Disable qwen3 thinking mode or cap output tokens to prevent reasoning loops (case-14).
- [ ] Clean up OCR artifacts in ingestion ("O" for "0", "Vou" for "You"), which can hurt embedding matches.

---

## Round 2 Evaluation — `llama3.2:3b`

Results of running [`evaluation.py`](evaluation.py) (`--input evaluation/test_cases.json --output evaluation/test_cases_llama3.2_3b.json`) against the same 27 test cases using `llama3.2:3b` and the active `.env` configuration, followed by manual review of every answer and citation in [`test_cases_llama3.2_3b.json`](test_cases_llama3.2_3b.json).

### Summary Overview

- **Date of Evaluation:** 2026-10-23
- **Model Evaluated:** `llama3.2:3b` (`Q4_K_M`, 3.2B parameters, temperature `0.15`, timeout `180.0s`)
- **Indexed Tomes in Local ChromaDB (`./data/chroma`):** 6 rulebooks totaling **9,675 chunks** — `Player's Handbook.pdf` (2,397), `Call_of_Cthulhu_Investigator_Handbook_7th_Edition.pdf` (1,517), `Pathfinder 2e - Player Core 2.pdf` (3,243), `The Witcher Pen & Paper RPG.pdf` (2,109), `Print_Weaver_Final.pdf` (237), `Fate_Accelerated_Edition_Electronic.pdf` (172)
- **Total Test Cases Executed:** 27 (0 timeouts, 100% completion)
- **Passed:** 10
- **Partial:** 10
- **Failed:** 7
- **Pass Rate:** 37.0% (74.1% pass or partial)
- **Execution Time:** 226.97s total | 8.41s mean across all 27 cases | 10.23s mean (8.65s median, range 3.16s–24.87s) across the 22 LLM-invoked cases

### Performance by Category

| Category | Total Cases | Passed | Partial | Failed | Pass Rate | Pass + Partial |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Successful Cases | 9 | 4 | 3 | 2 | 44.4% | 77.8% |
| Difficult Cases | 12 | 1 | 6 | 5 | 8.3% | 58.3% |
| Failure / Edge Cases | 6 | 5 | 1 | 0 | 83.3% | 100.0% |
| **Total** | **27** | **10** | **10** | **7** | **37.0%** | **74.1%** |

### Model, Runtime & RAG Parameters

| Parameter | Value | Source |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | `.env` |
| `MODEL_NAME` | `llama3.2:3b` (ID `a80c4f17acd5`) | `.env` / Ollama |
| `MODEL_TEMPERATURE` | `0.15` | `.env` |
| `OLLAMA_TIMEOUT` | `180.0` s | `.env` |
| Architecture / Parameters | `llama` / `3.2B` (`Q4_K_M`, 2.0 GB disk, 2.9 GB VRAM/RAM) | `ollama show` / `ollama ps` |
| Context Window | `4096` active runtime (`131072` model max) | `ollama ps` / `ollama show` |
| Hardware Offload | `82% GPU / 18% CPU` (`llamacpp` runner) | `ollama ps` |
| Stop Tokens | `<\|start_header_id\|>`, `<\|end_header_id\|>`, `<\|eot_id\|>` | `ollama show` |
| `CHROMA_DB_DIR` | `./data/chroma` | `.env` |
| `RAG_COLLECTION_NAME` | `oracle_rules` | `.env` |
| `SIMILARITY_THRESHOLD` | `1.2` (squared L2 distance cutoff) | `.env` |
| Top-$k$ (`n_results`) | `4` chunks per query | `src/capabilities/rag.py` |
| Embedding Model | Chroma `DefaultEmbeddingFunction` (`all-MiniLM-L6-v2` ONNX) | `src/capabilities/rag.py` |

### Round 1 (`qwen3:8b`, temp 0.0) vs. Round 2 (`llama3.2:3b`, temp 0.15)

| Metric | Round 1 (`qwen3:8b`) | Round 2 (`llama3.2:3b`) | Delta |
| :--- | :---: | :---: | :---: |
| Strict Pass | 14 / 27 (51.9%) | 10 / 27 (37.0%) | -4 (-14.8 pp) |
| Partial | 9 / 27 (33.3%) | 10 / 27 (37.0%) | +1 (+3.7 pp) |
| Failed | 4 / 27 (14.8%) | 7 / 27 (25.9%) | +3 (+11.1 pp) |
| Pass + Partial | 23 / 27 (85.2%) | 20 / 27 (74.1%) | -3 (-11.1 pp) |
| Successful Cases Pass Rate | 4 / 9 (44.4%) | 4 / 9 (44.4%) | 0.0 pp |
| Difficult Cases Pass Rate | 5 / 12 (41.7%) | 1 / 12 (8.3%) | -33.3 pp |
| Failure / Edge Cases Pass Rate | 5 / 6 (83.3%) | 5 / 6 (83.3%) | 0.0 pp |
| Timeouts / Stalled Runs | 1 (`case-14` >600s) | **0** (`case-14` in 7.83s) | -1 timeout |

**Status changes across the 27 cases (20 unchanged, 1 improved, 6 regressed):**
- **Improved:** `case-14` (`fail` -> `partial`) — `llama3.2:3b` completed in 7.83s without stalling and stated the exact bonus-action cantrip rule (PHB p.203) in its opening paragraph, before muddling multiple bonus actions in follow-up paragraphs.
- **Regressed (all in Difficult Cases):** `case-10` (`pass` -> `partial`), `case-11` (`partial` -> `fail`), `case-15` (`pass` -> `fail`), `case-16` (`pass` -> `partial`), `case-18` (`pass` -> `fail`), `case-19` (`partial` -> `fail`).

### Key Findings & Failure Analysis (Round 2)

#### What worked well
- **Fast, predictable inference with zero timeouts.** Every LLM call finished between 3.16s and 24.87s (median 8.65s), completely avoiding the reasoning-loop hang observed with `qwen3:8b` on `case-14`.
- **Matches `qwen3:8b` on standard lookups and guardrails.** Both models scored identically on Successful Cases (4/9 pass, 3/9 partial) and Failure/Edge Cases (5/6 pass, 1/6 partial). Even with 6 rulebooks (9,675 chunks) indexed locally, out-of-scope queries (`case-22`, `case-23`) were cleanly rejected by `SIMILARITY_THRESHOLD=1.2`, and prompt injection (`case-24`) was refused in-persona.

#### Observed failure modes specific to `llama3.2:3b` (temp 0.15)
- **Vulnerability to false user premises (anti-bleed & edition traps).** Unlike `qwen3:8b`, `llama3.2:3b` accepted the premise of "death saving throws" in Call of Cthulhu (`case-15`) and fabricated a hybrid `d% + CON` "Death Saving Throw" rule. In `case-18`, it labeled 5e grappling rules as "3.5 edition" (compounded by the local tome name `Player's Handbook.pdf` omitting `5e`).
- **Negation and reading-comprehension inversions on multi-rule chunks.** In `case-11`, it quoted *"don't automatically succeed on checks to escape"* and concluded the exact opposite (*"it automatically succeeds"*), while missing *"no more than one size larger"* in the same retrieved context. In `case-10`, it confused the grappler's half-speed dragging rule with the grappled target's speed of 0; in `case-19`, it misread Instant Death to claim 0 HP always kills a character outright; in `case-06`, it omitted key Sneak Attack clauses present in its citations.
- **System-prompt delimiter leakage (`--- TRAVELER'S INQUIRY ---`).** In `case-16`, `llama3.2:3b` mistook the thematic prompt header `TRAVELER'S INQUIRY` in `src/services/ai_service.py` for the *Traveller* tabletop RPG system and added an ungrounded claim about *Traveller* skill rolls.
- **Shared RAG retrieval bottlenecks persist.** Because retrieval parameters were unchanged (`n_results=4`, `SIMILARITY_THRESHOLD=1.2`), the same Call of Cthulhu misses (`case-07` Sanity, `case-09` success levels, `case-17` Bonus/Penalty Dice) and partial D&D retrievals (`case-02`, `case-04`, `case-13`) recurred.

#### Additional Action Items Discovered
- [x] Normalize machine-specific tome filenames (e.g. `D&D 5e Player's Handbook.pdf` -> `Player's Handbook.pdf`) and support `--input` / `--output` paths + per-case latency logging in `evaluation/evaluation.py`.
- [ ] Replace `--- TRAVELER'S INQUIRY ---` in `src/services/ai_service.py` with `--- PLAYER QUESTION ---` so smaller models do not confuse "Traveler" with the *Traveller* RPG system (`case-16`).
- [ ] Lower `MODEL_TEMPERATURE` from `0.15` to `0.0` and prefer a mid-sized model (7B–8B with thinking capped, or 4B+) when handling adversarial premise traps and multi-condition rule synthesis (`case-11`, `case-15`, `case-18`).
