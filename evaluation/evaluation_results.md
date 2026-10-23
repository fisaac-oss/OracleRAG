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
