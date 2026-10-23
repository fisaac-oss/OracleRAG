import argparse
import copy
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path and console handles UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.services.ai_service import AIService

TEST_PATH = Path(__file__).resolve().parent / "test_cases.json"


def save_cases(cases, output_path: Path = TEST_PATH):
    """Writes the current case results to output_path."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


def evaluate_rpg_systems(input_path: Path = TEST_PATH, output_path: Path = TEST_PATH):
    """
    Executes each test case from input_path through AIService,
    records actual_result, citations, latency, and saves results to output_path.
    """
    print(f"Loading test cases from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    cases = copy.deepcopy(raw_cases)
    if output_path.resolve() != input_path.resolve():
        for c in cases:
            c.pop("review_notes", None)

    service = AIService()
    indexed_tomes = service.get_indexed_tomes()
    total = len(cases)
    print(
        f"Configuration: model={config.model_name}, temp={config.model_temperature}, "
        f"timeout={config.ollama_timeout}s, similarity_threshold={config.similarity_threshold}"
    )
    print(f"Indexed tomes ({len(indexed_tomes)}): {indexed_tomes}")
    print(f"Starting evaluation of {total} test cases...\n" + "=" * 60)

    for idx, case in enumerate(cases, 1):
        case_id = case.get("id", f"case-{idx}")
        system_name = case.get("target_system", "General")
        query = case.get("input", "")
        tomes = case.get("selected_tomes")

        print(f"[{idx}/{total}] Running {case_id} ({case.get('category')}) | System: {system_name}")
        print(f"      Input: \"{query}\"")
        if tomes is not None:
            print(f"      Selected Tomes: {tomes}")

        t0 = time.perf_counter()
        try:
            # Handle conversational follow-ups if context is provided
            context = case.get("conversation_context")
            message_to_send = query
            if context and isinstance(context, list) and len(context) > 0:
                # Prepend conversation context to clarify references for single-turn prompt execution
                context_str = " | ".join(context)
                message_to_send = f"[Context: {context_str}] {query}"

            # Normalize tome names against indexed tomes if .pdf extension was omitted
            if tomes is not None and len(tomes) > 0:
                indexed = service.get_indexed_tomes()
                normalized = []
                for t in tomes:
                    if t in indexed:
                        normalized.append(t)
                    elif f"{t}.pdf" in indexed:
                        normalized.append(f"{t}.pdf")
                    else:
                        normalized.append(t)
                tomes = normalized

            response = service.process_message(
                user_message=message_to_send,
                selected_tomes=tomes,
            )
            elapsed = round(time.perf_counter() - t0, 2)

            case["actual_result"] = response.content
            case["latency_seconds"] = elapsed
            case["citations"] = [
                {
                    "source": c.source,
                    "section": c.section,
                    "page": c.page,
                    "text": c.text,
                }
                for c in response.citations
            ]

            # Automated preliminary status check. Content remains available for
            # human review when a rule requires more nuanced judgment.
            result_text = response.content.lower()
            citation_text = " ".join(c.text for c in response.citations).lower()
            keywords = [k.lower() for k in case.get("expected_citation_keywords", [])]
            expected_error = case.get("expected_error_message")

            if expected_error:
                case["status"] = (
                    "pass" if expected_error.lower() in result_text else "fail"
                )
            elif case.get("expected_fallback"):
                fallback_markers = (
                    "contain no decrees",
                    "no grimoires or scriptures have been selected",
                )
                case["status"] = (
                    "pass"
                    if not response.citations and any(m in result_text for m in fallback_markers)
                    else "fail"
                )
            elif not response.success or not response.citations:
                case["status"] = "fail"
            elif keywords and not all(
                keyword in citation_text or keyword in result_text
                for keyword in keywords
            ):
                case["status"] = "partial"
            else:
                case["status"] = "review"

            print(
                f"      -> Completed in {elapsed:.2f}s [{case['status']}]. "
                f"Output preview: {response.content[:80].strip()}...\n"
            )

        except Exception as err:
            elapsed = round(time.perf_counter() - t0, 2)
            print(f"      -> ERROR after {elapsed:.2f}s: {err}\n")
            case["actual_result"] = f"Execution error: {err}"
            case["latency_seconds"] = elapsed
            case["status"] = "fail"

        # Save after every case so progress survives an interrupted run
        save_cases(cases, output_path)

    print("=" * 60)
    print(f"Evaluations complete! Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Oracle RAG evaluation suite.")
    parser.add_argument(
        "--input",
        type=Path,
        default=TEST_PATH,
        help="Path to input test cases JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TEST_PATH,
        help="Path to write evaluated test cases JSON file.",
    )
    args = parser.parse_args()
    evaluate_rpg_systems(input_path=args.input, output_path=args.output)
