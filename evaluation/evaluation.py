import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.ai_service import AIService

TEST_PATH = Path(__file__).resolve().parent / "test_cases.json"


def evaluate_rpg_systems():
    """
    Executes each test case from evaluation/test_cases.json through AIService,
    records actual_result, citations, and prints progress to the console.
    """
    print(f"Loading test cases from: {TEST_PATH}")
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    service = AIService()
    total = len(cases)
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

            case["actual_result"] = response.content
            case["citations"] = [
                {"source": c.source, "section": c.section, "page": c.page}
                for c in response.citations
            ]

            # Automated preliminary status check
            if case.get("expected_fallback") and "contain no decrees" in response.content:
                case["status"] = "pass"
            elif case.get("expected_error_message") and case["expected_error_message"] in response.content:
                case["status"] = "pass"
            elif response.success and response.content:
                # Marked for human review, preliminary status can be 'review' or left for manual review
                if case.get("status") == "pending":
                    case["status"] = "review"

            print(f"      -> Completed. Output preview: {response.content[:80].strip()}...\n")

        except Exception as err:
            print(f"      -> ERROR: {err}\n")
            case["actual_result"] = f"Execution error: {err}"
            case["status"] = "fail"

    # Save results back to test_cases.json
    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Evaluations complete! Results saved to {TEST_PATH}")


if __name__ == "__main__":
    evaluate_rpg_systems()
