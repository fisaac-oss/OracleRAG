import json
from pathlib import Path
from src.services.ai_service import AIService

TEST_PATH = Path("evaluation/test_cases.json")

def evaluate_rpg_systems():
    service = AIService()
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        print(f"Testing [{case['id']}] - System: {case.get('target_system', 'General')}")
        
        # Pass the system-specific tomes defined in the test case
        response = service.process_message(
            user_message=case["input"],
            selected_tomes=case.get("selected_tomes"),
        )
        
        case["actual_result"] = response.content
        case["citations"] = [c.source for c in response.citations]

    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

if _name_ == "_main_": # type: ignore
    evaluate_rpg_systems()