from typing import override

from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset
from core.utils.cot_answers import extract_math_answer, verify_math_answer


class HendrycksMathCoTResponseDataset(HendrycksMathDirectResponseDataset):
    """Competition math reasoning evaluated with mathematical equivalence."""

    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are competition mathematics problems. "
            "Solve the problem step-by-step, then put the final answer on a new line.\n\n"
            "Required final-line format: Answer: <answer>\n"
            "Replace <answer> with your final answer in the form requested by the question. "
            "Do not put prose, a list number, Markdown formatting, math delimiters, or a \boxed{} wrapper on that line. "
            "Stop immediately after the final answer."
        )

    @override
    def assistant_response(self, row: dict) -> str:
        raise NotImplementedError("HendrycksMathCoTResponseDataset is for evaluation only.")

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        answer = extract_math_answer(assistant_response, self.tokenizer.all_special_tokens)
        return answer, verify_math_answer(self._ground_truth_answer(row), answer)
