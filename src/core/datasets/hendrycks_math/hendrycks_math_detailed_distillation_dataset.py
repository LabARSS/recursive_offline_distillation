from typing import override

from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset
from core.utils.cot_answers import extract_math_answer, verify_math_answer


class HendrycksMathDistillationDataset(HendrycksMathDirectResponseDataset):
    """Competition math reasoning evaluated with mathematical equivalence."""

    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are competition mathematics problems. "
            "Solve the problem step-by-step. Do not omit any steps. Explain in great detail. Put the final answer on a new line.\n\n"
            "Required final-line format: Answer: <answer>\n"
            "Replace <answer> with your final answer in the form requested by the question. "
            "Use LaTeX for mathematical expressions. The final line must start with the "
            "literal text Answer: followed by a space and the answer on the same line. "
            "Do not put prose, a list number, Markdown formatting, math delimiters, "
            r"or a \boxed{} wrapper on that line. "
            "Stop immediately after the final answer.\n\n"
            "Example response:\n"
            "Dividing both sides of 2x = 1 by 2 gives x = 1/2.\n"
            r"Answer: \frac{1}{2}"
        )

    @override
    def user_prompt(self, row: dict) -> str:
        return (
            f"{super().user_prompt(row)}\n\n"
            "Solve step-by-step. Do not omit any steps. Explain in great detail. End with a separate final line of the form "
            "Answer: <answer>, using the answer form requested above. "
            r"Keep the answer on that same line, without math delimiters or \boxed{}. "
            "Write nothing after it."
        )

    @override
    def assistant_response(self, row: dict) -> str:
        raise NotImplementedError("HendrycksMathCoTResponseDataset is for evaluation only.")

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        answer = extract_math_answer(assistant_response, [])
        return answer, verify_math_answer(self._ground_truth_answer(row), answer)
