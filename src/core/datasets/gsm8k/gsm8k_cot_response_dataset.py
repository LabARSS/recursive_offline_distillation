from typing import override

from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset
from core.utils.cot_answers import extract_math_answer, verify_math_answer


class GSM8KCoTResponseDataset(GSM8KDirectResponseDataset):
    """Grade-school math with prompted reasoning and a marked final answer."""

    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are grade school math word problems. "
            "Solve the problem step-by-step, then put the final answer on a new line.\n\n"
            "Required final-line format: Answer: <answer>\n"
            "Replace <answer> with the final number. The final line must start with the "
            "literal text Answer: followed by a space and the number on the same line. "
            "Use plain digits, with a minus sign or decimal point if needed. "
            "Do not put prose, units, currency symbols, thousands separators, a list number, "
            "or Markdown formatting on that line. Stop immediately after the final number.\n\n"
            "Example response:\n"
            "There are 3 groups of 4 items, so the total is 3 * 4 = 12.\n"
            "Answer: 12"
        )

    @override
    def user_prompt(self, row: dict) -> str:
        return (
            f"{super().user_prompt(row)}\n\n"
            "Solve step-by-step. End with a separate final line of the form "
            "Answer: <answer>, replacing <answer> with only the final number. "
            "Keep the number on that same line, without units, currency symbols, "
            "or Markdown formatting. Write nothing after it."
        )

    @override
    def assistant_response(self, row: dict) -> str:
        raise NotImplementedError("GSM8KCoTResponseDataset is for evaluation only.")

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        answer = extract_math_answer(assistant_response, self.tokenizer.all_special_tokens)
        return answer, verify_math_answer(self._ground_truth_answer(row), answer, normalize_thousands=True)
