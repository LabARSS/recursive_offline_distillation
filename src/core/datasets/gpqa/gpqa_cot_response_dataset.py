import ast
import re
from typing import override

from core.datasets.gpqa.gpqa_single_token_response_dataset import GPQASingleTokenResponseDataset
from core.utils.cot_answers import clean_assistant_response


class GPQACoTResponseDataset(GPQASingleTokenResponseDataset):
    """Multiple-choice reasoning with a final option in MMLU's [[letter]] format."""

    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are multiple choice questions. "
            "Explain your solution step-by-step, then put the chosen option letter on a new line.\n\n"
            "Required final-line format: [[letter]]\n"
            "Replace letter with exactly one lowercase letter from the provided options. "
            "Include both pairs of square brackets. The final line must contain only "
            "the bracketed letter, without prose, an answer prefix, option text, a list number, "
            "or Markdown formatting. Stop immediately after the bracketed letter.\n\n"
            "Example response:\n"
            "The calculation gives 4, which matches option b.\n"
            "[[b]]"
        )

    @override
    def user_prompt(self, row: dict) -> str:
        return (
            f"{super().user_prompt(row)}\n"
            "Reason step-by-step. End with a separate final line of the form [[letter]], "
            "replacing letter with the chosen lowercase option letter. "
            "Put only the bracketed letter on that line. Write nothing after it."
        )

    @override
    def assistant_response(self, row: dict) -> str:
        raise NotImplementedError("GPQACoTResponseDataset is for evaluation only.")

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        assistant_response = clean_assistant_response(assistant_response, self.tokenizer.all_special_tokens)
        # Models can put prose around the answer or explain it afterward.
        # Use the last marker, so a revision overrides an earlier answer;
        # an invalid/incomplete final marker must not revive an earlier one.
        answer_start = assistant_response.rfind("[[")
        if answer_start == -1:
            return "", False
        match = re.match(r"\[\[\s*([a-zA-Z])\s*\]\]", assistant_response[answer_start:])
        if not match:
            return "", False
        answer = match.group(1).lower()
        valid_options = self.option_ids[: len(ast.literal_eval(row["options"]))]
        if answer not in valid_options:
            return "", False
        return answer, self._ground_truth_answer(row) == answer
