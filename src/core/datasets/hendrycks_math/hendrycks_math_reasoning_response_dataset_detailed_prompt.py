from typing import override

from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset
from core.datasets.qa_dataset import InvalidAnswerError
from core.utils.cot_answers import clean_assistant_response


class HendrycksMathReasoningResponseDatasetDetailedPrompt(HendrycksMathDirectResponseDataset):
    """Distilled solutions using the tokenizer's thinking markers."""

    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are competition mathematics problems. "
            "Start with <think>, solve the problem step-by-step, then put the final answer after </think>.\n\n"
            "Required format: <think>...</think><answer>\n"
            "Replace <answer> with your final answer in the form requested by the question. "
            "Use LaTeX for mathematical expressions. The final line must start with the "
            "</think> followed by the answer on the same line. "
            "Do not put prose, a list number, Markdown formatting, math delimiters, "
            r"or a \boxed{} wrapper in the final answer."
            "Stop immediately after the final answer.\n\n"
            "Example response:\n"
            "<think>Dividing both sides of 2x = 1 by 2 gives x = 1/2.</think>\frac{1}{2}\n"
        )

    @override
    def user_prompt(self, row: dict) -> str:
        return (
            f"{super().user_prompt(row)}\n\n"
            "Start with <think>. Solve step-by-step. End with </think><answer>, using the answer form requested above."
            r"Keep the answer on that same line, without math delimiters or \boxed{}. "
            "Write nothing after it."
        )

    @override
    def assistant_response(self, row: dict) -> str:
        reasoning_chain = row["distill_reasoning"].strip()
        answer = str(row["distill_answer"]).strip()
        return f"{self.tokenizer.thinking_start_token}{reasoning_chain}{self.tokenizer.thinking_end_token}{answer}"

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        assert isinstance(self.tokenizer.thinking_end_token, str), (
            "Tokenizer must have a defined thinking_end_token to use HendrycksMathReasoningResponseDataset"
        )
        # Cut at chat-end markers while retaining </think> for answer extraction.
        assistant_response = clean_assistant_response(assistant_response, ())
        thinking_end_token_position = assistant_response.find(self.tokenizer.thinking_end_token)
        if thinking_end_token_position == -1:
            return super().verify_assistant_response(
                row, clean_assistant_response(assistant_response, self.tokenizer.all_special_tokens)
            )

        raw = assistant_response[thinking_end_token_position + len(self.tokenizer.thinking_end_token) :]
        extracted_answer = clean_assistant_response(raw, self.tokenizer.all_special_tokens)

        try:
            return super().verify_assistant_response(row, extracted_answer)
        except InvalidAnswerError:
            return extracted_answer, False
