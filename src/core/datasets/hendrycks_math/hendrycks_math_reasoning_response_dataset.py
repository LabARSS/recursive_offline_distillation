from typing import override

from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset
from core.datasets.qa_dataset import InvalidAnswerError
from core.utils.cot_answers import clean_assistant_response


class HendrycksMathReasoningResponseDataset(HendrycksMathDirectResponseDataset):
    """Distilled solutions using the tokenizer's thinking markers."""

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
