from typing import override

from core.datasets.gpqa.gpqa_single_token_response_dataset import GPQASingleTokenResponseDataset
from core.utils.cot_answers import clean_assistant_response


class GPQAEmptyThinkingResponseDataset(GPQASingleTokenResponseDataset):
    """Direct option letter after an empty thinking block: <think></think>letter.

    Mixed with reasoning traces that share the prompt, so the first assistant token is always
    <think>; the empty block marks a direct answer instead of a competing first token.
    """

    @override
    def assistant_response(self, row: dict) -> str:
        empty_thinking = f"{self.tokenizer.thinking_start_token}{self.tokenizer.thinking_end_token}"
        return f"{empty_thinking}{self._ground_truth_answer(row)}"

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        end = assistant_response.rfind(self.tokenizer.thinking_end_token)
        if end != -1:
            assistant_response = assistant_response[end + len(self.tokenizer.thinking_end_token) :]
        return super().verify_assistant_response(
            row, clean_assistant_response(assistant_response, self.tokenizer.all_special_tokens)
        )
