from typing import override

from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset


class HendrycksMathCorrectedAnswerDataset(HendrycksMathDirectResponseDataset):
    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are competition mathematics problems. "
            "You will be shown a question, a partial chain-of-thought that was previously produced for it, "
            "the incorrect answer that chain-of-thought arrived at, and the correct answer. "
            "Continue the partial chain-of-thought from where it left off so that it naturally arrives at "
            "the correct answer. Do not restart the reasoning, do not repeat what was already written, and "
            "do not acknowledge that the previous attempt was wrong or that the correct answer was given to "
            "you. Write the continuation as if you were the same reasoner noticing a mistake or new "
            "consideration mid-thought and revising course. "
            "End your response with Answer: <answer>, using LaTeX for mathematical expressions "
            "and following any answer-format instructions in the question."
        )

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"].strip()
        original_reasoning = str(row["distill_reasoning"]).strip()
        original_answer = str(row["distill_answer"]).strip()
        return (
            f"Question: {question}\n"
            f"Partial reasoning so far:\n{original_reasoning}\n"
            f"Incorrect answer this reasoning led to: {original_answer}\n"
            f"Correct answer: {self.assistant_response(row)}\n"
        )

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        answer = assistant_response.rsplit("Answer:", 1)[-1]
        return super().verify_assistant_response(row, answer)
