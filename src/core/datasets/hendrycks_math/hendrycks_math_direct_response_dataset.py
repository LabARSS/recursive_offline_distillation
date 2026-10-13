from typing import override

from core.datasets.qa_dataset import InvalidAnswerError, QADataset, QADatasetConfig
from core.utils.cot_answers import verify_math_answer


class HendrycksMathDirectResponseDataset(QADataset[QADatasetConfig]):
    """Competition math with final answers in the source artifact's LaTeX form."""

    @override
    def system_prompt(self, row: dict) -> str:
        return (
            "The following are competition mathematics problems. "
            "Return only the final answer in the form requested by the question. "
            "Use LaTeX for mathematical expressions, without surrounding math delimiters "
            r"or a \boxed{} wrapper. Do not include an explanation."
        )

    @override
    def user_prompt(self, row: dict) -> str:
        return row["question"].strip()

    @override
    def assistant_response(self, row: dict) -> str:
        return self._ground_truth_answer(row)

    @override
    def row_id(self, row: dict) -> str:
        return str(row["question_id"])

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        """Compare final answers mathematically, preserving variable case."""
        parsed_answer = assistant_response.strip()
        if not parsed_answer:
            raise InvalidAnswerError("empty math answer (model likely emitted a thinking/special token)")
        return parsed_answer, verify_math_answer(self._ground_truth_answer(row), parsed_answer)

    def _ground_truth_answer(self, row: dict) -> str:
        return str(row["answer"]).strip()
