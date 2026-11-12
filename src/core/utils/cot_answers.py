"""Assistant response cleanup, CoT answer extraction, and mathematical grading."""

import re
from collections.abc import Iterable

from latex2sympy2_extended import NormalizationConfig, latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig
from math_verify import verify
from math_verify.errors import TimeoutException
from math_verify.utils import timeout
from sympy import FiniteSet, Float, Interval, Number, Rational, Tuple, UnevaluatedExpr


CHAT_END_TOKENS = ("<|end|>", "<|endoftext|>", "<|im_end|>", "<|eot_id|>")


def clean_assistant_response(response: str, special_tokens: Iterable[str]) -> str:
    """Keep the first assistant turn and remove tokenizer control tokens."""
    # Some chat-end markers (notably Phi-4-mini's <|end|>) are absent from
    # all_special_tokens. Cut before stripping so later generated turns cannot
    # replace the answer from the actual response.
    end_positions = [position for token in CHAT_END_TOKENS if (position := response.find(token)) != -1]
    if end_positions:
        response = response[: min(end_positions)]
    for token in sorted(set(special_tokens), key=len, reverse=True):
        if token:
            response = response.replace(token, "")
    return response.strip()


def final_response_line(response: str, special_tokens: Iterable[str]) -> str:
    """Return the last nonempty line of the first assistant turn."""
    response = clean_assistant_response(response, special_tokens)
    return next((line.strip() for line in reversed(response.splitlines()) if line.strip()), "")


def extract_math_answer(response: str, special_tokens: Iterable[str]) -> str:
    match = re.fullmatch(r"Answer:\s*(.+)", final_response_line(response, special_tokens), re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _math_content(answer: str, normalize_thousands: bool) -> str:
    for opening, closing in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if answer.startswith(opening) and answer.endswith(closing) and len(answer) >= len(opening + closing):
            answer = answer[len(opening) : -len(closing)].strip()
            break
    if normalize_thousands:
        # Only GSM8K treats commas between numeric groups as thousands separators.
        # MATH also uses commas for coordinates and sets, so keep them there.
        answer = re.sub(
            r"(?<![\d,])([+-]?\d{1,3}(?:,\d{3})+)(?![\d,])",
            lambda match: match.group(1).replace(",", ""),
            answer,
        )
    return answer


@timeout(timeout_seconds=5)
def _parse_math_answer(answer: str, normalize_thousands: bool):
    # Math-Verify's default extraction parser lowercases symbols. Its LaTeX
    # parser exposes a case-preserving conversion mode when called directly.
    parsed = latex2sympy(
        _math_content(answer, normalize_thousands),
        normalization_config=NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="all",
            equations=False,
        ),
        conversion_config=ConversionConfig(lowercase_symbols=False),
    )
    # The parser leaves percentages unevaluated, and Math-Verify compares integer
    # percentages with their unscaled integer. Evaluate just that percentage
    # marker so 25% becomes 1/4 before grading.
    return parsed.xreplace({UnevaluatedExpr(Rational(1, 100)): Rational(1, 100)})


def _compatible_math_answers(gold, prediction) -> bool:
    # The grader intentionally treats open intervals like pairs of endpoints.
    # That would also accept a finite set in place of an interval here.
    for left, right in ((gold, prediction), (prediction, gold)):
        if isinstance(left, Interval) and isinstance(right, (FiniteSet, Tuple)):
            return False
        if isinstance(left, Tuple) and isinstance(right, FiniteSet):
            return False
    # The grader also compares some multi-letter symbol names case-insensitively.
    gold_names = {symbol.name for symbol in gold.free_symbols}
    prediction_names = {symbol.name for symbol in prediction.free_symbols}
    shared_folded_names = {name.casefold() for name in gold_names} & {
        name.casefold() for name in prediction_names
    }
    return all(name.casefold() not in shared_folded_names for name in gold_names ^ prediction_names)


def verify_math_answer(gold: str, prediction: str, *, normalize_thousands: bool = False) -> bool:
    """Compare isolated final answers; retain exact matches for literal answers."""
    gold, prediction = gold.strip(), prediction.strip()
    if not gold or not prediction:
        return False
    if gold == prediction:
        return True

    try:
        parsed_gold = _parse_math_answer(gold, normalize_thousands)
        parsed_prediction = _parse_math_answer(prediction, normalize_thousands)
        if parsed_gold is None or parsed_prediction is None:
            return False
        if not _compatible_math_answers(parsed_gold, parsed_prediction):
            return False
        if isinstance(parsed_gold, Number) and isinstance(parsed_prediction, Number) and (
            isinstance(parsed_gold, Float) or isinstance(parsed_prediction, Float)
        ):
            # Compare rounded decimal values as rationals: Integer(1) and
            # Float(1.0) are structurally unequal, while converting both to a
            # fixed-precision float would lose digits from large integers.
            parsed_gold = Rational(str(parsed_gold.round(6)))
            parsed_prediction = Rational(str(parsed_prediction.round(6)))
        return bool(
            verify(
                parsed_gold,
                parsed_prediction,
                strict=True,
                float_rounding=6,
                numeric_precision=15,
                timeout_seconds=5,
            )
        )
    except (Exception, TimeoutException):
        # Unsupported notation and timeouts are failed measurements, not
        # reasons to crash or restart an evaluation worker.
        return False
