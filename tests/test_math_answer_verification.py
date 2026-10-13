"""Mathematical grading contracts using the real parser, without models or data files.

Expected answers come from elementary identities and Python's Fraction/Decimal
arithmetic. Most comparisons run in both directions; equation/assignment grading
has a separate section because the gold answer intentionally determines its form.
"""

from decimal import Decimal
from fractions import Fraction
from itertools import product
import signal

from math_verify.errors import TimeoutException
import pytest
from sympy import FiniteSet, Interval, Symbol, Tuple

from core.utils import cot_answers


def case(name, gold, prediction):
    return pytest.param(gold, prediction, id=name)


class TestNumericAnswers:
    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("integer-decimal", "42", "42.0"),
            case("leading-zeroes", "007", "7"),
            case("leading-plus", "+7", "7"),
            case("negative-decimal", "-12", "-12.0"),
            case("negative-zero", "-0", "0.0"),
            case("decimal-without-leading-zero", ".125", r"\frac{1}{8}"),
            case("fraction-reduction", r"\frac{18}{24}", r"\frac{3}{4}"),
            case("negative-numerator", r"\frac{-3}{4}", "-0.75"),
            case("negative-denominator", r"\frac{3}{-4}", "-0.75"),
            case("two-negatives", r"\frac{-3}{-4}", "0.75"),
            case("fraction-outside-sign", r"-\frac{3}{4}", "-0.75"),
            case("addition", "7+5", "12"),
            case("subtraction", "7-5", "2"),
            case("multiplication", r"7\times 5", "35"),
            case("division", "12/4", "3"),
            case("parenthesized-arithmetic", "(3+5)/2", "4"),
            case("double-negative", "-(-7)", "7"),
            case("integer-power", "2^{10}", "1024"),
            case("negative-power", "2^{-3}", r"\frac{1}{8}"),
            case("negative-base-even-power", "(-2)^4", "16"),
            case("scientific-notation", r"1.25\times 10^3", "1250"),
            case("small-scientific-notation", r"5\times 10^{-3}", "0.005"),
            case("square-root", r"\sqrt{81}", "9"),
            case("radical-simplification", r"\sqrt{12}", r"2\sqrt{3}"),
            case("rationalized-denominator", r"\frac{1}{\sqrt{2}}", r"\frac{\sqrt{2}}{2}"),
            case("cube-root", r"\sqrt[3]{27}", "3"),
            case("nested-fraction", r"\frac{1}{\frac{2}{3}}", r"\frac{3}{2}"),
            case("pi-coefficient", r"\frac{2\pi}{4}", r"\frac{\pi}{2}"),
            case("factorial", "5!", "120"),
            case("binomial", r"\binom{5}{2}", "10"),
            case("percentage", r"25\%", "0.25"),
            case("percentage-plain-sign", "25%", r"\frac{1}{4}"),
            case("negative-percentage", r"-25\%", "-0.25"),
            case("decimal-percentage", r"12.5\%", "0.125"),
            case("percentage-word", "25 percent", "0.25"),
            case("percentage-latex-text", r"25\text{percent}", "0.25"),
            case("imaginary-unit", r"\sqrt{-1}", "I"),
        ],
    )
    def test_equivalent_numbers(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is True
        assert cot_answers.verify_math_answer(prediction, gold) is True

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("wrong-integer", "42", "43"),
            case("wrong-sign", "-7", "7"),
            case("missing-digit", "1000", "100"),
            case("wrong-numerator", r"\frac{3}{7}", r"\frac{4}{7}"),
            case("wrong-denominator", r"\frac{3}{7}", r"\frac{3}{8}"),
            case("fraction-inverted", r"\frac{2}{3}", r"\frac{3}{2}"),
            case("percent-is-not-whole-number", "25", r"25\%"),
            case("power-precedence", "-2^2", "4"),
            case("wrong-radical", r"\sqrt{2}", r"\sqrt{3}"),
            case("square-root-is-principal", r"\sqrt{9}", "-3"),
            case("pi-is-not-three", r"\pi", "3"),
            case("wrong-exponent", "2^3", "2^4"),
            case("close-exact-rationals", "0", r"\frac{1}{10000000}"),
            case("different-exact-thirds", r"\frac{1}{3}", r"\frac{333333}{1000000}"),
            case("one-percent-not-one", r"1\%", "1"),
        ],
    )
    def test_non_equivalent_numbers(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is False
        assert cot_answers.verify_math_answer(prediction, gold) is False

    @pytest.mark.parametrize(
        ("gold", "prediction", "correct"),
        [
            ("1", "1.0000004", True),
            ("1", "1.0000006", False),
            ("0", "0.0000004", True),
            ("0", "0.0000006", False),
            (r"\frac{1}{3}", "0.333333", True),
            (r"\frac{1}{3}", "0.333334", False),
            (r"\frac{2}{3}", "0.666667", True),
            (r"\frac{2}{3}", "0.666666", False),
            ("-1", "-1.0000004", True),
            ("-1", "-1.0000006", False),
            ("1000", "1000.001", False),
            ("9007199254740993", "9007199254740993.0", True),
            ("9007199254740993", "9007199254740992.0", False),
            ("100000000000000001", "100000000000000001.0", True),
            ("100000000000000001", "100000000000000000.0", False),
        ],
    )
    def test_six_decimal_float_precision(self, gold, prediction, correct):
        assert cot_answers.verify_math_answer(gold, prediction) is correct
        assert cot_answers.verify_math_answer(prediction, gold) is correct

    @pytest.mark.parametrize(
        ("numerator", "denominator"), product((-17, -1, 0, 1, 2, 23, 999), (1, 2, 3, 7, 8, 25))
    )
    def test_scaled_fractions_against_fraction_oracle(self, numerator, denominator):
        reference = Fraction(numerator, denominator)
        gold = f"{reference.numerator}/{reference.denominator}"
        equivalent = rf"\frac{{{numerator * 13}}}{{{denominator * 13}}}"
        wrong = rf"\frac{{{numerator * 13 + 1}}}{{{denominator * 13}}}"
        assert cot_answers.verify_math_answer(gold, equivalent) is True
        assert cot_answers.verify_math_answer(equivalent, gold) is True
        assert cot_answers.verify_math_answer(gold, wrong) is False
        assert cot_answers.verify_math_answer(wrong, gold) is False

    @pytest.mark.parametrize(
        ("numerator", "denominator"), product((-17, -1, 0, 1, 2, 37), (1, 2, 4, 5, 8, 10, 20, 25, 50, 100, 125))
    )
    def test_terminating_decimals_against_decimal_oracle(self, numerator, denominator):
        reference = Decimal(numerator) / Decimal(denominator)
        gold = format(reference, "f")
        fraction = rf"\frac{{{numerator}}}{{{denominator}}}"
        wrong = format(reference + Decimal("0.01"), "f")
        assert cot_answers.verify_math_answer(gold, fraction) is True
        assert cot_answers.verify_math_answer(fraction, gold) is True
        assert cot_answers.verify_math_answer(gold, wrong) is False


class TestSymbolicAnswers:
    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("commutative-addition", "x+2", "2+x"),
            case("commutative-multiplication", "xy", "yx"),
            case("collect-like-terms", "x+x+x", "3x"),
            case("distribute", "2(x+3)", "2x+6"),
            case("factor-quadratic", "x^2+3x+2", "(x+1)(x+2)"),
            case("square-expansion", "(x+1)^2", "x^2+2x+1"),
            case("difference-of-squares", "x^2-y^2", "(x-y)(x+y)"),
            case("combine-fractions", r"\frac{x}{2}+\frac{x}{3}", r"\frac{5x}{6}"),
            case("symbolic-zero", "x-x", "0"),
            case("uppercase-expression", "A+A", "2A"),
            case("mixed-case-commutativity", "A+a", "a+A"),
            case("subscripted-symbol", "x_1+x_1", "2x_1"),
            case("greek-symbol", r"\alpha+\alpha", r"2\alpha"),
            case("trig-identity", r"\sin^2{x}+\cos^2{x}", "1"),
        ],
    )
    def test_equivalent_expressions(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is True
        assert cot_answers.verify_math_answer(prediction, gold) is True

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("different-variable", "x", "y"),
            case("uppercase-variable", "X", "x"),
            case("uppercase-ratio", "A/B", "a/b"),
            case("mixed-case-sum", "A+a", "2a"),
            case("greek-case", r"\Delta", r"\delta"),
            case("different-subscript", "x_1", "x_2"),
            case("different-power", "x^2", "x^3"),
            case("different-coefficient", "2x", "3x"),
            case("constant-offset", "x+1", "x+2"),
            case("wrong-sign", "x-y", "y-x"),
            case("product-is-not-sum", "xy", "x+y"),
            case("wrong-expansion", "(x+1)^2", "x^2+1"),
            case("different-function", r"\sin{x}", r"\cos{x}"),
            case("square-root-domain", r"\sqrt{x^2}", "x"),
        ],
    )
    def test_non_equivalent_expressions(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is False
        assert cot_answers.verify_math_answer(prediction, gold) is False


class TestStructuredAnswers:
    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("set-order", r"\{1,2,3\}", r"\{3,1,2\}"),
            case("set-duplicates", r"\{1,2\}", r"\{1,1,2\}"),
            case("set-equivalent-elements", r"\{\frac{1}{2},2\}", r"\{2,0.5\}"),
            case("bare-list-order", "4,6,14,15", "15,14,6,4"),
            case("open-interval", "(0,2)", r"(0,\frac{4}{2})"),
            case("closed-interval", "[0,2]", r"[0,\frac{4}{2}]"),
            case("half-open-interval", "[0,2)", r"[0,\frac{4}{2})"),
            case("infinite-interval", r"[0,\infty)", r"[0,\infty)"),
            case("interval-union-order", r"(-\infty,0)\cup(1,\infty)", r"(1,\infty)\cup(-\infty,0)"),
            case(
                "matrix-environment",
                r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
                r"\begin{bmatrix}1&2\\3&4\end{bmatrix}",
            ),
            case(
                "matrix-equivalent-entries",
                r"\begin{pmatrix}\frac{1}{2}&0\\0&2\end{pmatrix}",
                r"\begin{pmatrix}0.5&0\\0&2.0\end{pmatrix}",
            ),
        ],
    )
    def test_equivalent_structures(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is True
        assert cot_answers.verify_math_answer(prediction, gold) is True

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("set-missing-member", r"\{1,2,3\}", r"\{1,2\}"),
            case("set-extra-member", r"\{1,2\}", r"\{1,2,3\}"),
            case("set-wrong-member", r"\{1,2\}", r"\{1,3\}"),
            case("list-not-concatenation", "4,6,14,15", "461415"),
            case("interval-left-boundary", "[0,1]", "(0,1]"),
            case("interval-right-boundary", "[0,1]", "[0,1)"),
            case("interval-endpoint", "[0,1]", "[0,2]"),
            case("interval-is-not-endpoint-set", "(1,2)", r"\{1,2\}"),
            case("interval-is-not-bare-list", "(1,2)", "1,2"),
            case("reversed-endpoints", "(1,2)", "(2,1)"),
            case("union-missing-piece", r"(-\infty,0)\cup(1,\infty)", r"(-\infty,0)"),
            case(
                "matrix-wrong-entry",
                r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
                r"\begin{pmatrix}1&2\\3&5\end{pmatrix}",
            ),
            case(
                "matrix-transposed",
                r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
                r"\begin{pmatrix}1&3\\2&4\end{pmatrix}",
            ),
            case(
                "matrix-wrong-shape",
                r"\begin{pmatrix}1&2\\3&4\end{pmatrix}",
                r"\begin{pmatrix}1&2&3&4\end{pmatrix}",
            ),
        ],
    )
    def test_non_equivalent_structures(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is False
        assert cot_answers.verify_math_answer(prediction, gold) is False

    @pytest.mark.parametrize(
        ("gold", "prediction", "correct"),
        [
            ("x=2", "2", True),
            ("2", "x=2", True),
            ("x=2", "x=3", False),
            ("x=2", "y=2", False),
            pytest.param(
                "2x=4", "x=2", True,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Math-Verify 0.9.0 expects dicts from sympy.solve, which returns a list for one variable",
                ),
                id="known-upstream-scaled-equation",
            ),
            ("x<2", "2>x", True),
            ("x<=2", "2>=x", True),
            ("x<2", "x<=2", False),
            ("x<2", "x>2", False),
            ("x<2", "x<3", False),
        ],
    )
    def test_equations_and_relations(self, gold, prediction, correct):
        assert cot_answers.verify_math_answer(gold, prediction) is correct


class TestFormatting:
    @pytest.mark.parametrize("gold_wrapper", ["{}", "${}$", "$${}$$", r"\({}\)", r"\[{}\]"])
    @pytest.mark.parametrize("prediction_wrapper", ["{}", "${}$", "$${}$$", r"\({}\)", r"\[{}\]"])
    def test_math_delimiters_on_either_answer(self, gold_wrapper, prediction_wrapper):
        gold = gold_wrapper.format(r"\frac{3}{4}")
        prediction = prediction_wrapper.format("0.75")
        assert cot_answers.verify_math_answer(gold, prediction) is True

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("surrounding-whitespace", " \n42\t", "42.0 "),
            case("internal-expression-whitespace", "x + x", "2 x"),
            case("display-fraction", r"\frac{3}{4}", r"\dfrac{3}{4}"),
            case("text-fraction", r"\frac{3}{4}", r"\tfrac{3}{4}"),
            case("left-right-delimiters", r"\left(\frac{3}{4}\right)", "0.75"),
            case("latex-spacing", r"\frac{3}{4}\,", "0.75"),
            case("boxed-fraction", r"\frac{3}{4}", r"\boxed{\frac{3}{4}}"),
            case("boxed-scalar", "42", r"\boxed{42.0}"),
            case("boxed-set", r"\{1,2\}", r"\boxed{\{2,1\}}"),
            case("escaped-currency", r"\$40", "40"),
            case("source-multiline-text", "90\\text{ square\nunits}", r"90\text{ square units}"),
        ],
    )
    def test_equivalent_formatting(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is True

    @pytest.mark.parametrize("normalize_thousands", [False, True])
    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("positive-thousands", "1234567", "1,234,567"),
            case("negative-thousands", "-1234567", "-1,234,567"),
            case("decimal-thousands", "1234567.25", "1,234,567.25"),
            case("boxed-thousands", "1234567", r"\boxed{1,234,567}"),
        ],
    )
    def test_numeric_thousands(self, gold, prediction, normalize_thousands):
        assert cot_answers.verify_math_answer(gold, prediction, normalize_thousands=normalize_thousands) is True

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [("1234", "12,34"), ("123456", "1,23,456"), ("1000", "1,,000"), ("1000", "1,000,"), ("10000", "1,0000")],
    )
    def test_malformed_thousands_are_not_the_right_number(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction, normalize_thousands=True) is False

    @pytest.mark.parametrize("answer", ["no solution", "all real numbers", r"\text{None}", "yes"])
    def test_exact_literal_answers_do_not_require_parsing(self, monkeypatch, answer):
        def fail_if_called(*args, **kwargs):
            pytest.fail("An exact literal answer should not invoke mathematical parsing")

        monkeypatch.setattr(cot_answers, "_parse_math_answer", fail_if_called)
        monkeypatch.setattr(cot_answers, "verify", fail_if_called)
        assert cot_answers.verify_math_answer(f" {answer}\n", f"\t{answer} ") is True


class TestInvalidAnswersAndFailures:
    @pytest.mark.parametrize("empty", ["", " ", "\t\r\n", "\u2003"])
    @pytest.mark.parametrize("side", ["gold", "prediction", "both"])
    def test_empty_answers_are_never_correct(self, monkeypatch, empty, side):
        def fail_if_called(*args, **kwargs):
            pytest.fail("Empty answers should not invoke parsing")

        monkeypatch.setattr(cot_answers, "_parse_math_answer", fail_if_called)
        gold = empty if side in ("gold", "both") else "2"
        prediction = empty if side in ("prediction", "both") else "2"
        assert cot_answers.verify_math_answer(gold, prediction) is False

    @pytest.mark.parametrize(
        "invalid",
        [r"\frac{1}{", r"\sqrt{", r"\unknowncommand{2}", "not a mathematical answer", "[]", "()", "$", "$$", r"\(\)"],
    )
    @pytest.mark.parametrize("side", ["gold", "prediction"])
    def test_unparseable_answers_count_as_incorrect(self, invalid, side):
        gold, prediction = (invalid, "2") if side == "gold" else ("2", invalid)
        assert cot_answers.verify_math_answer(gold, prediction) is False

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [
            case("zero-over-zero", "0/0", r"\frac{0}{0}"),
            case("division-by-zero", "1/0", r"\frac{1}{0}"),
            case("undefined-not-zero", "0/0", "0"),
            case("undefined-not-one", "1/0", "1"),
        ],
    )
    def test_undefined_expressions_are_not_equivalent(self, gold, prediction):
        assert cot_answers.verify_math_answer(gold, prediction) is False

    @pytest.mark.parametrize("stage", ["gold", "prediction", "comparison"])
    @pytest.mark.parametrize("exception_cls", [ValueError, RuntimeError, TimeoutException])
    def test_parser_and_comparison_errors_do_not_escape(self, monkeypatch, stage, exception_cls):
        calls = 0
        real_parse = cot_answers._parse_math_answer

        def parse(answer, normalize_thousands):
            nonlocal calls
            calls += 1
            if calls == (1 if stage == "gold" else 2):
                raise exception_cls("simulated grading failure")
            return real_parse(answer, normalize_thousands)

        def fail_verify(*args, **kwargs):
            raise exception_cls("simulated grading failure")

        if stage == "comparison":
            monkeypatch.setattr(cot_answers, "verify", fail_verify)
        else:
            monkeypatch.setattr(cot_answers, "_parse_math_answer", parse)
        assert cot_answers.verify_math_answer("0.5", r"\frac{1}{2}") is False

    @pytest.mark.parametrize("exception_cls", [KeyboardInterrupt, SystemExit])
    def test_process_control_exceptions_are_not_swallowed(self, monkeypatch, exception_cls):
        def interrupted(*args, **kwargs):
            raise exception_cls("stop grading")

        monkeypatch.setattr(cot_answers, "_parse_math_answer", interrupted)
        with pytest.raises(exception_cls):
            cot_answers.verify_math_answer("0.5", r"\frac{1}{2}")

    @pytest.mark.parametrize("none_on_call", [1, 2])
    def test_missing_parse_result_counts_as_incorrect(self, monkeypatch, none_on_call):
        calls = 0

        def parse(*args):
            nonlocal calls
            calls += 1
            return None if calls == none_on_call else Symbol("x")

        monkeypatch.setattr(cot_answers, "_parse_math_answer", parse)
        assert cot_answers.verify_math_answer("x", "x+x") is False

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX parser timeout uses SIGALRM")
    @pytest.mark.parametrize("parser_raises", [False, True])
    def test_parsing_restores_the_alarm_handler(self, monkeypatch, parser_raises):
        def parser(*args, **kwargs):
            if parser_raises:
                raise ValueError("invalid")
            return Symbol("x")

        original = signal.getsignal(signal.SIGALRM)
        monkeypatch.setattr(cot_answers, "latex2sympy", parser)
        if parser_raises:
            with pytest.raises(ValueError):
                cot_answers._parse_math_answer("x", False)
        else:
            cot_answers._parse_math_answer("x", False)
        assert signal.getsignal(signal.SIGALRM) == original
        assert signal.alarm(0) == 0

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX parser timeout uses SIGALRM")
    @pytest.mark.parametrize("timeout_on_call", [1, 2])
    def test_real_parser_timeout_is_incorrect_and_restores_signals(self, monkeypatch, timeout_on_call):
        calls = 0
        original = signal.getsignal(signal.SIGALRM)

        def parser(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == timeout_on_call:
                # Exercise the actual installed timeout handler without a slow
                # sleep or expensive symbolic calculation.
                signal.raise_signal(signal.SIGALRM)
                pytest.fail("The timeout signal should interrupt parsing")
            return Symbol("x")

        monkeypatch.setattr(cot_answers, "latex2sympy", parser)
        assert cot_answers.verify_math_answer("x", "x+x") is False
        assert signal.getsignal(signal.SIGALRM) == original
        assert signal.alarm(0) == 0

    @pytest.mark.parametrize(
        ("gold", "prediction"),
        [(Tuple(1, 2), FiniteSet(1, 2)), (Interval.open(1, 2), FiniteSet(1, 2))],
    )
    def test_structural_mismatch_cannot_reach_the_permissive_grader(self, monkeypatch, gold, prediction):
        parsed = iter([gold, prediction])
        monkeypatch.setattr(cot_answers, "_parse_math_answer", lambda *args: next(parsed))

        def fail_if_called(*args, **kwargs):
            pytest.fail("An incompatible mathematical structure should be rejected before comparison")

        monkeypatch.setattr(cot_answers, "verify", fail_if_called)
        assert cot_answers.verify_math_answer("gold", "prediction") is False
