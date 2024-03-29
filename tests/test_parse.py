import math
from unittest import TestCase

from latexpr import Math


class TestParse(TestCase):
    # GOOD_PAIRS = [
    #     ("-3.14", _Mul(-1, 3.14)),
    #     ("(-7.13)(1.5)", _Mul(_Mul(-1, 7.13), 1.5)),
    #     ("x", x),
    #     ("2x", 2 * x),
    #     ("x^2", x ** 2),
    #     ("x^{3 + 1}", x ** _Add(3, 1)),
    #     ("-c", -c),
    #     ("a \\cdot b", a * b),
    #     ("a / b", a / b),
    #     ("a \\div b", a / b),
    #     ("a + b", a + b),
    #     ("a + b - a", _Add(a + b, -a)),
    #     ("a^2 + b^2 = c^2", Eq(a ** 2 + b ** 2, c ** 2)),
    #     ("\\sin \\theta", sin(theta)),
    #     ("\\sin(\\theta)", sin(theta)),
    #     ("\\sin^{-1} a", asin(a)),
    #     ("\\sin a \\cos b", _Mul(sin(a), cos(b))),
    #     ("\\sin \\cos \\theta", sin(cos(theta))),
    #     ("\\sin(\\cos \\theta)", sin(cos(theta))),
    #     ("\\frac{a}{b}", a / b),
    #     ("\\frac{a + b}{c}", _Mul(a + b, _Pow(c, -1))),
    #     ("\\frac{7}{3}", _Mul(7, _Pow(3, -1))),
    #     ("(\\csc x)(\\sec y)", csc(x) * sec(y)),
    #     ("\\lim_{x \\to 3} a", Limit(a, x, 3)),
    #     ("\\lim_{x \\rightarrow 3} a", Limit(a, x, 3)),
    #     ("\\lim_{x \\Rightarrow 3} a", Limit(a, x, 3)),
    #     ("\\lim_{x \\longrightarrow 3} a", Limit(a, x, 3)),
    #     ("\\lim_{x \\Longrightarrow 3} a", Limit(a, x, 3)),
    #     ("\\lim_{x \\to 3^{+}} a", Limit(a, x, 3, dir = '+')),
    #     ("\\lim_{x \\to 3^{-}} a", Limit(a, x, 3, dir = '-')),
    #     ("\\infty", oo),
    #     ("\\lim_{x \\to \\infty} \\frac{1}{x}", Limit(_Mul(1, _Pow(x, -1)), x, oo)),
    #     ("\\frac{d}{dx} x", Derivative(x, x)),
    #     ("\\frac{d}{dt} x", Derivative(x, t)),
    #     ("f(x)", f(x)),
    #     ("f(x, y)", f(x, y)),
    #     ("f(x, y, z)", f(x, y, z)),
    #     ("\\frac{d f(x)}{dx}", Derivative(f(x), x)),
    #     ("\\frac{d\\theta(x)}{dx}", Derivative(theta(x), x)),
    #     ("|x|", _Abs(x)),
    #     ("||x||", _Abs(Abs(x))),
    #     ("|x||y|", _Abs(x) * _Abs(y)),
    #     ("||x||y||", _Abs(_Abs(x) * _Abs(y))),
    #     ("\pi^{|xy|}", Symbol('pi') ** _Abs(x * y)),
    #     ("\\int x dx", Integral(x, x)),
    #     ("\\int x d\\theta", Integral(x, theta)),
    #     ("\\int (x^2 - y)dx", Integral(x ** 2 - y, x)),
    #     ("\\int x + a dx", Integral(_Add(x, a), x)),
    #     ("\\int da", Integral(1, a)),
    #     ("\\int_0^7 dx", Integral(1, (x, 0, 7))),
    #     ("\\int_a^b x dx", Integral(x, (x, a, b))),
    #     ("\\int^b_a x dx", Integral(x, (x, a, b))),
    #     ("\\int_{a}^b x dx", Integral(x, (x, a, b))),
    #     ("\\int^{b}_a x dx", Integral(x, (x, a, b))),
    #     ("\\int_{a}^{b} x dx", Integral(x, (x, a, b))),
    #     ("\\int^{b}_{a} x dx", Integral(x, (x, a, b))),
    #     ("\\int_{f(a)}^{f(b)} f(z) dz", Integral(f(z), (z, f(a), f(b)))),
    #     ("\\int (x+a)", Integral(_Add(x, a), x)),
    #     ("\\int a + b + c dx", Integral(_Add(_Add(a, b), c), x)),
    #     ("\\int \\frac{dz}{z}", Integral(Pow(z, -1), z)),
    #     ("\\int \\frac{3 dz}{z}", Integral(3 * Pow(z, -1), z)),
    #     ("\\int \\frac{1}{x} dx", Integral(Pow(x, -1), x)),
    #     ("\\int \\frac{1}{a} + \\frac{1}{b} dx", Integral(_Add(_Pow(a, -1), Pow(b, -1)), x)),
    #     ("\\int \\frac{3 \cdot d\\theta}{\\theta}", Integral(3 * _Pow(theta, -1), theta)),
    #     ("\\int \\frac{1}{x} + 1 dx", Integral(_Add(_Pow(x, -1), 1), x)),
    #     ("x_0", Symbol('x_{0}')),
    #     ("x_{1}", Symbol('x_{1}')),
    #     ("x_a", Symbol('x_{a}')),
    #     ("x_{b}", Symbol('x_{b}')),
    #     ("h_\\theta", Symbol('h_{theta}')),
    #     ("h_{\\theta}", Symbol('h_{theta}')),
    #     ("h_{\\theta}(x_0, x_1)", Symbol('h_{theta}')(Symbol('x_{0}'), Symbol('x_{1}'))),
    #     ("x!", _factorial(x)),
    #     ("100!", _factorial(100)),
    #     ("\\theta!", _factorial(theta)),
    #     ("(x + 1)!", _factorial(_Add(x, 1))),
    #     ("(x!)!", _factorial(_factorial(x))),
    #     ("x!!!", _factorial(_factorial(_factorial(x)))),
    #     ("5!7!", _Mul(_factorial(5), _factorial(7))),
    #     ("\\sqrt{x}", sqrt(x)),
    #     ("\\sqrt{x + b}", sqrt(_Add(x, b))),
    #     ("\\sqrt[3]{\\sin x}", root(sin(x), 3)),
    #     ("\\sqrt[y]{\\sin x}", root(sin(x), y)),
    #     ("\\sqrt[\\theta]{\\sin x}", root(sin(x), theta)),
    #     ("x < y", StrictLessThan(x, y)),
    #     ("x \\leq y", LessThan(x, y)),
    #     ("x > y", StrictGreaterThan(x, y)),
    #     ("x \\geq y", GreaterThan(x, y)),
    #     ("\\mathit{x}", Symbol('x')),
    #     ("\\mathit{test}", Symbol('test')),
    #     ("\\mathit{TEST}", Symbol('TEST')),
    #     ("\\mathit{HELLO world}", Symbol('HELLO world')),
    #     ("\\sum_{k = 1}^{3} c", Sum(c, (k, 1, 3))),
    #     ("\\sum_{k = 1}^3 c", Sum(c, (k, 1, 3))),
    #     ("\\sum^{3}_{k = 1} c", Sum(c, (k, 1, 3))),
    #     ("\\sum^3_{k = 1} c", Sum(c, (k, 1, 3))),
    #     ("\\sum_{k = 1}^{10} k^2", Sum(k ** 2, (k, 1, 10))),
    #     ("\\sum_{n = 0}^{\\infty} \\frac{1}{n!}", Sum(_Pow(_factorial(n), -1), (n, 0, oo))),
    #     ("\\prod_{a = b}^{c} x", Product(x, (a, b, c))),
    #     ("\\prod_{a = b}^c x", Product(x, (a, b, c))),
    #     ("\\prod^{c}_{a = b} x", Product(x, (a, b, c))),
    #     ("\\prod^c_{a = b} x", Product(x, (a, b, c))),
    #     ("\\ln x", _log(x, E)),
    #     ("\\ln xy", _log(x * y, E)),
    #     ("\\log x", _log(x, 10)),
    #     ("\\log xy", _log(x * y, 10)),
    #     ("\\log_2 x", _log(x, 2)),
    #     ("\\log_{2} x", _log(x, 2)),
    #     ("\\log_a x", _log(x, a)),
    #     ("\\log_{a} x", _log(x, a)),
    #     ("\\log_{11} x", _log(x, 11)),
    #     ("\\log_{a^2} x", _log(x, _Pow(a, 2))),
    #     ("[x]", x),
    #     ("[a + b]", _Add(a, b)),
    #     ("\\frac{d}{dx} [ \\tan x ]", Derivative(tan(x), x))
    # ]

    def test_constant_zero(self):
        result = Math.parse(latex = "0")
        assert result() == 0

    def test_constant_one(self):
        result = Math.parse(latex = "1")
        assert result() == 1

    def test_constant_negative(self):
        result = Math.parse(latex = "-3.14")
        assert result() == -3.14

    def test_large_constant(self):
        result = Math.parse(latex = "1000")
        assert result() == 1000

    def test_mult(self):
        result = Math.parse(latex = "(-7.13)(1.5)")
        assert result() == -7.13 * 1.5

    def test_var(self):
        result = Math.parse(latex = "x")
        assert result(x = 1) == 1

    def test_var_neg(self):
        result = Math.parse(latex = "-x")
        assert result(x = 1) == -1

    def test_var_mult(self):
        result = Math.parse(latex = "2x")
        assert result(x = 3) == 2 * 3

    def test_dot_var_mult(self):
        result = Math.parse(latex = "a \\cdot b")
        assert result(a = 2, b = 3) == 2 * 3

    def test_var_div(self):
        result = Math.parse(latex = "a / b")
        assert result(a = 10, b = 2) == 5

    def test_var_div2(self):
        result = Math.parse(latex = "a \\div b")
        assert result(a = 10, b = 2) == 5

    def test_var_add(self):
        result = Math.parse(latex = "a + b")
        assert result(a = 10, b = 2) == 12

    def test_var_add_multiple(self):
        result = Math.parse(latex = "a + b - a")
        assert result(a = 10, b = 2) == 2

    def test_var_pow(self):
        result = Math.parse(latex = "x^3")
        assert result(x = "3") == 3 ** 3

    def test_var_pow_complex(self):
        result = Math.parse(latex = "x^{3 + 1}")
        assert result(x = 3) == 3 ** 4

    def test_sqrt_simple(self):
        result = Math.parse(latex = "\\sqrt{x}")
        assert result(x = 9) == 3

    def test_sqrt_complex(self):
        result = Math.parse(latex = "\\sqrt{x + b}")
        assert result(x = 4, b = 5) == 3

    def test_sin(self):
        result = Math.parse(latex = "\\sin(\\theta)")
        assert result(theta = math.pi / 2) == 1

    def test_cos(self):
        result = Math.parse(latex = "\\cos(\\theta)")
        assert result(theta = 0) == 1

    def test_frac(self):
        result = Math.parse(latex = "\\frac {a + b} {c}")
        assert result(a = 1, b = 2, c = 3) == 1

    def test_func(self):
        result = Math.parse(latex = "f(x) = x")
        x = result(x = 1)
        print(x)
