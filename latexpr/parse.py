import os
import re
from typing import List

import sympy
import antlr4
from sympy.core.numbers import One, Zero, Integer
from sympy.parsing.latex._parse_latex_antlr import MathErrorListener

from latexpr.ps.PSParser import PSParser
from latexpr.ps.PSLexer import PSLexer

from sympy.printing.str import StrPrinter


class Math(object):
    @staticmethod
    def parse(latex: str):
        return Math.__process_sympy(latex)

    @staticmethod
    def parse_mult_file(file: str, tag: List[str]):
        if not os.path.exists(file):
            raise FileNotFoundError("Could not find parse file: " + file)
        with open(file, 'r') as f:
            did_find = False
            cur_eq = ""
            found_funcs = []
            for line in f:
                if line.startswith("```{eq:"):
                    m = re.search("```{eq:([a-zA-Z0-9_-]+)}", line)
                    if m:
                        found = m.group(1)
                        if found in tag:
                            did_find = True
                            found_funcs.append(found)
                elif line.startswith("```") and did_find:
                    # We reached the end of the equation.
                    did_find = False
                    yield found_funcs[-1], Math.parse(latex = cur_eq)
                    cur_eq = ""

                elif did_find:
                    # Still reading equation
                    cur_eq += line

            for t in tag:
                fail_str = "Did not find function with name: {0}".format(t)
                assert t in found_funcs, fail_str

    @staticmethod
    def parse_file(file: str, tag: str):
        for t, f in Math.parse_mult_file(file, [tag]):
            # We just return the first one because we only asked for 1.
            return f

    @staticmethod
    def __process_sympy(latex):
        matherror = MathErrorListener(latex)

        stream = antlr4.InputStream(latex)
        lex = PSLexer(stream)
        lex.removeErrorListeners()
        lex.addErrorListener(matherror)

        tokens = antlr4.CommonTokenStream(lex)
        parser = PSParser(tokens)

        # remove default console error listener
        parser.removeErrorListeners()
        parser.addErrorListener(matherror)

        relation = parser.math().relation()
        expr = Math.__convert_relation(relation)

        if type(expr) in [int, float, One, Zero] or issubclass(Integer, type(expr)):
            return lambda: expr
        else:
            def fn(**kwargs):
                if type(expr) in [sympy.Integral, sympy.Limit]:
                    return expr.doit(**kwargs)
                else:
                    return expr.evalf(subs=kwargs)

            return fn

    @staticmethod
    def __convert_relation(rel: PSParser.RelationContext):
        if rel.expr():
            return Math.__convert_expr(rel.expr())

        lh = Math.__convert_relation(rel.relation(0))
        rh = Math.__convert_relation(rel.relation(1))
        if rel.LT():
            return sympy.StrictLessThan(lh, rh)
        elif rel.LTE():
            return sympy.LessThan(lh, rh)
        elif rel.GT():
            return sympy.StrictGreaterThan(lh, rh)
        elif rel.GTE():
            return sympy.GreaterThan(lh, rh)
        elif rel.EQUAL():
            return sympy.Eq(lh, rh)
        else:
            raise NotImplementedError("Function implemented")

    @staticmethod
    def __convert_expr(expr: PSParser.ExprContext):
        if expr.additive():
            return Math.__convert_add(expr.additive())
        else:
            raise NotImplementedError("Could not understand expression")

    @staticmethod
    def __convert_add(add: PSParser.AdditiveContext):
        if add.ADD() or add.SUB():
            lhs, rhs = (Math.__convert_add(add.additive(i = 0)),
                        Math.__convert_add(add.additive(i = 1)) if not add.SUB() else -1 * Math.__convert_add(add.additive(i = 1)))
        else:
            return Math.__convert_mp(add.mp())

        return sympy.Add(lhs, rhs, evaluate=False)

    @staticmethod
    def __convert_mp(mp):
        if hasattr(mp, 'mp'):
            mp_left = mp.mp(0)
            mp_right = mp.mp(1)
        else:
            mp_left = mp.mp_nofunc(0)
            mp_right = mp.mp_nofunc(1)

        if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
            lh = Math.__convert_mp(mp_left)
            rh = Math.__convert_mp(mp_right)
            return sympy.Mul(lh, rh, evaluate = False)
        elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
            lh = Math.__convert_mp(mp_left)
            rh = Math.__convert_mp(mp_right)
            return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate = False), evaluate = False)
        else:
            if hasattr(mp, 'unary'):
                return Math.__convert_unary(mp.unary())
            else:
                return Math.__convert_unary(mp.unary_nofunc())

    @staticmethod
    def __convert_unary(unary):
        if hasattr(unary, 'unary'):
            nested_unary = unary.unary()
        else:
            nested_unary = unary.unary_nofunc()
        if hasattr(unary, 'postfix_nofunc'):
            first = unary.postfix()
            tail = unary.postfix_nofunc()
            postfix = [first] + tail
        else:
            postfix = unary.postfix()

        if unary.ADD():
            return Math.__convert_unary(nested_unary)
        elif unary.SUB():
            return sympy.Mul(-1, Math.__convert_unary(nested_unary), evaluate = False)
        elif postfix:
            return Math.__convert_postfix_list(postfix)

    @staticmethod
    def __convert_postfix_list(arr, i = 0):
        if i >= len(arr):
            raise Exception("Index out of bounds")

        res = Math.__convert_postfix(arr[i])
        if isinstance(res, sympy.Expr):
            if i == len(arr) - 1:
                return res  # nothing to multiply by
            else:
                if i > 0:
                    left = Math.__convert_postfix(arr[i - 1])
                    right = Math.__convert_postfix(arr[i + 1])
                    if isinstance(left, sympy.Expr) and isinstance(right, sympy.Expr):
                        left_syms = Math.__convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                        right_syms = Math.__convert_postfix(arr[i + 1]).atoms(sympy.Symbol)
                        # if the left and right sides contain no variables and the
                        # symbol in between is 'x', treat as multiplication.
                        if len(left_syms) == 0 and len(right_syms) == 0 and str(res) == "x":
                            return Math.__convert_postfix_list(arr, i + 1)
                # multiply by next
                return sympy.Mul(res, Math.__convert_postfix_list(arr, i + 1), evaluate = False)
        else:  # must be derivative
            wrt = res[0]
            if i == len(arr) - 1:
                raise Exception("Expected expression for derivative")
            else:
                expr = Math.__convert_postfix_list(arr, i + 1)
                return sympy.Derivative(expr, wrt)

    @staticmethod
    def __do_subs(expr, at):
        if at.expr():
            at_expr = Math.__convert_expr(at.expr())
            syms = at_expr.atoms(sympy.Symbol)
            if len(syms) == 0:
                return expr
            elif len(syms) > 0:
                sym = next(iter(syms))
                return expr.subs(sym, at_expr)
        elif at.equality():
            lh = Math.__convert_expr(at.equality().expr(0))
            rh = Math.__convert_expr(at.equality().expr(1))
            return expr.subs(lh, rh)

    @staticmethod
    def __convert_postfix(postfix):
        if hasattr(postfix, 'exp'):
            exp_nested = postfix.exp()
        else:
            exp_nested = postfix.exp_nofunc()

        exp = Math.__convert_exp(exp_nested)
        for op in postfix.postfix_op():
            if op.BANG():
                if isinstance(exp, list):
                    raise Exception("Cannot apply postfix to derivative")
                exp = sympy.factorial(exp, evaluate = False)
            elif op.eval_at():
                ev = op.eval_at()
                at_b = None
                at_a = None
                if ev.eval_at_sup():
                    at_b = Math.__do_subs(exp, ev.eval_at_sup())
                if ev.eval_at_sub():
                    at_a = Math.__do_subs(exp, ev.eval_at_sub())
                if at_b != None and at_a != None:
                    exp = sympy.Add(at_b, -1 * at_a, evaluate = False)
                elif at_b != None:
                    exp = at_b
                elif at_a != None:
                    exp = at_a

        return exp

    @staticmethod
    def __convert_exp(exp):
        if hasattr(exp, 'exp'):
            exp_nested = exp.exp()
        else:
            exp_nested = exp.exp_nofunc()

        if exp_nested:
            base = Math.__convert_exp(exp_nested)
            if isinstance(base, list):
                raise Exception("Cannot raise derivative to power")
            if exp.atom():
                exponent = Math.__convert_atom(exp.atom())
            elif exp.expr():
                exponent = Math.__convert_expr(exp.expr())
            return sympy.Pow(base, exponent, evaluate = False)
        else:
            if hasattr(exp, 'comp'):
                return Math.__convert_comp(exp.comp())
            else:
                return Math.__convert_comp(exp.comp_nofunc())

    @staticmethod
    def __convert_comp(comp):
        if comp.group():
            return Math.__convert_expr(comp.group().expr())
        elif comp.abs_group():
            return sympy.Abs(Math.__convert_expr(comp.abs_group().expr()), evaluate = False)
        elif comp.atom():
            return Math.__convert_atom(comp.atom())
        elif comp.frac():
            return Math.__convert_frac(comp.frac())
        elif comp.func():
            return Math.__convert_func(comp.func())

    @staticmethod
    def __convert_atom(atom):
        if atom.LETTER():
            subscriptName = ''
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    subscript = Math.__convert_expr(atom.subexpr().expr())
                else:  # subscript is atom
                    subscript = Math.__convert_atom(atom.subexpr().atom())
                subscriptName = '_{' + StrPrinter().doprint(subscript) + '}'
            return sympy.Symbol(atom.LETTER().getText() + subscriptName)
        elif atom.SYMBOL():
            s = atom.SYMBOL().getText()[1:]
            if s == "infty":
                return sympy.oo
            else:
                if atom.subexpr():
                    subscript = None
                    if atom.subexpr().expr():  # subscript is expr
                        subscript = Math.__convert_expr(atom.subexpr().expr())
                    else:  # subscript is atom
                        subscript = Math.__convert_atom(atom.subexpr().atom())
                    subscriptName = StrPrinter().doprint(subscript)
                    s += '_{' + subscriptName + '}'
                return sympy.Symbol(s)
        elif atom.NUMBER():
            s = atom.NUMBER().getText().replace(",", "")
            return sympy.Number(s)
        elif atom.DIFFERENTIAL():
            var = Math.__get_differential_var(atom.DIFFERENTIAL())
            return sympy.Symbol('d' + var.name)
        elif atom.mathit():
            text = Math.__rule2text(atom.mathit().mathit_text())
            return sympy.Symbol(text)

    @staticmethod
    def __rule2text(ctx):
        stream = ctx.start.getInputStream()
        # starting index of starting token
        startIdx = ctx.start.start
        # stopping index of stopping token
        stopIdx = ctx.stop.stop

        return stream.getText(startIdx, stopIdx)

    @staticmethod
    def __convert_frac(frac):
        diff_op = False
        partial_op = False
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1
        if (frac.lower.start == frac.lower.stop and
                frac.lower.start.type == PSLexer.DIFFERENTIAL):
            wrt = Math.__get_differential_var_str(frac.lower.start.text)
            diff_op = True
        elif (lower_itv_len == 2 and
              frac.lower.start.type == PSLexer.SYMBOL and
              frac.lower.start.text == '\\partial' and
              (frac.lower.stop.type == PSLexer.LETTER or frac.lower.stop.type == PSLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == PSLexer.SYMBOL:
                wrt = wrt[1:]

        if diff_op or partial_op:
            wrt = sympy.Symbol(wrt)
            if (diff_op and frac.upper.start == frac.upper.stop and
                    frac.upper.start.type == PSLexer.LETTER and
                    frac.upper.start.text == 'd'):
                return [wrt]
            elif (partial_op and frac.upper.start == frac.upper.stop and
                  frac.upper.start.type == PSLexer.SYMBOL and
                  frac.upper.start.text == '\\partial'):
                return [wrt]
            upper_text = Math.__rule2text(frac.upper)

            expr_top = None
            if diff_op and upper_text.startswith('d'):
                expr_top = Math.__process_sympy(upper_text[1:])
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = Math.__process_sympy(upper_text[len('\\partial'):])
            if expr_top:
                return sympy.Derivative(expr_top, wrt)

        expr_top = Math.__convert_expr(frac.upper)
        expr_bot = Math.__convert_expr(frac.lower)
        return sympy.Mul(expr_top, sympy.Pow(expr_bot, -1, evaluate = False), evaluate = False)

    @staticmethod
    def __convert_func(func):
        if func.func_normal():
            if func.L_PAREN():  # function called with parenthesis
                arg = Math.__convert_func_arg(func.func_arg())
            else:
                arg = Math.__convert_func_arg(func.func_arg_noparens())

            name = func.func_normal().start.text[1:]

            # change arc<trig> -> a<trig>
            if name in ["arcsin", "arccos", "arctan", "arccsc", "arcsec",
                        "arccot"]:
                name = "a" + name[3:]
                expr = getattr(sympy.functions, name)(arg, evaluate = False)
            if name in ["arsinh", "arcosh", "artanh"]:
                name = "a" + name[2:]
                expr = getattr(sympy.functions, name)(arg, evaluate = False)

            if (name == "log" or name == "ln"):
                if func.subexpr():
                    base = Math.__convert_expr(func.subexpr().expr())
                elif name == "log":
                    base = 10
                elif name == "ln":
                    base = sympy.E
                expr = sympy.log(arg, base, evaluate = False)

            func_pow = None
            should_pow = True
            if func.supexpr():
                if func.supexpr().expr():
                    func_pow = Math.__convert_expr(func.supexpr().expr())
                else:
                    func_pow = Math.__convert_atom(func.supexpr().atom())

            if name in ["sin", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh", "tanh"]:
                if func_pow == -1:
                    name = "a" + name
                    should_pow = False
                expr = getattr(sympy.functions, name)(arg, evaluate = False)

            if func_pow and should_pow:
                expr = sympy.Pow(expr, func_pow, evaluate = False)

            return expr
        elif func.LETTER() or func.SYMBOL():
            if func.LETTER():
                fname = func.LETTER().getText()
            elif func.SYMBOL():
                fname = func.SYMBOL().getText()[1:]
            fname = str(fname)  # can't be unicode
            if func.subexpr():
                subscript = None
                if func.subexpr().expr():  # subscript is expr
                    subscript = Math.__convert_expr(func.subexpr().expr())
                else:  # subscript is atom
                    subscript = Math.__convert_atom(func.subexpr().atom())
                subscriptName = StrPrinter().doprint(subscript)
                fname += '_{' + subscriptName + '}'
            input_args = func.args()
            output_args = []
            while input_args.args():  # handle multiple arguments to function
                output_args.append(Math.__convert_expr(input_args.expr()))
                input_args = input_args.args()
            output_args.append(Math.__convert_expr(input_args.expr()))
            return sympy.Function(fname)(*output_args)
        elif func.FUNC_INT():
            return Math.__handle_integral(func)
        elif func.FUNC_SQRT():
            expr = Math.__convert_expr(func.base)
            if func.root:
                r = Math.__convert_expr(func.root)
                return sympy.root(expr, r)
            else:
                return sympy.sqrt(expr)
        elif func.FUNC_SUM():
            return Math.__handle_sum_or_prod(func, "summation")
        elif func.FUNC_PROD():
            return Math.__handle_sum_or_prod(func, "product")
        elif func.FUNC_LIM():
            return Math.__handle_limit(func)

    @staticmethod
    def __convert_func_arg(arg):
        if hasattr(arg, 'expr'):
            return Math.__convert_expr(arg.expr())
        else:
            return Math.__convert_mp(arg.mp_nofunc())

    @staticmethod
    def __handle_integral(func):
        if func.additive():
            integrand = Math.__convert_add(func.additive())
        elif func.frac():
            integrand = Math.__convert_frac(func.frac())
        else:
            integrand = 1

        int_var = None
        if func.DIFFERENTIAL():
            int_var = Math.__get_differential_var(func.DIFFERENTIAL())
        else:
            for sym in integrand.atoms(sympy.Symbol):
                s = str(sym)
                if len(s) > 1 and s[0] == 'd':
                    if s[1] == '\\':
                        int_var = sympy.Symbol(s[2:])
                    else:
                        int_var = sympy.Symbol(s[1:])
                    int_sym = sym
            if int_var:
                integrand = integrand.subs(int_sym, 1)
            else:
                # Assume dx by default
                int_var = sympy.Symbol('x')

        if func.subexpr():
            if func.subexpr().atom():
                lower = Math.__convert_atom(func.subexpr().atom())
            else:
                lower = Math.__convert_expr(func.subexpr().expr())
            if func.supexpr().atom():
                upper = Math.__convert_atom(func.supexpr().atom())
            else:
                upper = Math.__convert_expr(func.supexpr().expr())
            return sympy.Integral(integrand, (int_var, lower, upper))
        else:
            return sympy.Integral(integrand, int_var)

    @staticmethod
    def __handle_sum_or_prod(func, name):
        val = Math.__convert_mp(func.mp())
        iter_var = Math.__convert_expr(func.subeq().equality().expr(0))
        start = Math.__convert_expr(func.subeq().equality().expr(1))
        if func.supexpr().expr():  # ^{expr}
            end = Math.__convert_expr(func.supexpr().expr())
        else:  # ^atom
            end = Math.__convert_atom(func.supexpr().atom())

        if name == "summation":
            return sympy.Sum(val, (iter_var, start, end))
        elif name == "product":
            return sympy.Product(val, (iter_var, start, end))

    @staticmethod
    def __handle_limit(func):
        sub = func.limit_sub()
        if sub.LETTER():
            var = sympy.Symbol(sub.LETTER().getText())
        elif sub.SYMBOL():
            var = sympy.Symbol(sub.SYMBOL().getText()[1:])
        else:
            var = sympy.Symbol('x')
        if sub.SUB():
            direction = "-"
        else:
            direction = "+"
        approaching = Math.__convert_expr(sub.expr())
        content = Math.__convert_mp(func.mp())

        return sympy.Limit(content, var, approaching, direction)

    @staticmethod
    def __get_differential_var(d):
        text = Math.__get_differential_var_str(d.getText())
        return sympy.Symbol(text)

    @staticmethod
    def __get_differential_var_str(text):
        for i in range(1, len(text)):
            c = text[i]
            if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
                idx = i
                break
        text = text[idx:]
        if text[0] == "\\":
            text = text[1:]
        return text
