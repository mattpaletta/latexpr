import latexpr


def single():
    # Note the variable names here are arbitrary
    # The only thing that matters is the tag names must match
    # and the kwargs to the functions must match the variable names declared in your Latex.

    print("Finding single")

    func1 = latexpr.Math.parse_file(file = "README.md", tag = "func1")
    func1_result = func1()
    print("Func1 result: {0}".format(func1_result))

    foo = latexpr.Math.parse_file(file = "README.md", tag = "foo")
    foo_result = foo(a = 1, b = 2, c = 3)
    print("foo result: {0}".format(foo_result))

    super_important_eq = latexpr.Math.parse_file(file = "README.md", tag = "super_important_eq")
    super_important_eq_result = super_important_eq(a = 1, b = 2, c = 3)
    print("super_important_eq result: {0}".format(super_important_eq_result))


def batch():
    funcs = latexpr.Math.parse_mult_file(file = "README.md", tag = ["foo", "super_important_eq"])
    print("Finding in batch")
    for t, f in funcs:
        print(t, f(a = 1, b = 2, c = 3))


def main():
    single()
    batch()


if __name__ == "__main__":
    main()
