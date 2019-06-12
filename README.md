# latexpr
```
python3 setup.py antlr build
```


### About
Heavily influenced by:
`https://github.com/jackatbancast/latex2sympy`

This is a python library to write latex functions in your programs.
With this library, you can keep your documentation and implementation 
consistent because your implementation can read directly from the documentation.

Eventually I would like to allow for automatic distributed computation through a backend
such as spark or dask (as extra flags).  This would allow for larger scale computations
to be performed automatically if available.  I would also like to take the python code that is
output from the function and turn it into efficient LLVM code (if possible) for faster execution
in a variety of environments.

### Installation
```
pip3 install git+git://github.com/mattpaletta/optional-grpc.git
```

### Usage:
There are only 2 very 'simple' API functions.
```python
from latexpr import Math
Math.parse(latex: str)
Math.parse_file(file: str, tag: str)
Math.parse_mult_file(file: str, tag: List[str])
```

`Parse` and `parse_file` return a function that accept the variables from your 
latex equations as `**kwargs`.

`parse_mult_file` allows you to extract multiple functions from a file at once.  This avoids
the time taken to parse a markdown file.  This function yields a pair with the function name, 
and the python function as second value.  This python function behaves the same as `parse` or `parse_file`.

Note, when parsing a file, the equation declarations must follow the following format specification:
```
```{eq:<func_name>} // Begin your function
// put latex here
``` // End your function
```
Your function name must match: `{eq:([a-zA-Z0-9_-]+)}`


### Example
You can find runnable examples in the `examples/` folder for both `parse_file` and `parse_mult_file`.
These also include a sample markdown file for reference.  More latex examples can be found
in the tests folder.

```python
from latexpr import Math
f = Math.parse(latex = "2x + 3")

# We can now call f as a regular python function
value1: int = f(x = 1)
value2: int = f(x = 2)

assert value1 == (2 * 1) + 3
assert value1 == (2 * 2) + 3
```

Unfortunately, because I do not make guarantees about the order of the variables, you must use
kwargs to address variables.

Functions that do not take any variables are also valid.
```python
import math
from latexpr import Math

f = Math.parse(latex = "(2 * 3) + \\sqrt(9)")
value = f()
assert value == (2 * 3) + math.sqrt(9)
```

The `parse_file` function works very similarly.
```python
from latexpr import Math
func1 = Math.parse_file(file = "examples/README.md", tag = "func1")
func1_result = func1()
```

### Questions, Comments, Concerns, Queries, Qwibbles?

If you have any questions, comments, or concerns please leave them in the GitHub
Issues tracker.

### Bug reports

If you discover any bugs, feel free to create an issue on GitHub. Please add as much information as
possible to help us fixing the possible bug. We also encourage you to help even more by forking and
sending us a pull request.

## Maintainers

* Matthew Paletta (https://github.com/mattpaletta)

## License

MIT License. Copyright 2019 Matthew Paletta. http://mrated.ca