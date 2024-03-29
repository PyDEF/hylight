"""A useful set of utils to create Python CLI tools.

This module is designed to be embeded into your own package. It is built around
wrappers of the builtin python module argparse and helps building entrypoints
compatible with setuptools, as well as package entrypoint. The idea is to write
the bulk of the script in a function that is decorated with the specification
of its CLI interface. The function simply takes the Namespace object with the
parameters already parsed.

Here is a usage example of the module:
.. code-block:: python

    # my_package/cli.py
    from .script_utils import MultiCmd, script, positional, flag

    # Single command entry point:
    @script(
        positional("FOO", help="the first parameter."),
        flag("--bar", "-b", help="enable barring the foo."),
    )
    def treat_foo(opts):
        "Treat the foo with determination."
        # by default the docstring is used as the command description.
        #
        foo = opts.foo
        if opts.bar:
            # do the barring

        return 0

    # Multiple subcommands in the same entrypoint

    treat_baz = MultiCmd(prog="treat_baz", description="Multiple treatement of the baz.")

    @treat_baz.subcmd(
        positional("BAZ", help="The baz to fizz."),
        flag("--care", "-c", "Fizz the baz with extra care."),
    )
    def fizz(opts):
        if opts.care:
            # extra care
        else:
            # quick fizzing

    @treat_baz.subcmd(
        positional("BAZ", help="The baz to buzz."),
        positional("QUX", default=basic_qux, type=qux_from_string,
                   help="The qux used in buzzing the baz"),
    )
    def buzz(opts):
        # opts.qux is either basic_qux or the result of calling qux_from_string
        # on the user parameter.


You can then call `treat_foo` and `treat_baz` with no argument or with
`sys.argv` in you `__name__ == "__main__"` section. Alternatively, they are
both valid callable entrypoint to set in your setup.py/setup.cfg.

Here is a short example of a corresponding setup.cfg:

.. code-block:: cfg

    [metadata]
    name = my_package
    author = Your name
    author_email = your.address@email.net
    description = A short description

    [options]
    packages = find:

    [options.entry_points]
    console_scripts =
        treat_foo = mypackage.cli:treat_foo
        treat_baz = mypackage.cli:treat_baz

License:
    Copyright (c) 2022 Théo Cavignac, All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""
import sys
import os
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from functools import wraps
from statistics import mean, stdev, variance
from time import perf_counter

from typing import Callable, List, TypeVar, Optional, Generic

_debug = os.environ.get("DEBUG", False) in {"1", "yes", "true"}


T = TypeVar("T")
ArgSpec = Callable[[ArgumentParser], None]

ScriptWrapper = Callable[[Callable[[Namespace], T]], Callable[[List[str]], T]]
PostFn = Callable[[T], int]


def _default_post(ret: T) -> int:
    if isinstance(ret, int):
        return ret
    else:
        return 0


class MultiCmd(Generic[T]):
    """A CLI with several subcommands.

    Build an entrypoint with several subcommands.
    Register subcommands with MultiCmd.subcmd.
    Call the instance (or its run method) to run the script.
    """

    def __init__(self, **kwargs) -> None:
        self.main_parser = ArgumentParser(**kwargs)
        self.sub_parsers = self.main_parser.add_subparsers()

        self._pre: Optional[Callable[[Namespace], None]] = None
        self._post: PostFn[T] = _default_post

    def add(self, option: ArgSpec):
        "Add a global parameter, common to all subcommands."
        option(self.main_parser)

    def pre(self, f: Callable[[Namespace], None]):
        "Register a function to be called on the namespace before the command."
        assert callable(f), "pre applies to functions"
        self._pre = f
        return f

    def post(self, f: Callable[[T], int]) -> Callable[[T], int]:
        "Register a function to be called on the return value of the command at the end."
        assert callable(f), "post applies to functions"
        self._post = f
        return f

    def __call__(self, argv: Optional[List[str]] = None) -> int:
        "See :meth:`run`."
        return self.run(argv=argv)

    def subcmd(self, *args: ArgSpec, name: Optional[str] = None) -> ScriptWrapper:
        """Register a new subcommand.

        See script.
        """

        def decorator(f):
            help_msg = f.__doc__ and f.__doc__.replace("%", "%%")
            parser = self.sub_parsers.add_parser(name or f.__name__, help=help_msg)

            for param in args:
                param(parser)

            parser.set_defaults(handler=f)

            @wraps(f)
            def wrapper(argv: Optional[List[str]] = None):
                if argv is None:
                    opts = parser.parse_args()
                else:
                    opts = parser.parse_args(argv)

                return f(opts)

            return wrapper

        return decorator

    def run(self, argv: Optional[List[str]] = None) -> int:
        """Execute the script (calling the instance redirect here).

        You can explicitly pass a list of string parameters instead of using
        sys.argv.
        """

        if argv is None:
            opts = self.main_parser.parse_args()
        else:
            opts = self.main_parser.parse_args(argv)

        if self._pre:
            self._pre(opts)

        if not hasattr(opts, "handler"):
            self.main_parser.print_help()
            exit(1)

        return self._post(opts.handler(opts))


def script(
    *args: ArgSpec, name: Optional[str] = None, doc: Optional[str] = None
) -> ScriptWrapper:
    """Wrap a function for a script.

    Each argument of specify an element of the CLI wrapping the function (see
    positional, optional, flag, rest). The function will be called with a
    argparse.Namespace instance containing the parser parameters.
    """

    def decorator(f):
        if doc is None:
            doc_ = f.__doc__ and f.__doc__.replace("%", "%%")
        else:  # doc == "" is valid and cause the description to be empty
            doc_ = doc.replace("%", "%%")
        parser = ArgumentParser(prog=name or f.__name__, description=doc_)

        for param in args:
            param(parser)

        @wraps(f)
        def wrapper(argv=None):
            if argv is None:
                opts = parser.parse_args()
            else:
                opts = parser.parse_args(argv)

            return f(opts)

        return wrapper

    return decorator


def error(msg: str):
    "Write msg to stderr and exit with -1 return code."
    print(msg, file=sys.stderr)
    exit(-1)


@contextmanager
def error_catch(prefix="Error: "):
    """Catch all errors, print them and exit with -1 return code.

    Context manager to provide cleaner errors when building a script around a
    function that can raise.
    If the environment variable DEBUG is set to "yes", this does nothing in
    order to let the exception crash the program and show the traceback.
    """
    if _debug:
        yield
    else:
        try:
            yield
        except Exception as e:
            error(prefix + str(e))


def flag(*names: str, **kwargs) -> ArgSpec:
    """Add a flag dashed parameter that always.

    You can provide as many aliases as you want but they must all start with a
    dash.
    The first name provided is the name of Namespace attribute (dash removed).
    The value is always False by default and True if the user provided the flag.
    """

    if "help" in kwargs:
        kwargs["help"] = kwargs["help"].replace("%", "%%")

    def add(parser):
        parser.add_argument(
            *names,
            action="store_true",
            **kwargs,
        )

    return add


_not_a_value = object()


def positional(name: str, default=_not_a_value, **kwargs) -> ArgSpec:
    """Add a positional parameter.

    The name is used as the metavar and must not start with a dash.
    If the `default` kwarg is provided it is an optional parameter.
    kwargs are passed to ArgumentParser.add_argument
    """
    assert not name.startswith("-"), "Please use flag or optional for dash parameters."

    if default is not _not_a_value:
        kwargs["nargs"] = "?"
        kwargs["default"] = default

    if "help" in kwargs:
        kwargs["help"] = kwargs["help"].replace("%", "%%")

    def add(parser):
        parser.add_argument(
            name.lower(),
            metavar=name,
            **kwargs,
        )

    return add


def rest(name: str, **kwargs) -> ArgSpec:
    """Add a catch-all parameter for the end of the parameter list.

    If no default value is provided, it expect at least one parameter.
    If a default value is provided, it will produce the default value
    if no parameter match.
    The name is used as the metavar and must not start with a dash.
    kwargs are passed to ArgumentParser.add_argument
    """
    assert not name.startswith("-"), "Please use flag or optional for dash parameters."

    if "help" in kwargs:
        kwargs["help"] = kwargs["help"].replace("%", "%%")

    def add(parser):
        parser.add_argument(
            name.lower(),
            metavar=name,
            nargs=("*" if "default" in kwargs else "+"),
            **kwargs,
        )

    return add


def optional(*names: str, **kwargs) -> ArgSpec:
    """Add an optional parameter that uses a dashed trigger.

    You can provide as many aliases as you want, but all of them must at least
    start with a dash.
    kwargs are passed to ArgumentParser.add_argument.
    """
    assert all(
        n.startswith("-") for n in names
    ), "Either use a single non-dash name or only dash names."

    assert (
        kwargs.get("type", str) == str or "default" in kwargs
    ), "You should provide a default value."

    if "help" in kwargs:
        kwargs["help"] = kwargs["help"].replace("%", "%%")

    def add(parser):
        parser.add_argument(
            *names,
            **kwargs,
        )

    return add


class PerfCounterCollec:
    """A collection of perfomance counter.

    A simple facility to time sections of a program.

    Example:

        pc = PerfCounterCollec()

        with pc.foo:
            # do some stuff

        with pc.bar:
            # do other stuff

        print(pc.summary())

        # foo: 0.65344 s
        # bar: 1.32645 s
    """

    def __init__(self, format="{name}: {c.total:.05f} s"):
        self.counters = {}
        self.fmt = format

    def __getattr__(self, name):
        return self.counters.setdefault(name, PerfCounter())

    def collect(self):
        for name, counter in self.counters.items():
            yield (name, counter)

    def summary(self, format=None):
        lines = []

        if format is None:
            fmt = self.fmt
        else:
            fmt = format

        for name, counter in self.counters.items():
            lines.append(fmt.format(name=name, c=counter))

        return "\n".join(lines)


class PerfCounter:
    """An individual counter.

    Use it as a context manager.
    Each time it is entered, it start a new counter.
    Each time it is exited, it add the elapsed time of the last counter to the total.
    You can also access basic statistics (number of slices, mean and standard deviation).
    """

    def __init__(self):
        self.slices = []
        self.opened = []

    def __enter__(self):
        self.opened.append(perf_counter())

    def __exit__(self, *args):
        self.slices.append(perf_counter() - self.opened.pop())

    @property
    def total(self):
        "Total time spent under this counter."
        return sum(self.slices)

    @property
    def slice_number(self):
        "Number of separate slices added to this counter."
        return len(self.slices)

    @property
    def mean(self):
        "Mean time spent in a single slice."
        return mean(self.slices)

    @property
    def stdev(self):
        "Standard deviation of time spent in a slice."
        return stdev(self.slices)

    @property
    def variance(self):
        "Variance of time spent in a slice."
        return variance(self.slices)
