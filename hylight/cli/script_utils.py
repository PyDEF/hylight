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
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps


def _default_post(ret):
    if isinstance(ret, int):
        return ret
    else:
        return 0


class MultiCmd:
    """A CLI with several subcommands.

    Build an entrypoint with several subcommands.
    Register subcommands with MultiCmd.subcmd.
    Call the instance (or its run method) to run the script.
    """

    def __init__(self, **kwargs):
        self.main_parser = ArgumentParser(**kwargs)
        self.sub_parsers = self.main_parser.add_subparsers()

        self._pre = None
        self._post = _default_post

    def add(self, option):
        "Add a global parameter, common to all subcommands."
        option(self.main_parser)

    def pre(self, f):
        "Register a function to be called on the namespace before the command."
        assert callable(f), "pre applies to functions"
        self._pre = f
        return f

    def post(self, f):
        "Register a function to be called on the return value of the command at the end."
        assert callable(f), "post applies to functions"
        self._post = f
        return f

    def __call__(self, *args):
        return self.run(*args)

    def subcmd(self, *args):
        """Register a new subcommand.

        See script.
        """
        def decorator(f):
            parser = self.sub_parsers.add_parser(f.__name__, help=f.__doc__)

            for param in args:
                param(parser)

            parser.set_defaults(handler=f)

            return f

        return decorator

    def run(self, argv=None):
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

        if self._post:
            return self._post(opts.handler(opts))
        else:
            return opts.handler(opts)


def script(*args, name=None, doc=None):
    """Wrap a function for a script.

    Each argument of specify an element of the CLI wrapping the function (see
    positional, optional, flag, rest). The function will be called with a
    argparse.Namespace instance containing the parser parameters.
    """

    def decorator(f):
        parser = ArgumentParser(prog=name or f.__name__,
                                description=doc or f.__doc__)

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


def error(msg):
    "Write msg to stderr and exit with -1 return code."
    print(msg, file=sys.stderr)
    exit(-1)


@contextmanager
def error_catch():
    """Catch all errors, print them and exit with -1 return code.

    Context manager to provide cleaner errors when building a script around a
    function that can raise.
    """
    try:
        yield
    except Exception as e:
        error(str(e))


def flag(*names, **kwargs):
    """Add a flag dashed parameter that always.

    You can provide as many aliases as you want but they must all start with a
    dash.
    The first name provided is the name of Namespace attribute (dash removed).
    The value is always False by default and True if the user provided the flag.
    """
    def add(parser):
        parser.add_argument(
            *names,
            action="store_true",
            **kwargs,
        )

    return add


_not_a_value = object()


def positional(name, default=_not_a_value, **kwargs):
    """Add a positional parameter.

    The name is used as the metavar and must not start with a dash.
    If the `default` kwarg is provided it is an optional parameter.
    kwargs are passed to ArgumentParser.add_argument
    """
    assert not name.startswith("-"), "Please use flag or optional for dash parameters."

    if default is not _not_a_value:
        kwargs["nargs"] = "?"
        kwargs["default"] = default

    def add(parser):
        parser.add_argument(
            name.lower(),
            metavar=name,
            **kwargs,
        )

    return add


def rest(name, **kwargs):
    """Add a catch-all parameter for the end of the parameter list.

    The name is used as the metavar and must not start with a dash.
    kwargs are passed to ArgumentParser.add_argument
    """
    assert not name.startswith("-"), "Please use flag or optional for dash parameters."

    def add(parser):
        parser.add_argument(
            name.lower(),
            metavar=name,
            nargs="+",
            **kwargs,
        )

    return add


def optional(*names, **kwargs):
    """Add an optional parameter that uses a dashed trigger.

    You can provide as many aliases as you want, but all of them must at least
    start with a dash.
    kwargs are passed to ArgumentParser.add_argument.
    """
    assert all(
        n.startswith("-") for n in names
    ), "Either use a single non-dash name or only dash names."

    def add(parser):
        parser.add_argument(
            *names,
            **kwargs,
        )

    return add
