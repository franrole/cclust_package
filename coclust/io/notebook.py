# -*- coding: utf-8 -*-

"""
The :mod:`coclust.io.notebook` module provides functions to manage input and
output in the evaluation notebook.
"""


def input_with_default_int(prompt, prefill):
    """Prompt an int.

    Parameters
    ----------
    prompt: string
        The value entered by the user

    prefill: string
        The default value

    Returns
    -------
    int
        The value entered by the user or the default value.
    """

    retint = input('%s: [default: %s] ' % (prompt, prefill))

    if len(retint) > 0:
        retint = int(retint)
    else:
        retint = prefill

    return retint


def input_with_default_str(prompt, prefill):
    """Prompt a string.

    Parameters
    ----------
    prompt: string
        The value entered by the user

    prefill: string
        The default value

    Returns
    -------
    string
        The value entered by the user or the default value.
    """

    retstr = input('%s: [default: %s] ' % (prompt, prefill))

    if len(retstr) == 0:
        retstr = prefill

    return retstr
