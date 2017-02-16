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
        The message printed before the field.

    prefill: int
        The default value.

    Returns
    -------
    int
        The value entered by the user or the default value.
    """
    try:
        # Python 2
        value = raw_input('%s: [default: %s] ' % (prompt, prefill))
    except NameError:
        # Python 3
        value = input('%s: [default: %s] ' % (prompt, prefill))

    # value is a string
    if len(value) == 0:
        return prefill
    else:
        return int(value)


def input_with_default_str(prompt, prefill):
    """Prompt a string.

    Parameters
    ----------
    prompt: string
        The message printed before the field.

    prefill: string
        The default value.

    Returns
    -------
    string
        The value entered by the user or the default value.
    """
    try:
        # Python 2
        value = raw_input('%s: [default: %s] ' % (prompt, prefill))
    except NameError:
        # Python 3
        value = input('%s: [default: %s] ' % (prompt, prefill))

    # value is a string
    if len(value) == 0:
        value = prefill

    return value
