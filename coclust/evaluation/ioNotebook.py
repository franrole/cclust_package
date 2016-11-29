# -*- coding: utf-8 -*-

"""
Managing input/output in the evaluation notebook
"""

def input_with_default_int(prompt, prefill):
    retint = input('%s: [default: %s] ' % (prompt, prefill))

    if len(retint)>0:
        retint = int(retint)
    else:
        retint = prefill

    return retint

def input_with_default_str(prompt, prefill):
    retstr = input('%s: [default: %s] ' % (prompt, prefill))

    if len(retstr) == 0:
        retstr = prefill

    return retstr
