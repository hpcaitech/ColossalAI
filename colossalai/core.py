#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.context import ParallelContext

global_context = ParallelContext()


def set_global_context(context: ParallelContext):
    '''Reset global context to be identical to a given :class:ParallelContext.

    :param context: Parallel context to generate our global parallel context.
    :type context: ParallelContext
    '''
    global global_context
    global_context = context
