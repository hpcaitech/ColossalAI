#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.context import ParallelContext, MoeContext

global_context = ParallelContext.get_instance()
MOE_CONTEXT = MoeContext.get_instance()
