#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.context import ParallelContext

global_context = ParallelContext.get_instance()
