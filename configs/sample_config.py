#!/usr/bin/env python
# -*- encoding: utf-8 -*-

NUM_EPOCH = int

model = dict()
train_data = dict()
test_data = dict()
optimizer = dict()
loss = dict()

fp16 = dict()
zero = dict()

gradient_handler = []
parallel = dict()
hooks = []

cudnn_benchmark = True
cudnn_deterministic = False

logging = dict()
