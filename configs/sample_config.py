#!/usr/bin/env python
# -*- encoding: utf-8 -*-

model = dict()
train_data = dict()
test_data = dict()
optimizer = dict()
loss = dict()
lr_scheduler = dict()

fp16 = dict()
zero = dict()

gradient_handler = []
parallel = dict()

num_epochs = int
num_steps = int

cudnn_benchmark = True
cudnn_deterministic = False

logging = dict()
