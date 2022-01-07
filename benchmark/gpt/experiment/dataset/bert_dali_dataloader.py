#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import torch
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
import glob
import os


class DaliDataloader(DALIGenericIterator):
    def __init__(self,
                 tfrec_filenames,
                 tfrec_idx_filenames,
                 output_map=["input_ids", "input_mask", "segment_ids", "masked_lm_positions",
                             "masked_lm_ids", "masked_lm_weights", "next_sentence_labels"],
                 shard_id=0,
                 num_shards=1,
                 batch_size=32,
                 num_threads=4,
                 prefetch=2,
                 max_seq_length=512,
                 max_predictions_per_seq=76,
                 training=True,
                 cuda=True):
        pipe = Pipeline(batch_size=batch_size,
                        num_threads=num_threads,
                        device_id=torch.cuda.current_device() if cuda else None,
                        seed=1024)
        with pipe:
            inputs = fn.readers.tfrecord(
                path=tfrec_filenames,
                index_path=tfrec_idx_filenames,
                random_shuffle=training,
                shard_id=shard_id,
                num_shards=num_shards,
                initial_fill=10000,
                read_ahead=True,
                prefetch_queue_depth=prefetch,
                name='Reader',
                features={
                    "input_ids":
                        tfrec.FixedLenFeature([max_seq_length], tfrec.int64, 0),
                    "input_mask":
                        tfrec.FixedLenFeature([max_seq_length], tfrec.int64, 0),
                    "segment_ids":
                        tfrec.FixedLenFeature([max_seq_length], tfrec.int64, 0),
                    "masked_lm_positions":
                        tfrec.FixedLenFeature([max_predictions_per_seq], tfrec.int64, 0),
                    "masked_lm_ids":
                        tfrec.FixedLenFeature([max_predictions_per_seq], tfrec.int64, 0),
                    "masked_lm_weights":
                        tfrec.FixedLenFeature([max_predictions_per_seq], tfrec.float32, 0.0),
                    "next_sentence_labels":
                        tfrec.FixedLenFeature([1], tfrec.int64, 0)}
            )
            input_ids = inputs["input_ids"]
            input_mask = inputs["input_mask"]
            segment_ids = inputs["segment_ids"]
            masked_lm_positions = inputs["masked_lm_positions"]
            masked_lm_ids = inputs["masked_lm_ids"]
            masked_lm_weights = inputs["masked_lm_weights"]
            next_sentence_labels = inputs["next_sentence_labels"]

            if cuda:  # transfer data to gpu
                pipe.set_outputs(input_ids.gpu(),
                                 input_mask.gpu(),
                                 segment_ids.gpu(),
                                 masked_lm_positions.gpu(),
                                 masked_lm_ids.gpu(),
                                 masked_lm_weights.gpu(),
                                 next_sentence_labels.gpu())
            else:
                pipe.set_outputs(input_ids,
                                 input_mask,
                                 segment_ids,
                                 masked_lm_positions,
                                 masked_lm_ids,
                                 masked_lm_weights,
                                 next_sentence_labels)

        pipe.build()
        last_batch_policy = 'DROP' if training else 'PARTIAL'
        super().__init__(pipe, reader_name="Reader", output_map=output_map,
                         auto_reset=True,
                         last_batch_policy=last_batch_policy)

    def __iter__(self):
        # if not reset (after an epoch), reset; if just initialize, ignore
        if self._counter >= self._size or self._size < 0:
            self.reset()
        return self

    def __next__(self):
        data = super().__next__()

        # input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels
        input_ids = data[0]["input_ids"]
        input_mask = data[0]["input_mask"]
        segment_ids = data[0]["segment_ids"]
        masked_lm_positions = data[0]["masked_lm_positions"]
        masked_lm_ids = data[0]["masked_lm_ids"]
        masked_lm_weights = data[0]["masked_lm_weights"]
        next_sentence_labels = data[0]["next_sentence_labels"]

        return {'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids}, {'masked_lm_positions': masked_lm_positions, 'masked_lm_ids': masked_lm_ids,
                                                                                                       'masked_lm_weights': masked_lm_weights, 'next_sentence_labels': next_sentence_labels}

