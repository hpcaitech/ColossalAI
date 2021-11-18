from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import torch
import numpy as np


class DaliDataloader(DALIClassificationIterator):
    def __init__(self,
                 tfrec_filenames,
                 tfrec_idx_filenames,
                 shard_id=0,
                 num_shards=1,
                 batch_size=128,
                 num_threads=4,
                 resize=256,
                 crop=224,
                 prefetch=2,
                 training=True,
                 gpu_aug=False,
                 cuda=True,
                 mixup_alpha=0.0):
        self.mixup_alpha = mixup_alpha
        self.training = training
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
                    'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                    'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                })
            images = inputs["image/encoded"]

            if training:
                images = fn.decoders.image(images,
                                           device='mixed' if gpu_aug else 'cpu',
                                           output_type=types.RGB)
                images = fn.random_resized_crop(images,
                                                size=crop,
                                                device='gpu' if gpu_aug else 'cpu')
                flip_lr = fn.random.coin_flip(probability=0.5)
            else:
                # decode jpeg and resize
                images = fn.decoders.image(images,
                                           device='mixed' if gpu_aug else 'cpu',
                                           output_type=types.RGB)
                images = fn.resize(images,
                                   device='gpu' if gpu_aug else 'cpu',
                                   resize_x=resize,
                                   resize_y=resize,
                                   dtype=types.FLOAT,
                                   interp_type=types.INTERP_TRIANGULAR)
                flip_lr = False

            # center crop and normalise
            images = fn.crop_mirror_normalize(images,
                                              dtype=types.FLOAT,
                                              crop=(crop, crop),
                                              mean=[127.5],
                                              std=[127.5],
                                              mirror=flip_lr)
            label = inputs["image/class/label"] - 1  # 0-999
            # LSG: element_extract will raise exception, let's flatten outside
            # label = fn.element_extract(label, element_map=0)  # Flatten
            if cuda:  # transfer data to gpu
                pipe.set_outputs(images.gpu(), label.gpu())
            else:
                pipe.set_outputs(images, label)

        pipe.build()
        last_batch_policy = 'DROP' if training else 'PARTIAL'
        super().__init__(pipe, reader_name="Reader",
                         auto_reset=True,
                         last_batch_policy=last_batch_policy)

    def __iter__(self):
        # if not reset (after an epoch), reset; if just initialize, ignore
        if self._counter >= self._size or self._size < 0:
            self.reset()
        return self

    def __next__(self):
        data = super().__next__()
        img, label = data[0]['data'], data[0]['label']
        label = label.squeeze()
        if self.mixup_alpha > 0.0:
            if self.training:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                idx = torch.randperm(img.size(0)).to(img.device)
                img = lam * img + (1 - lam) * img[idx, :]
                label_a, label_b = label, label[idx]
                lam = torch.tensor([lam], device=img.device, dtype=img.dtype)
                label = (label_a, label_b, lam)
            else:
                label = (label, label, torch.ones(
                    1, device=img.device, dtype=img.dtype))
            return (img,), label
        return (img,), (label,)
