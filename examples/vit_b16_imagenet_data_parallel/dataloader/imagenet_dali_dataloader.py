from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import torch
import numpy as np
from .rand_augment import RandAugment


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
                 mixup_alpha=0.0,
                 randaug_magnitude=10,
                 randaug_num_layers=0):
        self.mixup_alpha = mixup_alpha
        self.training = training
        self.randaug_magnitude = randaug_magnitude
        self.randaug_num_layers = randaug_num_layers
        pipe = Pipeline(batch_size=batch_size,
                        num_threads=num_threads,
                        device_id=torch.cuda.current_device() if cuda else None,
                        seed=42)
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
            images = fn.decoders.image(images,
                                       device='mixed' if gpu_aug else 'cpu',
                                       output_type=types.RGB)
            if training:
                images = fn.random_resized_crop(images,
                                                size=crop,
                                                device='gpu' if gpu_aug else 'cpu')
                if randaug_num_layers == 0:
                    flip_lr = fn.random.coin_flip(probability=0.5)
                    images = fn.flip(images, horizontal=flip_lr)
            else:
                images = fn.resize(images,
                                   device='gpu' if gpu_aug else 'cpu',
                                   resize_x=resize,
                                   resize_y=resize,
                                   dtype=types.FLOAT,
                                   interp_type=types.INTERP_TRIANGULAR)
                images = fn.crop(images,
                                 dtype=types.FLOAT,
                                 crop=(crop, crop))
            label = inputs["image/class/label"] - 1  # 0-999
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
        img = img.permute(0, 3, 1, 2)
        if self.randaug_num_layers > 0 and self.training:
            img = RandAugment(img, num_layers=self.randaug_num_layers, magnitude=self.randaug_magnitude)
        img = (img - 127.5) / 127.5
        label = label.squeeze()
        if self.mixup_alpha > 0.0:
            if self.training:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                idx = torch.randperm(img.size(0)).to(img.device)
                img = lam * img + (1 - lam) * img[idx, :]
                label_a, label_b = label, label[idx]
                lam = torch.tensor([lam], device=img.device, dtype=img.dtype)
                label = {'targets_a': label_a, 'targets_b': label_b, 'lam': lam}
            else:
                label = {'targets_a': label, 'targets_b': label, 'lam': torch.ones(
                    1, device=img.device, dtype=img.dtype)}
            return img, label
        return img, label
