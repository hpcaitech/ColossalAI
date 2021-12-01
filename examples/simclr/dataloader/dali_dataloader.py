from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy, DALIGenericIterator
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
from tensorboard.plugins.image.summary_v2 import image
import torch
import numpy as np
from nvidia.dali import pipeline_def

class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, tfrec_filenames, tfrec_idx_filenames, training, shard_id, num_shards, prefetch, gpu_aug, crop, resize, cuda, mean_std, seed=1024):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.crop = crop
        self.gpu_aug = gpu_aug
        self.resize = resize
        self.training = training
        self.cuda=cuda
        self.mean=mean_std[0]
        self.std=mean_std[1]
        self.inputs = fn.readers.tfrecord(
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

    def train_aug(self, images):
        images = fn.random_resized_crop(images,
                            size=self.crop,
                            random_area = [0.2, 1.0],
                            device='gpu' if self.gpu_aug else 'cpu')
        # NOTE: hyper-parameters in color_twist is a little bit different from the original paper. 
        # I discover an overly-strong color_twist makes it difficult for cifar10 to learn valid representations. 
        # if np.random.uniform() < 0.8:
        #     images = fn.color_twist(images,
        #                     brightness = np.random.uniform(0.8, 1.2),
        #                     contrast = np.random.uniform(0.8, 1.2), 
        #                     saturation = np.random.uniform(0.8, 1.2), 
        #                     hue = np.random.uniform(-0.1, 0.1), 
        #                     device='gpu' if self.gpu_aug else 'cpu')
        # TODO: RGB2GRAY in DALI would change the channel from 3 to 1, causing errors. 
        # Here I omit it, making it a little bit different from the original SimCLR implementation. 
        # if np.random.uniform() < 0.2:
        #     images = fn.color_space_conversion(images,
        #                     image_type = types.RGB,
        #                     output_type = types.GRAY,
        #                     device='gpu' if self.gpu_aug else 'cpu')
        if np.random.uniform() < 0.5:
            images = fn.gaussian_blur(images,
                            sigma = [0.1, 2.0],
                            window_size = self.crop//20*2+1,
                            device='gpu' if self.gpu_aug else 'cpu')
        flip_lr = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(images,
                            dtype=types.FLOAT,
                            crop=(self.crop, self.crop),
                            mean=self.mean,
                            std=self.std,
                            mirror=flip_lr)
        return images

    def val_aug(self, images):
        images = fn.resize(images,
                            device='gpu' if self.gpu_aug else 'cpu',
                            resize_x=self.resize,
                            resize_y=self.resize,
                            dtype=types.FLOAT,
                            interp_type=types.INTERP_TRIANGULAR)
        flip_lr = False
        images = fn.crop_mirror_normalize(images,
                                dtype=types.FLOAT,
                                crop=(self.crop, self.crop),
                                mean=self.mean,
                                std=self.std,
                                mirror=flip_lr)
        return images

    def define_graph(self):
        images = self.inputs["image/encoded"]
        images = fn.decoders.image(images,
                            device='mixed' if self.gpu_aug else 'cpu',
                            output_type=types.RGB)
        label = self.inputs["image/class/label"]  # if imagenet, then remember -1, since the class ids of imagenet are 1~1000.

        if self.training:
            aug_images1 = self.train_aug(images)
            aug_images2 = self.train_aug(images)
            if self.cuda:
                return aug_images1.gpu(), aug_images2.gpu(), label.gpu()
            else:
                return aug_images1, aug_images2, label
        else:
            aug_images1 = self.val_aug(images)
            aug_images2 = self.val_aug(images)
            if self.cuda:
                return aug_images1.gpu(), aug_images2.gpu(), label.gpu()
            else:
                return aug_images1, aug_images2, label
            

class DALICustomIterator(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size=-1, reader_name="Reader", auto_reset=False, fill_last_batch=False, dynamic_shape=False, last_batch_policy=False):
        super(DALICustomIterator, self).__init__(pipelines, output_map, size, reader_name, auto_reset, fill_last_batch, dynamic_shape, last_batch_policy)

    def __next__(self):
        feed = super().__next__()
        data1, data2, label = feed[0]['view1'], feed[0]['view2'], feed[0]['label']
        label = label.squeeze()
        return (data1, data2,), (label,)

    def __iter__(self):
        # if not reset (after an epoch), reset; if just initialize, ignore
        if self._counter >= self._size or self._size < 0:
            self.reset()
        return self

def DALIDataLoader(tfrec_filenames,
                 tfrec_idx_filenames,
                 shard_id=0,
                 num_shards=1,
                 batch_size=128,
                 num_threads=4,
                 resize=32,
                 crop=32,
                 prefetch=2,
                 training=True,
                 gpu_aug=False,
                 cuda=True,
                 mean_std=None):

    pipes = DALIPipeline(batch_size=batch_size, 
                        num_threads=num_threads, 
                        device_id=torch.cuda.current_device() if cuda else None, 
                        tfrec_filenames=tfrec_filenames, 
                        tfrec_idx_filenames=tfrec_idx_filenames, 
                        training=training, 
                        shard_id=shard_id, 
                        num_shards=num_shards, 
                        prefetch=prefetch, 
                        gpu_aug=gpu_aug, 
                        crop=crop, 
                        resize=resize,
                        cuda=cuda,
                        mean_std=mean_std)
    pipes.build()

    dali_iter = DALICustomIterator(pipes, ['view1', 'view2', 'label'], reader_name="Reader", auto_reset=True, last_batch_policy = False)  #'DROP' if training else 'PARTIAL'
    return dali_iter