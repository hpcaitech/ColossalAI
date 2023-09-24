import argparse
import datetime
import glob
import os
import sys
import time
from functools import partial

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision
from ldm.models.diffusion.ddpm import LatentDiffusion
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import ColossalAIStrategy, DDPStrategy
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset

LIGHTNING_PACK_NAME = "lightning.pytorch."

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

# from ldm.modules.attention import enable_flash_attentions


class DataLoaderX(DataLoader):
    # A custom data loader class that inherits from DataLoader
    def __iter__(self):
        # Overriding the __iter__ method of DataLoader to return a BackgroundGenerator
        # This is to enable data loading in the background to improve training performance
        return BackgroundGenerator(super().__iter__())


def get_parser(**parser_kwargs):
    # A function to create an ArgumentParser object and add arguments to it

    def str2bool(v):
        # A helper function to parse boolean values from command line arguments
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # Create an ArgumentParser object with specifies kwargs
    parser = argparse.ArgumentParser(**parser_kwargs)

    # Add various command line arguments with their default values and descriptions
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="load pretrained checkpoint from stable AI",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    return parser


# A function that returns the non-default arguments between two objects
def nondefault_trainer_args(opt):
    # create an argument parser
    parser = argparse.ArgumentParser()
    # add pytorch lightning trainer default arguments
    parser = Trainer.add_argparse_args(parser)
    # parse the empty arguments to obtain the default values
    args = parser.parse_args([])
    # return all non-default arguments
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# A dataset wrapper class to create a pytorch dataset from an arbitrary object
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# A function to initialize worker processes
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        # divide the dataset into equal parts for each worker
        split_size = dataset.num_records // worker_info.num_workers
        # set the sample IDs for the current worker
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size : (worker_id + 1) * split_size]
        # set the seed for the current worker
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


# Provide functionality for creating data loaders based on provided dataset configurations
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        # Set data module attributes
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        # If a dataset is passed, add it to the dataset configs and create a corresponding dataloader method
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # Instantiate datasets
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        # Instantiate datasets from the dataset configs
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

        # If wrap is true, create a WrappedDataset for each dataset
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        # Check if the train dataset is iterable
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        # Set the worker initialization function of the dataset is iterable or use_worker_init_fn is True
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # Return a DataLoaderX object for the train dataset
        return DataLoaderX(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if is_iterable_dataset else True,
            worker_init_fn=init_fn,
        )

    def _val_dataloader(self, shuffle=False):
        # Check if the validation dataset is iterable
        if isinstance(self.datasets["validation"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        # Return a DataLoaderX object for the validation dataset
        return DataLoaderX(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _test_dataloader(self, shuffle=False):
        # Check if the test dataset is iterable
        is_iterable_dataset = isinstance(self.datasets["train"], Txt2ImgIterableBaseDataset)
        # Set the worker initialization function if the dataset is iterable or use_worker_init_fn is True
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoaderX(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            shuffle=shuffle,
        )

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets["predict"], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoaderX(
            self.datasets["predict"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn
        )


class SetupCallback(Callback):
    # Initialize the callback with the necessary parameters

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    # Save a checkpoint if training is interrupted with keyboard interrupt
    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # Create necessary directories and save configuration files before training starts
    # def on_pretrain_routine_start(self, trainer, pl_module):
    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # Create trainstep checkpoint directory if necessary
            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    os.makedirs(os.path.join(self.ckptdir, "trainstep_checkpoints"), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # Save project config and lightning config as YAML files
            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        # Remove log directory if resuming training and directory already exists
        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    # def on_fit_end(self, trainer, pl_module):
    #     if trainer.global_rank == 0:
    #         ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
    #         rank_zero_info(f"Saving final checkpoint in {ckpt_path}.")
    #         trainer.save_checkpoint(ckpt_path)


# PyTorch Lightning callback for logging images during training and validation of a deep learning model
class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,  # Frequency of batches on which to log images
        max_images,  # Maximum number of images to log
        clamp=True,  # Whether to clamp pixel values to [-1,1]
        increase_log_steps=True,  # Whether to increase frequency of log steps exponentially
        rescale=True,  # Whether to rescale pixel values to [0,1]
        disabled=False,  # Whether to disable logging
        log_on_batch_idx=False,  # Whether to log on batch index instead of global step
        log_first_step=False,  # Whether to log on the first step
        log_images_kwargs=None,
    ):  # Additional keyword arguments to pass to log_images method
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            # Dictionary of logger classes and their corresponding logging methods
            pl.loggers.CSVLogger: self._testtube,
        }
        # Create a list of exponentially increasing log steps, starting from 1 and ending at batch_frequency
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only  # Ensure that only the first process in distributed training executes this method
    def _testtube(
        self,  # The PyTorch Lightning module
        pl_module,  # A dictionary of images to log.
        images,  #
        batch_idx,  # The batch index.
        split,  # The split (train/val) on which to log the images
    ):
        # Method for logging images using test-tube logger
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            # Add image grid to logger's experiment
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,  # The split (train/val) on which to log the images
        images,  # A dictionary of images to log
        global_step,  # The global step
        current_epoch,  # The current epoch.
        batch_idx,
    ):
        # Method for saving image grids to local file system
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # Save image grid as PNG file
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # Function for logging images to both the logger and local file system.
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        # check if it's time to log an image batch
        if (
            self.check_frequency(check_idx)
            and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            # Get logger type and check if training mode is on
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                # Get images from log_images method of the pl_module
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            # Clip images if specified and convert to CPU tensor
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            # Log images locally to file system
            self.log_local(
                pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx
            )

            # log the images using the logger
            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            # switch back to training mode if necessary
            if is_train:
                pl_module.train()

    # The function checks if it's time to log an image batch
    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
            check_idx > 0 or self.log_first_step
        ):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
            return True
        return False

    # Log images on train batch end if logging is not disabled
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        #     self.log_img(pl_module, batch, batch_idx, split="train")
        pass

    # Log images on validation batch end if logging is not disabled and in validation mode
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        # log gradients during calibration if necessary
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py

    def on_train_start(self, trainer, pl_module):
        rank_zero_info("Training is starting")

    # the method is called at the end of each training epoch
    def on_train_end(self, trainer, pl_module):
        rank_zero_info("Training is ending")

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.strategy.reduce(max_memory)
            epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    # get the current time to create a new logging directory
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    # Verify the arguments are both specified
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    # Check if the "resume" option is specified, resume training from the checkpoint if it is true
    ckpt = None
    if opt.resume:
        rank_zero_info("Resuming from {}".format(opt.resume))
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            rank_zero_info("logdir: {}".format(logdir))
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        # Finds all ".yaml" configuration files in the log directory and adds them to the list of base configurations
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        # Gets the name of the current log directory by splitting the path and taking the last element.
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            rank_zero_info("Using base config {}".format(opt.base))
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

        # Sets the checkpoint path of the 'ckpt' option is specified
        if opt.ckpt:
            ckpt = opt.ckpt

    # Create the checkpoint and configuration directories within the log directory.
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    # Sets the seed for the random number generator to ensure reproducibility
    seed_everything(opt.seed)

    # Initialize and save configuration using teh OmegaConf library.
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        # Check whether the accelerator is gpu
        if not trainer_config["accelerator"] == "gpu":
            del trainer_config["accelerator"]
            cpu = True
        else:
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        use_fp16 = trainer_config.get("precision", 32) == 16
        if use_fp16:
            config.model["params"].update({"use_fp16": True})
        else:
            config.model["params"].update({"use_fp16": False})

        if ckpt is not None:
            # If a checkpoint path is specified in the ckpt variable, the code updates the "ckpt" key in the "params" dictionary of the config.model configuration with the value of ckpt
            config.model["params"].update({"ckpt": ckpt})
            rank_zero_info("Using ckpt_path = {}".format(config.model["params"]["ckpt"]))

        model = LatentDiffusion(**config.model.get("params", dict()))
        # trainer and callbacks
        trainer_kwargs = dict()

        # config the logger
        # Default logger configs to  log training metrics during the training process.
        default_logger_cfgs = {
            "wandb": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            },
            "tensorboard": {"save_dir": logdir, "name": "diff_tb", "log_graph": True},
        }

        # Set up the logger for TensorBoard
        default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
            trainer_kwargs["logger"] = WandbLogger(**logger_cfg)
        else:
            logger_cfg = default_logger_cfg
            trainer_kwargs["logger"] = TensorBoardLogger(**logger_cfg)

        # config the strategy, defualt is ddp
        if "strategy" in trainer_config:
            strategy_cfg = trainer_config["strategy"]
            trainer_kwargs["strategy"] = ColossalAIStrategy(**strategy_cfg)
        else:
            strategy_cfg = {"find_unused_parameters": False}
            trainer_kwargs["strategy"] = DDPStrategy(**strategy_cfg)

        # Set up ModelCheckpoint callback to save best models
        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
        if hasattr(model, "monitor"):
            default_modelckpt_cfg["monitor"] = model.monitor
            default_modelckpt_cfg["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint["params"]
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        if version.parse(pl.__version__) < version.parse("1.4.0"):
            trainer_kwargs["checkpoint_callback"] = ModelCheckpoint(**modelckpt_cfg)

        # Create an empty OmegaConf configuration object

        callbacks_cfg = OmegaConf.create()

        # Instantiate items according to the configs
        trainer_kwargs.setdefault("callbacks", [])
        setup_callback_config = {
            "resume": opt.resume,  # resume training if applicable
            "now": now,
            "logdir": logdir,  # directory to save the log file
            "ckptdir": ckptdir,  # directory to save the checkpoint file
            "cfgdir": cfgdir,  # directory to save the configuration file
            "config": config,  # configuration dictionary
            "lightning_config": lightning_config,  # LightningModule configuration
        }
        trainer_kwargs["callbacks"].append(SetupCallback(**setup_callback_config))

        image_logger_config = {
            "batch_frequency": 750,  # how frequently to log images
            "max_images": 4,  # maximum number of images to log
            "clamp": True,  # whether to clamp pixel values to [0,1]
        }
        trainer_kwargs["callbacks"].append(ImageLogger(**image_logger_config))

        learning_rate_logger_config = {
            "logging_interval": "step",  # logging frequency (either 'step' or 'epoch')
            # "log_momentum": True                            # whether to log momentum (currently commented out)
        }
        trainer_kwargs["callbacks"].append(LearningRateMonitor(**learning_rate_logger_config))

        metrics_over_trainsteps_checkpoint_config = {
            "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
            "filename": "{epoch:06}-{step:09}",
            "verbose": True,
            "save_top_k": -1,
            "every_n_train_steps": 10000,
            "save_weights_only": True,
        }
        trainer_kwargs["callbacks"].append(ModelCheckpoint(**metrics_over_trainsteps_checkpoint_config))
        trainer_kwargs["callbacks"].append(CUDACallback())

        # Create a Trainer object with the specified command-line arguments and keyword arguments, and set the log directory
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir

        # Create a data module based on the configuration file
        data = DataModuleFromConfig(**config.data)

        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # Print some information about the datasets in the data module
        for k in data.datasets:
            rank_zero_info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # Configure learning rate based on the batch size, base learning rate and number of GPUs
        # If scale_lr is true, calculate the learning rate based on additional factors
        bs, base_lr = config.data.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = trainer_config["devices"]
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_info(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            rank_zero_info(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            rank_zero_info("++++ NOT USING LR SCALING ++++")
            rank_zero_info(f"Setting learning rate to {model.learning_rate:.2e}")

        # Allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb

                pudb.set_trace()

        import signal

        # Assign melk to SIGUSR1 signal and divein to SIGUSR2 signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # Run the training and validation
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        # Print the maximum GPU memory allocated during training
        print(f"GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
        # if not opt.no_test and not trainer.interrupted:
        #     trainer.test(model, data)
    except Exception:
        # If there's an exception, debug it if opt.debug is true and the trainer's global rank is 0
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        #  Move the log directory to debug_runs if opt.debug is true and the trainer's global
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
