import colossalai
from colossalai.amp import AMP_TYPE
from colossalai.logging import get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import get_dataloader
from colossalai.utils.common import print_rank_0
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
import torch
import glob
import os

from torch.utils.data.dataloader import T
from dataset.webtext import WebtextDataset
from torch.utils.data import DataLoader
from model.bert import BERTLMLoss
from dataset.bert_dali_dataloader import DaliDataloader, build_dali_train
from transformers import BertForPreTraining, BertConfig
from model.bert import BertForPreTraining1D


BATCH_SIZE = 32
NUM_EPOCHS = 60

CONFIG = dict(
    parallel=dict(
        pipeline=dict(size=1),
        tensor=dict(size=1, mode='1d'),
    ),
    # fp16=dict(
    #     mode=AMP_TYPE.NAIVE,
    #     clip_grad=1
    # ),
    gradient_accumulation=16,
)

# need to manully modify the sourcecode of Huggingface Transformers
# modelingbert.py -> class BertForPretraining -> set return_dict = False


def run_trainer():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    if args.from_torch:
        colossalai.launch_from_torch(config=CONFIG,
                                     host=os.environ['MASTER_ADDR'],
                                     port=os.environ['MASTER_PORT'])
    else:
        colossalai.launch_from_slurm(config=CONFIG,
                                     host=args.host,
                                     port=29500,
                                     seed=42)

    parser = colossalai.get_default_parser()
    logger = get_dist_logger()

    # instantiate your components

    # instantiate hugging face BERT
    # config = BertConfig()
    # model = BertForPreTraining(config)

    model = BertForPreTraining1D()
    optimizer = torch.optim.Adam(model.parameters())

    # dataset
    TRAIN_RECS = '/mnt/tier1/project/p200012/dataset/wikitext/train/tfrecords/*'
    TRAIN_IDX = '/mnt/tier1/project/p200012/dataset/wikitext/train/tfrecords_idx/*'
    VAL_RECS = '/mnt/tier1/project/p200012/dataset/wikitext/eval/tfrecords/*'
    VAL_IDX = '/mnt/tier1/project/p200012/dataset/wikitext/eval/tfrecords_idx/*'
    logger.info('Processing train dataloader:')
    train_dataloader = DaliDataloader(
        sorted(glob.glob(TRAIN_RECS)),
        sorted(glob.glob(TRAIN_IDX)),
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        cuda=True,
        training=True,

    )

    test_dataloader = DaliDataloader(
        sorted(glob.glob(VAL_RECS)),
        sorted(glob.glob(VAL_IDX)),
        batch_size=BATCH_SIZE,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        cuda=True,
        training=False,
    )

    criterion = BERTLMLoss()

    logger.info("components are built")
    engine, dataloader, _,  _ = colossalai.initialize(model, optimizer, criterion, train_dataloader)

    trainer = Trainer(engine=engine)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(),
        hooks.TensorboardHook(log_dir='./tb_logs/gpt_uncut/', ranks=[0]),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.fit(
        train_dataloader=dataloader,
        epochs=NUM_EPOCHS,
        hooks=hook_list,
        display_progress=True,
    )


if __name__ == '__main__':
    # for ddp debugging
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    run_trainer()
