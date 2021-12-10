from datetime import datetime
import os
import os.path as osp

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
import argparse
from colossalai.trainer import Trainer

from dataloaders.vessel import RetinalVesselSegmentation
from dataloaders import custom_transforms as tr
from networks.unet.unet_model import UNet

local_path = osp.dirname(osp.abspath(__file__))




def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument('--datasetTrain', type=list, default=[1], help='train folder id contain images ROIs to train range from [1,2,3,4]')
    parser.add_argument('--datasetTest', type=list, default=[1], help='test folder id contain images ROIs to test one of [1,2,3,4]')
    parser.add_argument('--data-dir', default='../../../../Dataset/Fundus/', help='data root path')
    parser.add_argument('--config', default='../../../../Dataset/Fundus/', help='config path')
    args = parser.parse_args()
    # datasetTrain = [0,1,2]
    # datasetTest = [3]
    # data_dir = '../RVS'
    gpc.load_config(args.config)
    # gpc.init_global_dist(rank=0,
    #       world_size=1,
    #       host='127.0.0.1',
    #       port=8888,
    #       backend='nccl')
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        colossalai.launch_from_openmpi(config=args.config,
        host='gpu01',
        port='11455',
        backend='nccl')
    elif 'SLURM_PROCID' in os.environ:
        colossalai.launch_from_slurm(config=args.config,
        host='localhost',
        port='11455',
        backend='nccl')
    elif 'WORLD_SIZE' in os.environ:
        colossalai.launch_from_torch(config=args.config,
        host='localhost',
        port='11455',
        backend='nccl')
    else:
        colossalai.launch(
            config=args.config,
            host='localhost',
            port='11455',
            rank=0,
            world_size=1,
            backend='nccl')
        

    now = datetime.now()
    # args.out = osp.join(local_path, 'logs', 'test'+str(args.datasetTest[0]), 'lam'+str(args.lam), now.strftime('%Y%m%d_%H%M%S.%f'))
    # os.makedirs(args.out)
    # with open(osp.join(args.out, 'config.yaml'), 'w') as f:
    #     yaml.safe_dump(args.__dict__, f, default_flow_style=False)


    colossalai.context.config.Config.from_file(args.config)
    torch.cuda.manual_seed(1337)
    splitidTrain = []
    for x in args.datasetTrain:
        splitidTrain.append(int(x))
    splitidTest = []
    for x in args.datasetTest:
        splitidTest.append(int(x))


    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.RandomScaleCrop(256),
        # tr.RandomCrop(512),
        # tr.RandomRotate(),
        # tr.RandomFlip(),
        # tr.elastic_transform(),
        # tr.add_salt_pepper_noise(),
        # tr.adjust_light(),
        # tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.RandomCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = RetinalVesselSegmentation(base_dir=args.data_dir, phase='train', splitid=splitidTrain,
                                                         transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)

    domain_val = RetinalVesselSegmentation(base_dir=args.data_dir, phase='test', splitid=splitidTest,
                                       transform=composed_transforms_ts)
    val_loader = DataLoader(domain_val, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)

    # 2. model
    model = UNet(3,2).cuda()
    print('parameter numer:', sum([p.numel() for p in model.parameters()]))

    # # load weights
    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     pretrained_dict = checkpoint['model_state_dict']
    #     model_dict = model.state_dict()
    #     # 1. filter out unnecessary keys
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     # 2. overwrite entries in the existing state dict
    #     model_dict.update(pretrained_dict)
    #     # 3. load the new state dict
    #     model.load_state_dict(model_dict)

    #     print('Before ', model.centroids.data)
    #     model.centroids.data = centroids_init(model, args.data_dir, args.datasetTrain, composed_transforms_ts)
    #     print('Before ', model.centroids.data)
    #     # model.freeze_para()

    # 3. optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.99)
    )
    # optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 4. Engine
    def batch_data_process_func(sample):
        image = None
        label = None
        for domain in sample:
            if image is None:
                image = domain['image']
                label = domain['label']
            else:
                image = torch.cat([image, domain['image']], 0)
                label = torch.cat([label, domain['label']], 0)
        return image,label
    logger = get_dist_logger('root')
    schedule=colossalai.engine.schedule.NonPipelineSchedule()
    # lr_scheduler=colossalai.nn.lr_scheduler.CosineAnnealingLR(optim, 1000)
    criterion=torch.nn.BCELoss()
    schedule.batch_data_process_func = batch_data_process_func
    # engine = colossalai.engine.Engine(
    # model=model,
    # criterion=criterion,
    # optimizer=optim,
    # )
    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model=model,
    optimizer=optim,
    criterion=criterion,
    train_dataloader=train_loader,
    test_dataloader=val_loader,
    verbose=True,)

    # engine.schedule.batch_data_process_func = batch_data_process_func

    logger.info("engine is built", ranks=[0])
    trainer = Trainer(engine=engine,
          schedule=schedule, logger=logger)
    logger.info("trainer is built", ranks=[0])

    logger.info("start training", ranks=[0])
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=gpc.config.num_epochs,
        display_progress=True,
        test_interval=2
    )

    # trainer = Trainer.Trainer(
    #     cuda=cuda,
    #     model=model,
    #     lr=args.lr,
    #     lr_decrease_rate=args.lr_decrease_rate,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optim=optim,
    #     out=args.out,
    #     max_epoch=args.max_epoch,
    #     stop_epoch=args.stop_epoch,
    #     interval_validate=args.interval_validate,
    #     batch_size=args.batch_size,
    # )
    # trainer.epoch = start_epoch
    # trainer.iteration = start_iteration
    # trainer.train()

if __name__ == '__main__':
    main()
