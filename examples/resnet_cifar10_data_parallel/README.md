# Train ResNet34 on CIFAR10

## Prepare Dataset

In the script, we used CIFAR10 dataset provided by the `torchvision` library. The code snippet is shown below:

```python
train_dataset = CIFAR10(
        root=Path(os.environ['DATA']),
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                    0.2023, 0.1994, 0.2010]),
            ]
        )
    )
```

Firstly, you need to specify where you want to store your CIFAR10 dataset by setting the environment variable `DATA`. 

```bash
export DATA=/path/to/data

# example
# this will store the data in the current directory
export DATA=$PWD/data
```

The `torchvison` module will download the data automatically for you into the specified directory.


## Run training

We provide two examples of training resnet 34 on the CIFAR10 dataset. One example is with engine and the other is 
with the trainer. You can invoke the training script by the following command. This batch size and learning rate 
are for a single GPU. Thus, in the following command, `nproc_per_node` is 1, which means there is only one process 
invoked. If you change `nproc_per_node`, you will have to change the learning rate accordingly as the global batch
size has changed.

```bash
# with engine
python -m torch.distributed.launch --nproc_per_node 1 run_resnet_cifar10_with_engine.py

# with trainer
python -m torch.distributed.launch --nproc_per_node 1 run_resnet_cifar10_with_trainer.py
```