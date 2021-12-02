# Overview

Here is an example of applying PreAct-ResNet18 to train SimCLR on CIFAR10. We use 1x RTX 3090 in this example. 

# How to run
Specified in:
```shell
bash train.sh
```

# Results
The loss curve of SimCLR self-supervised training is as follows:
![SimCLR Loss Curve](./ssl_loss.png)
The loss curve of linear evaluation is as follows:
![Linear Evaluation Loss Curve](./linear_eval_loss.png)
The accuracy curve of linear evaluation is as follows:
![Linear Evaluation Accuracy](./linear_eval_acc.png)
The t-SNE of the training set of CIFAR10 is as follows:
![train tSNE](./train_tsne.png)
The t-SNE of the test set of CIFAR10 is as follows:
![test tSNE](./test_tsne.png)