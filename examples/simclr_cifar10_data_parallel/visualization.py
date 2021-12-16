import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.simclr import SimCLR
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

log_name = 'cifar-simclr'
epoch = 800

fea_flag = True
tsne_flag = True
plot_flag = True

if fea_flag:
    path = f'ckpt/{log_name}/epoch{epoch}-tp0-pp0.pt'
    net = SimCLR('resnet18').cuda()
    print(net.load_state_dict(torch.load(path)['model']))

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_dataset = CIFAR10(root='./dataset', train=True, transform=transform_eval)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)

    test_dataset = CIFAR10(root='./dataset', train=False, transform=transform_eval)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

def feature_extractor(model, loader):
    model.eval()
    all_fea = []
    all_targets = []
    for img, target in loader:
        img = img.cuda()
        fea = model.backbone(img)
        all_fea.append(fea.detach().cpu())
        all_targets.append(target)
    all_fea = torch.cat(all_fea)
    all_targets = torch.cat(all_targets)
    return all_fea.numpy(), all_targets.numpy()

if tsne_flag:
    train_fea, train_targets = feature_extractor(net, train_dataloader)
    train_embedded = TSNE(n_components=2).fit_transform(train_fea)
    test_fea, test_targets = feature_extractor(net, test_dataloader)
    test_embedded = TSNE(n_components=2).fit_transform(test_fea)
    np.savez('results/embedding.npz', train_embedded=train_embedded, train_targets=train_targets, test_embedded=test_embedded, test_targets=test_targets)

if plot_flag: 
    npz = np.load('embedding.npz')
    train_embedded = npz['train_embedded']
    train_targets = npz['train_targets']
    test_embedded = npz['test_embedded']
    test_targets = npz['test_targets']

    plt.figure(figsize=(16,16))
    for i in range(len(np.unique(train_targets))):
        plt.scatter(train_embedded[train_targets==i,0], train_embedded[train_targets==i,1], label=i)
    plt.title('train')
    plt.legend()
    plt.savefig('results/train_tsne.png')

    plt.figure(figsize=(16,16))
    for i in range(len(np.unique(test_targets))):
        plt.scatter(test_embedded[test_targets==i,0], test_embedded[test_targets==i,1], label=i)
    plt.title('test')
    plt.legend()
    plt.savefig('results/test_tsne.png')
