from abc import ABC

import loralib as lora
from chatgpt.dataset import RewardDataset
from chatgpt.nn import PairWiseLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        num_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(self,
                 model,
                 train_dataset: RewardDataset,
                 eval_dataset: RewardDataset,
                 batch_size: int = 1,
                 num_epochs: int = 2,
                 optim_kwargs: dict = {'lr': 1e-4}) -> None:
        super().__init__()
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.loss_fn = PairWiseLoss()
        self.optimizer = Adam(self.model.parameters(), **optim_kwargs)
        self.epochs = num_epochs

    def fit(self, use_lora):
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch')
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()), desc='Train step of epoch %d' % epoch)
            # train
            if use_lora > 0:
                print("Using Lora")
                lora.mark_only_lora_as_trainable(self.model)
            else:
                self.model.train()
            for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).cuda()
                c_mask = c_mask.squeeze(1).cuda()
                reject_ids = reject_ids.squeeze(1).cuda()
                r_mask = r_mask.squeeze(1).cuda()
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                loss = self.loss_fn(chosen_reward, reject_reward)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                step_bar.update()
                step_bar.set_postfix({'loss': loss.item()})

            # eval
            self.model.eval()
            for chosen_ids, c_mask, reject_ids, r_mask in self.eval_dataloader:
                dist = 0
                chosen_ids = chosen_ids.squeeze(1).cuda()
                c_mask = c_mask.squeeze(1).cuda()
                reject_ids = reject_ids.squeeze(1).cuda()
                r_mask = r_mask.squeeze(1).cuda()
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                dist += (chosen_reward - reject_reward)
            dist_mean = dist / self.eval_dataloader.__len__()
            epoch_bar.update()
            step_bar.set_postfix({'loss': loss.item(), 'dist_mean': dist_mean.item()})
            step_bar.close()
