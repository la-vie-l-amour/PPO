
import torch

class Trainer:
    def __init__(self, model, batch_size):

        self.batch_size = batch_size
        self.model =
        self.optimizer = torch.optim.Adam([
        {'params': model.actor.parameters(), 'lr': args.lr_actor},
        {'params': model.critic.parameters(), 'lr': args.lr_critic}
    ])
        self.

    def train(self):

    def reset_models(self):

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batch for loss, n_batch in zip(losses, n_batches)]
        return sum(epoch_loss) / epoch
