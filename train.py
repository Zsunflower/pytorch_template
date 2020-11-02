import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prepare_data()
        self.setup_train()

    def prepare_data(self):
        train_ds =
        val_ds   =
        test_ds  =
        self.train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_ds,     batch_size=self.args.batch_size, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_ds,   batch_size=self.args.batch_size, shuffle=False)

    def setup_train(self):
        self.model =
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.lr).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_one_epoch(self):
        train_loss = 0.0
        self.model.train()
        for i, sample in enumerate(tqdm(self.train_loader)):
            X, Y_true = sample['X'].to(
                self.device), sample['Y'].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, Y_true)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def evaluate(self):
        val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_loader)):
                X, Y_true = sample['X'].to(
                    self.device), sample['Y'].to(self.device)
                output = self.model(X)
                loss = self.criterion(output, Y_true)
                val_loss += loss.item()
        return val_loss / (len(self.val_loader))

    def run(self):
        min_loss = 10e4
        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(self.model.state_dict(),
                    os.path.join(self.args.checkpoint,
                                 "{}_{}_{}.pth".format(self.args.name, epoch, val_loss)))


def main(args):
    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch training script')
    parser.add_argument('--name', default=None, type=str,help='Name script')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Checkpoint directory')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Number of epochs to train')
    args = parser.parse_args()
    main(args)
