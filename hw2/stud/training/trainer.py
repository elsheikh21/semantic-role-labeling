import os
import time
import logging

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from training.earlystopping import EarlyStopping


class Trainer:
    def __init__(self, model, loss_function, optimizer, epochs,
                 num_classes, verbose, writer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self._verbose = verbose
        self._epochs = epochs + 1
        self.writer = writer

    def train(self, train_dataset, valid_dataset, save_to=None):
        train_loss, best_val_loss = 0.0, float(1e4)
        es = EarlyStopping(patience=5)
        # lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=5, verbose=True)
        for epoch in tqdm(range(1, self._epochs), desc="Training"):
            epoch_loss = 0.0
            self.model.train()
            for step, sample in tqdm(enumerate(train_dataset), desc='Fit On Batches',
                                     leave=False, total=len(train_dataset)):
                inputs, labels = sample["inputs"], sample["outputs"]
                pos, predicates = sample["pos"], sample["predicates"]

                self.optimizer.zero_grad()
                predictions = self.model(inputs, predicates, pos)

                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)

                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.)  # Gradient Clipping
                self.optimizer.step()
                epoch_loss += sample_loss.tolist()

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss

            valid_loss = self.evaluate(valid_dataset)
            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', avg_epoch_loss )
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', valid_loss)

            # lr_scheduler.step(valid_loss)

            is_best = valid_loss <= best_val_loss
            if is_best:
                logging.info("Model Checkpoint saved")
                best_val_loss = valid_loss
                model_dir = os.path.join(os.getcwd(), 'model',
                                         f'{self.model.name}_ckpt_best.pt')
                self.model._save(model_dir)
            if self._verbose > 0:
                print(f'| Epoch: {epoch:02} | Loss: {avg_epoch_loss:.4f} | Val Loss: {valid_loss:.4f} |')
            if es.step(valid_loss):
                print(f"Training Stopped early, epoch #: {epoch}")
                break
        avg_epoch_loss = train_loss / self._epochs
        if save_to is not None:
            self.model._save(save_to)

        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in tqdm(valid_dataset, desc='Evaluating',
                               leave=False, total=len(valid_dataset)):
                inputs, labels = sample["inputs"], sample["outputs"]
                pos, predicates = sample["pos"], sample["predicates"]
                predictions = self.model(inputs, predicates, pos)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)
                valid_loss += sample_loss.tolist()
        return valid_loss / len(valid_dataset)
