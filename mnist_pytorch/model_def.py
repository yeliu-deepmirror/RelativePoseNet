"""
This example shows how to interact with the Determined PyTorch interface to
build a basic MNIST network.

In the `__init__` method, the model and optimizer are wrapped with `wrap_model`
and `wrap_optimizer`. This model is single-input and single-output.

The methods `train_batch` and `evaluate_batch` define the forward pass
for training and evaluation respectively.
"""

from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from torch import nn

from layers import Flatten

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

import data

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class MNistTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        # Initialize the trial class and wrap the models, optimizers, and LR schedulers.

        # Store trial context for later use.
        self.context = context

        # Create a unique download directory for each rank so they don't overwrite each
        # other when doing distributed training.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.data_downloaded = False

        # Initialize the model and wrap it using self.context.wrap_model().
        self.model = self.context.wrap_model(nn.Sequential(
            nn.Conv2d(1, self.context.get_hparam("n_filters1"), 3, 1),
            nn.ReLU(),
            nn.Conv2d(
                self.context.get_hparam("n_filters1"), self.context.get_hparam("n_filters2"), 3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.context.get_hparam("dropout1")),
            Flatten(),
            nn.Linear(144 * self.context.get_hparam("n_filters2"), 128),
            nn.ReLU(),
            nn.Dropout2d(self.context.get_hparam("dropout2")),
            nn.Linear(128, 10),
            nn.LogSoftmax(),
        ))

        # Initialize the optimizer and wrap it using self.context.wrap_optimizer().
        self.optimizer = self.context.wrap_optimizer(torch.optim.Adadelta(
            self.model.parameters(), lr=self.context.get_hparam("learning_rate"))
        )

    def build_training_data_loader(self) -> DataLoader:
        if not self.data_downloaded:
            self.download_directory = data.download_dataset(
                download_directory=self.download_directory,
                data_config=self.context.get_data_config(),
            )
            self.data_downloaded = True

        train_data = data.get_dataset(self.download_directory, train=True)
        return DataLoader(train_data, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:
        if not self.data_downloaded:
            self.download_directory = data.download_dataset(
                download_directory=self.download_directory,
                data_config=self.context.get_data_config(),
            )
            self.data_downloaded = True

        validation_data = data.get_dataset(self.download_directory, train=False)
        return DataLoader(validation_data, batch_size=self.context.get_per_slot_batch_size())

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        # Define the training forward pass and calculate loss.
        output = self.model(data)
        loss = torch.nn.functional.nll_loss(output, labels)

        # Define the training backward pass and step the optimizer.
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        # return a dictionary with user-defined training metrics
        # Determined will automatically average all the metrics across batches.
        return {"loss": loss}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        data, labels = batch

        output = self.model(data)
        validation_loss = torch.nn.functional.nll_loss(output, labels).item()

        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

        return {"validation_loss": validation_loss, "accuracy": accuracy}
