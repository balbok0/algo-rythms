import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchvision.transforms import ToTensor

from models import train, test, SimpleLSTM
from datasets import SpectrogramImageDataset, SpectrogramSequenceDataset, get_paths
from logger import Logger

import traceback

DEBUG = True

GENRE = 'instrumental'
SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
USE_CUDA = True
PRINT_INTERVAL = 10
PATH = 'data/saved_models/'
MODEL_NAME = 'latest_model.pth'

device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'

def main(logger: Logger):
    train_losses, test_losses = [], []

    print('Creating datasets')
    train_data, val_data, _ = get_paths(GENRE, numpy=True)
    if DEBUG:
        train_data = train_data[:100]
        val_data = val_data[:100]
    train_data = SpectrogramSequenceDataset(train_data, SEQUENCE_LENGTH, BATCH_SIZE)
    val_data = SpectrogramSequenceDataset(val_data, SEQUENCE_LENGTH, BATCH_SIZE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE)
    print('Finished creating datasets')

    print('Creating model')
    model = SimpleLSTM(SEQUENCE_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    print('Finished creating model')

    for epoch in range(1, 1 + EPOCHS):
        lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
        train_loss = train(model, device, optimizer, criterion, train_loader, lr, epoch, PRINT_INTERVAL)
        test_loss = test(model, device, criterion, test_loader)

        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))

    # logger.save_end(
    #     {
    #         'train_loss': train_losses,
    #         'test_loss': test_losses
    #     }
    # )

    # Save the model parameters to be used again. Change MODEL_NAME to reflect model name and version
    if not os.path.isdir(PATH):
        os.makedirs(PATH)
    torch.save(model.state_dict(), PATH + MODEL_NAME)



if __name__ == "__main__":
    logger = Logger()
    try:
        main(logger)
    except Exception as e:
        logger.clean()
        print(traceback.format_exc())
