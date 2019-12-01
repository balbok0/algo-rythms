import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torchvision.transforms import ToTensor

from models import *
from datasets import SpectrogramImageDataset, SpectrogramSequenceDataset, get_paths
from logger import Logger

import traceback

DEBUG = True

GENRE = 'instrumental'
SEQUENCE_LENGTH = 100
IMAGE_LENGTH = 1024
BATCH_IMAGE_SIZE = 8
BATCH_SEQUENCE_SIZE = 256
TEST_BATCH_IMAGE_SIZE = 8
TEST_BATCH_SEQUENCE_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
USE_CUDA = True
PRINT_INTERVAL = 10
PATH = 'data/saved_models/'
MODEL_NAME = 'latest_model.pth'
MODEL_NAME_GENERATOR = 'latest_model_generator.pth'
MODEL_NAME_ADVERSIAL = 'latest_model_adversial.pth'

device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'

def main(logger: Logger):
    train_losses, test_losses = [], []

    print('Creating datasets')
    train_data, val_data, _ = get_paths(GENRE, numpy=True)
    if DEBUG:
        train_data = train_data[:100]
        val_data = val_data[:100]
    train_data = SpectrogramSequenceDataset(train_data, SEQUENCE_LENGTH, BATCH_SEQUENCE_SIZE)
    val_data = SpectrogramSequenceDataset(val_data, SEQUENCE_LENGTH, BATCH_SEQUENCE_SIZE)

    train_loader = DataLoader(train_data, batch_size=BATCH_SEQUENCE_SIZE)
    test_loader = DataLoader(val_data, batch_size=TEST_BATCH_SEQUENCE_SIZE)
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

    # Logger will also save the model in the same folder as the losses
    logger.save_end(
        model, 
        {
            'train_loss': train_losses,
            'test_loss': test_losses
        }
    )

def main_gan(logger: Logger):
    generator_losses, adversial_losses = [], []

    print('Creating datasets')
    train_data, val_data, _ = get_paths(GENRE, numpy=True)
    if DEBUG:
        train_data = train_data[:100]
        val_data = val_data[:100]
    train_data = SpectrogramImageDataset(train_data, IMAGE_LENGTH, ToTensor())
    val_data = SpectrogramImageDataset(val_data, IMAGE_LENGTH, ToTensor())

    train_loader = DataLoader(train_data, batch_size=BATCH_IMAGE_SIZE)
    test_loader = DataLoader(val_data, batch_size=TEST_BATCH_IMAGE_SIZE)
    print('Finished creating datasets')

    print('Creating model')
    model_generator = SimpleGenerator(200).to(device)
    model_adversial = SimpleDecoder().to(device)

    optimizer_generator = optim.Adam(model_generator.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_adversial = optim.Adam(model_adversial.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss()
    print('Finished creating model')

    for epoch in range(1, 1 + EPOCHS):
        loss_generator, _ = train_generator(
            model_generator, model_adversial,
            optimizer_generator, criterion,
            train_loader
        )

        loss_adversial = train_adversary(
            model_adversial, model_generator,
            optimizer_adversial, criterion,
            train_loader
        )

        generator_losses.append((epoch, loss_generator))
        adversial_losses.append((epoch, loss_adversial))


    # Save the model parameters to be used again. Change MODEL_NAME to reflect model name and version
    if not os.path.isdir(PATH):
        os.makedirs(PATH)
    torch.save(model_generator.state_dict(), PATH + MODEL_NAME_GENERATOR)
    torch.save(model_adversial.state_dict(), PATH + MODEL_NAME_ADVERSIAL)


if __name__ == "__main__":
    logger = Logger()
    try:
        # main(logger)
        main_gan(logger)
    except Exception as e:
        logger.clean()
        print(traceback.format_exc())
