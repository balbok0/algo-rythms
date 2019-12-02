import torch

import numpy as np

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(model, device, optimizer, criterion, train_loader, lr, epoch, log_interval):
    losses = []
    hidden = None
    for batch_idx, (data, label) in enumerate(tqdm(train_loader, total=len(train_loader))):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches.
        # Otherwise the backward would try to go all the way to the beginning every time.
        if hidden is not None:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data)
        loss = criterion(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))
    return np.mean(losses)


def train_adversary(adversial, generator, optimizer_adversial, criterion, train_loader):
    loss = 0.
    for x in train_loader:
        batch_size = len(x)

        x = x.to(device)
        z = torch.tensor(np.random.normal(0, 1, (batch_size, 200)), dtype=torch.float32).to(device)

        adversial.zero_grad()
        y_real_pred = adversial(x)

        # Create labels
        ones = torch.ones(batch_size).long().to(device)
        zeros = torch.zeros(batch_size).long().to(device)

        # Make generated samples
        generated = generator(z)
        y_fake_pred = adversial(generated)

        # Calculate overall loss
        loss_real = criterion(y_real_pred, ones)
        loss_fake = criterion(y_fake_pred, zeros)
        loss_total = loss_fake + loss_real

        # Backprop
        loss_total.backward()
        optimizer_adversial.step()

        loss += loss_total
    return loss


def train_generator(generator, adversial, optimizer_generator, criterion, train_loader):
    loss = 0.
    for x in tqdm(train_loader, desc="Training Generator"):
        batch_size = len(x)

        z = torch.tensor(np.random.normal(0, 1, (batch_size, 200)), dtype=torch.float32).to(device)

        generator.zero_grad()
        generated = generator(z)
        y_fake = adversial(generated)

        ones = torch.ones(batch_size, dtype=torch.long).to(device)

        loss_total = criterion(y_fake, ones)
        loss_total.backward()
        optimizer_generator.step()

        loss += loss_total
    return loss, generated


def test(model, device, criterion, test_loader):
    test_loss = 0
    correct = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)
            test_loss += criterion(output, label).item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss
