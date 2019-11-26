import torch.nn as nn
import torch.nn.functional as F


# CODE FOR USING LSTM INSTEAD OF GRU (Task 2)
TEMPERATURE = 0.5

class SimpleLSTM(nn.Module):
    def __init__(self, data_size):
        super(SimpleLSTM, self).__init__()

        self.data_size = data_size
        self.lstm1 = nn.LSTM(self.data_size, self.data_size, batch_first=True)

        self.best_accuracy = -1

    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        print('x: {}'.format(x.shape))
        x = x.view(sequence_length, batch_size, -1)
        print('x: {}'.format(x.shape))
        if hidden_state is None:
            x, hidden_state = self.lstm1(x)
        else:
            x, hidden_state = self.lstm1(x, hidden_state)

        x = x.contiguous().view(batch_size, sequence_length, -1)

        return x, hidden_state
