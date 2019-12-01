import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# CODE FOR USING LSTM INSTEAD OF GRU (Task 2)
TEMPERATURE = 0.5

class SimpleLSTM(nn.Module):
    def __init__(self, data_size):
        super(SimpleLSTM, self).__init__()

        self.data_size = data_size
        self.lstm1 = nn.LSTM(self.data_size, self.data_size, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM((int)(self.data_size / 2), self.data_size, batch_first=True)
        self.best_accuracy = -1

    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        x = x.view(sequence_length, batch_size, -1)

        if hidden_state is None:
            x, hidden_state = self.lstm1(x)
        else:
            x, hidden_state = self.lstm1(x, hidden_state)

        x = self.drop1(x)

        # Take care of odd shapes. If the dimension is odd then the just drop the last value and then add
        # adjacent elements
        if x.shape[-1] % 2 == 1:
            x = np.delete(x, 0, -1)
        x = x[:, :, 1::2] + x[:, :, 0::2]

        x, hidden_state = self.lstm2(x, hidden_state)
        x = x.contiguous().view(batch_size, sequence_length, -1)

        return x, hidden_state
