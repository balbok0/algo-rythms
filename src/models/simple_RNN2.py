import torch.nn as nn
import numpy as np
import torch.nn.functional as F


TEMPERATURE = 0.5

class SimpleLSTM2(nn.Module):
    def __init__(self, data_size):
        super(SimpleLSTM2, self).__init__()

        self.data_size = data_size
        self.lstm1 = nn.LSTM(self.data_size, self.data_size, batch_first=True)
        new_data = (int)(self.data_size / 2)
        self.lstm2 = nn.LSTM(new_data, self.data_size, batch_first=True)
        new_data = (int)(new_data / 2)
        self.lstm3 = nn.LSTM(new_data, self.data_size, batch_first=True)
        new_data = (int)(new_data / 2)
        self.lstm4 = nn.LSTM(new_data, self.data_size, batch_first=True)
        new_data = (int)(new_data / 2)
        self.lstm5 = nn.LSTM(new_data, self.data_size, batch_first=True)
        self.best_accuracy = -1

    # Crops the given input by adding adjacent elements. Takes care of odd shapes by dropping one element from the end.
    def crop(self, x):
        if x.shape[-1] % 2 == 1:
            x = np.delete(x, 0, -1)
        return x[:, :, 1::2] + x[:, :, 0::2]
    
    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        if hidden_state is None:
            hidden_state = (None, None, None, None, None)

        hs_1, hs_2, hs_3, hs_4, hs_5 = hidden_state[0], hidden_state[1], hidden_state[2], hidden_state[3], hidden_state[4]

        sig = nn.Sigmoid()
        x = sig(x /10) * 5

        # print('x: {}'.format(x.shape))
        x = x.view(sequence_length, batch_size, -1)
        # print('x: {}'.format(x.shape))
        # First LSTM
        if hs_1 is None:
            x, hs_1 = self.lstm1(x)
        else:
            x, hs_1 = self.lstm1(x, hs_1)


        # Second LSTM
        x = self.crop(x)
        if hs_2 is None:
            x, hs_2 = self.lstm2(x)
        else:
            x, hs_2 = self.lstm2(x, hs_2)


        # Third LSTM
        x = self.crop(self.crop(x))
        if hs_3 is None:
            x, hs_3 = self.lstm3(x)
        else:
            x, hs_3 = self.lstm3(x, hs_3)


        # Fourth LSTM
        x = self.crop(self.crop(self.crop(x)))
        if hs_4 is None:
            x, hs_4 = self.lstm4(x)
        else:
            x, hs_4 = self.lstm4(x, hs_4)


        # Fifth LSTM
        x = x = self.crop(self.crop(self.crop(self.crop(x))))
        if hs_5 is None:
            x, hs_5 = self.lstm5(x)
        else:
            x, hs_5 = self.lstm5(x, hs_5)

        x = x.contiguous().view(batch_size, sequence_length, -1)

        return x, (hs_1, hs_2, hs_3, hs_4, hs_5)
