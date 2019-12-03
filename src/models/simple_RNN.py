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
        new_data = (int)(self.data_size / 2)
        self.lstm2 = nn.LSTM(new_data, self.data_size, batch_first=True)
        # ##########################################
        # self.lstm3 = nn.LSTM(new_data, self.data_size, batch_first=True)
        # #############################################
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
            hidden_state = (None, None)

        hs_1, hs_2 = hidden_state[0], hidden_state[1]

        sig = nn.Sigmoid()
        x = sig(x /10) * 5

        print('x: {}'.format(x.shape))
        x = x.view(sequence_length, batch_size, -1)
        print('x: {}'.format(x.shape))
        if hs_1 is None:
            x, hs_1 = self.lstm1(x)
        else:
            x, hs_1 = self.lstm1(x, hs_1)

        x = self.crop(x)

        if hs_2 is None:
            x, hs_2 = self.lstm2(x)
        else:
            x, hs_2 = self.lstm2(x, hs_2)
# ############################
#         x = self.crop(x)
#         x, hidden_state = self.lstm3(x, hidden_state)
#         #############################

        x = x.contiguous().view(batch_size, sequence_length, -1)

        return x, (hs_1, hs_2)
