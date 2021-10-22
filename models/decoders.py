"""
Decoder for SELD network / MAP task, one input, several output
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import init_layer, init_gru, PositionalEncoding


class SeldDecoder(nn.Module):
    """
    Decoder for SELD.
    input: batch_size x n_frames x input_size
    """
    def __init__(self, n_output_channels, n_classes: int = 12, output_format: str = 'reg_xyz',
                 decoder_type: str = None, freq_pool: str = None, decoder_size: int = 128, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.decoder_type = decoder_type
        self.freq_pool = freq_pool
        self.doa_format = output_format

        logger = logging.getLogger('lightning')
        logger.info('Map decoder type: {}'.format(self.decoder_type))
        assert self.decoder_type in ['gru', 'bigru', 'lstm', 'bilstm', 'transformer'], \
            'Invalid decoder type {}'.format(self.decoder_type)

        if self.decoder_type == 'gru':
            self.gru_input_size = n_output_channels
            self.gru_size = decoder_size
            self.fc_size = self.gru_size

            self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=False, dropout=0.3)
            init_gru(self.gru)
        elif self.decoder_type == 'bigru':
            self.gru_input_size = n_output_channels
            self.gru_size = decoder_size
            self.fc_size = self.gru_size * 2

            self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            init_gru(self.gru)
        elif self.decoder_type == 'lstm':
            self.lstm_input_size = n_output_channels
            self.lstm_size = decoder_size
            self.fc_size = self.lstm_size

            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.gru_size,
                                num_layers=2, batch_first=True, bidirectional=False, dropout=0.3)
            init_gru(self.lstm)
        elif self.decoder_type == 'bilstm':
            self.lstm_input_size = n_output_channels
            self.lstm_size = decoder_size
            self.fc_size = self.lstm_size * 2

            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.gru_size,
                               num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            init_gru(self.lstm)
        elif self.decoder_type == 'transformer':
            dim_feedforward = 1024
            self.decoder_input_size = n_output_channels
            self.fc_size = self.decoder_input_size
            self.pe = PositionalEncoding(pos_len=2000, d_model=self.decoder_input_size, dropout=0.0)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.decoder_input_size,
                                                       dim_feedforward=dim_feedforward, nhead=8, dropout=0.2)
            self.decoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            raise NotImplementedError('decoder type: {} is not implemented'.format(self.decoder_type))

        # sed
        self.event_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.event_dropout_1 = nn.Dropout(p=0.2)
        self.event_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.event_dropout_2 = nn.Dropout(p=0.2)

        # doa
        self.x_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.y_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.z_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.x_dropout_1 = nn.Dropout(p=0.2)
        self.y_dropout_1 = nn.Dropout(p=0.2)
        self.z_dropout_1 = nn.Dropout(p=0.2)
        self.x_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.y_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.z_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.x_dropout_2 = nn.Dropout(p=0.2)
        self.y_dropout_2 = nn.Dropout(p=0.2)
        self.z_dropout_2 = nn.Dropout(p=0.2)

        self.init_weights()

    def init_weights(self):
        init_layer(self.event_fc_1)
        init_layer(self.event_fc_2)
        init_layer(self.x_fc_1)
        init_layer(self.y_fc_1)
        init_layer(self.z_fc_1)
        init_layer(self.x_fc_2)
        init_layer(self.y_fc_2)
        init_layer(self.z_fc_2)

    def forward(self, x):
        """
        :params x: (batch_size, n_channels, n_timesteps/n_frames (downsampled), n_features/n_freqs (downsampled)
        """
        if self.freq_pool == 'avg':
            x = torch.mean(x, dim=3)
        elif self.freq_pool == 'max':
            (x, _) = torch.max(x, dim=3)
        elif self.freq_pool == 'avg_max':
            x1 = torch.mean(x, dim=3)
            (x, _) = torch.max(x, dim=3)
            x = x1 + x
        else:
            raise NotImplementedError('freq pooling {} is not implemented'.format(self.freq_pool))
        '''(batch_size, feature_maps, time_steps)'''

        # swap dimension: batch_size, n_timesteps, n_channels/n_features
        x = x.transpose(1, 2)

        if self.decoder_type in ['gru', 'bigru']:
            x, _ = self.gru(x)
        elif self.decoder_type in ['lsmt', 'bilstm']:
            x, _ = self.lstm(x)
        elif self.decoder_type == 'transformer':
            x = x.transpose(1, 2)  # undo swap: batch size,  n_features, n_timesteps,
            x = self.pe(x)  # batch_size, n_channels/n features, n_timesteps
            x = x.permute(2, 0, 1)  # T (n_timesteps), N (batch_size), C (n_features)
            x = self.decoder_layer(x)
            x = x.permute(1, 0, 2)  # batch_size, n_timesteps, n_features

        # SED: multi-label multi-class classification, without sigmoid
        event_frame_logit = F.relu_(self.event_fc_1(self.event_dropout_1(x)))  # (batch_size, time_steps, n_classes)
        event_frame_logit = self.event_fc_2(self.event_dropout_2(event_frame_logit))

        # DOA: regression
        x_output = F.relu_(self.x_fc_1(self.x_dropout_1(x)))
        x_output = torch.tanh(self.x_fc_2(self.x_dropout_2(x_output)))
        y_output = F.relu_(self.y_fc_1(self.y_dropout_1(x)))
        y_output = torch.tanh(self.y_fc_2(self.y_dropout_2(y_output)))
        z_output = F.relu_(self.z_fc_1(self.z_dropout_1(x)))
        z_output = torch.tanh(self.z_fc_2(self.z_dropout_2(z_output)))
        doa_output = torch.cat((x_output, y_output, z_output), dim=-1)  # (batch_size, time_steps, 3 * n_classes)

        output = {
            'event_frame_logit': event_frame_logit,
            'doa_frame_output': doa_output,
        }

        return output