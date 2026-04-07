"""
Multi-head variant of SwinCVS.
Replaces the single Linear(256, 3) head with three separate Linear(256, 1) heads.
Same for the multiclassifier SwinV2 head.
Output shape is identical: (batch, 3) — concatenated from 3 separate heads.
"""

import torch
import torch.nn as nn


class SwinCVSMultiHeadModel(nn.Module):
    def __init__(self, swinv2_model, config, num_classes=3):
        super(SwinCVSMultiHeadModel, self).__init__()
        self.swinv2_model = swinv2_model
        self.lstm_hidden_size = config.MODEL.LSTM_PARAMS.HIDDEN_SIZE
        self.num_classes = num_classes
        if config.MODEL.E2E != True:
            self.multiclassifier = False
        else:
            self.multiclassifier = config.MODEL.MULTICLASSIFIER
        self.inference = config.MODEL.INFERENCE

        # LSTM (identical to original)
        self.lstm = nn.LSTM(input_size=self.swinv2_model.num_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=config.MODEL.LSTM_PARAMS.NUM_LAYERS,
                            batch_first=True)

        # Multi-head: separate Linear(256, 1) per criterion
        self.fc_c1 = nn.Linear(self.lstm_hidden_size, 1)
        self.fc_c2 = nn.Linear(self.lstm_hidden_size, 1)
        self.fc_c3 = nn.Linear(self.lstm_hidden_size, 1)

        if self.multiclassifier:
            # Multi-head for SwinV2 mid-stream classifier too
            self.fc_swin_c1 = nn.Linear(self.swinv2_model.num_features, 1)
            self.fc_swin_c2 = nn.Linear(self.swinv2_model.num_features, 1)
            self.fc_swin_c3 = nn.Linear(self.swinv2_model.num_features, 1)

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.size()
        x = x.view(-1, 3, 384, 384)
        features = self.swinv2_model.forward_features(x)

        if self.multiclassifier and not self.inference:
            swin_c1 = self.fc_swin_c1(features)
            swin_c2 = self.fc_swin_c2(features)
            swin_c3 = self.fc_swin_c3(features)
            swin_classification = torch.cat([swin_c1, swin_c2, swin_c3], dim=1)
            swin_classification = swin_classification.view(batch_size, seq_len, -1)

        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]

        lstm_c1 = self.fc_c1(last_hidden)
        lstm_c2 = self.fc_c2(last_hidden)
        lstm_c3 = self.fc_c3(last_hidden)
        lstm_classification = torch.cat([lstm_c1, lstm_c2, lstm_c3], dim=1)

        if self.multiclassifier and not self.inference:
            return swin_classification[:, -1, :], lstm_classification
        else:
            return lstm_classification
