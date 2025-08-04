import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    """
    Bi-LSTM classifier for pelvic-floor-relevant exercise windows.

    Input  : (B, T, 34)   - B batchsize of T time-steps of 17 joints x 2 (x,y) coords
    Output : (B, 4)       - logits for 4 exercise classes
    """
    def __init__(self,
        input_size: int = 34, # 17 x 2, 2 for 2 dimensions
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, num_classes),
        )
    
    def forward(self, x):
        feats, _ = self.lstm(x) # -> (B, T, 2H)
        feats = feats[:, -1] 
        return self.head(feats)

