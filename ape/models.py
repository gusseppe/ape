import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out += self.shortcut(residual)
        return self.relu(out)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, \
            "Embedding size needs to be divisible by number of heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        N, seq_len, embed_size = x.shape

        assert embed_size == self.embed_size, \
            f"Expected embedding size {self.embed_size}, but got {embed_size}"

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, seq_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_len, self.embed_size)

        return self.fc_out(out).squeeze(1)


def create_model(architecture_name, n_inputs=768, n_outputs=2, hidden_layer_size=256):
    """Create a neural network model for the given architecture name.

    Args:
        architecture_name: One of 'smlp', 'residual', 'attention'
        n_inputs: Input feature dimension (default: 768 for CLIP ViT-L/14@336px)
        n_outputs: Number of output classes (default: 2)
        hidden_layer_size: Hidden layer width (default: 256)
    """
    if architecture_name == "smlp":
        return nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, n_outputs)
        )
    elif architecture_name == "residual":
        return nn.Sequential(
            ResidualBlock(n_inputs, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size),
            ResidualBlock(hidden_layer_size, hidden_layer_size // 2),
            nn.Linear(hidden_layer_size // 2, n_outputs)
        )
    elif architecture_name == "attention":
        return nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_size),
            nn.ReLU(),
            SelfAttention(hidden_layer_size, 4),
            nn.Linear(hidden_layer_size, n_outputs)
        )
    else:
        raise ValueError(f"Unknown architecture name: {architecture_name}")
