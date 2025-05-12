import numpy as np
import torch

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = torch.nn.Embedding(max_len, embedding_dim)

    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=seq_len.device).unsqueeze(0)  # (1, seq_len)
        return self.position_embeddings(positions)

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SelfAttentionLayer, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads, dropout=dropout_rate)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_output

class FeedForwardLayer(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FeedForwardLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x.transpose(-1, -2))))))  # (B, C, L)
        return x.transpose(-1, -2)


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, hidden_units, num_blocks, num_heads, maxlen, dropout_rate, device):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.device = device

        # Embedding layers
        self.item_emb = torch.nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = PositionalEmbedding(maxlen, hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        # Layers for SASRec
        self.attn_layers = torch.nn.ModuleList([SelfAttentionLayer(hidden_units, num_heads, dropout_rate) for _ in range(num_blocks)])
        self.ffn_layers = torch.nn.ModuleList([FeedForwardLayer(hidden_units, dropout_rate) for _ in range(num_blocks)])
        self.layer_norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)])
        self.final_layer_norm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        # Prediction layer (logits calculation)
        self.predict_layer = torch.nn.Linear(hidden_units, item_num)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # Get item embeddings and add positional embeddings
        seq_embeds = self.item_emb(log_seqs)  # (B, T, C)
        pos_embeds = self.pos_emb(log_seqs.size(1))  # (T, C)
        seq_embeds += pos_embeds

        seq_embeds = self.emb_dropout(seq_embeds)  # Apply dropout

        # Attention and Feedforward blocks
        for i in range(len(self.attn_layers)):
            attn_out = self.attn_layers[i](seq_embeds.transpose(0, 1))  # (T, B, C)
            seq_embeds = self.layer_norms[i](attn_out + seq_embeds)

            ff_out = self.ffn_layers[i](seq_embeds)
            seq_embeds = self.layer_norms[i](ff_out + seq_embeds)

        # Final layer normalization
        seq_embeds = self.final_layer_norm(seq_embeds)

        # Prediction logits for positive and negative samples
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (seq_embeds * pos_embs).sum(dim=-1)
        neg_logits = (seq_embeds * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        seq_embeds = self.item_emb(log_seqs)  # (B, T, C)
        pos_embeds = self.pos_emb(log_seqs.size(1))  # (T, C)
        seq_embeds += pos_embeds
        seq_embeds = self.emb_dropout(seq_embeds)

        # Forward through attention layers and FFN layers
        for i in range(len(self.attn_layers)):
            attn_out = self.attn_layers[i](seq_embeds.transpose(0, 1))
            seq_embeds = self.layer_norms[i](attn_out + seq_embeds)

            ff_out = self.ffn_layers[i](seq_embeds)
            seq_embeds = self.layer_norms[i](ff_out + seq_embeds)

        # Final prediction
        final_emb = seq_embeds[:, -1, :]
        item_embs = self.item_emb(item_indices)
        logits = item_embs.matmul(final_emb.unsqueeze(-1)).squeeze(-1)

        return logits
