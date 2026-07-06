"""Phase 3: subgraph-level retrieval (#9 K-RagRec, arXiv:2501.02226).

SubgraphProjector maps a pooled l-hop subgraph representation (produced by
extract_subgraph_embeds.py, using the Phase 1/2 R-GCN encoder) into an LLM's
hidden size, to be injected as a single soft-prompt token -- replacing
G-Refer's node/path text serialization with a continuous embedding, per the
paper's "GNN encoder + MLP projector as soft prompt" idea.

This module only defines the projector (trained jointly with LoRA during
SFT); subgraph extraction and pooling live in extract_subgraph_embeds.py so
they can be precomputed once and cached, since re-running k-hop extraction +
GNN encoding per training step would be far slower than the LLM forward pass
itself.
"""
import torch
import torch.nn as nn


class SubgraphProjector(nn.Module):
    def __init__(self, in_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, subgraph_embed):
        return self.net(subgraph_embed)


def prepend_soft_prompt(input_embeds, attention_mask, labels, soft_embeds):
    """Prepend one soft-prompt token (from `soft_embeds`, [B, H]) to a batch
    of input embeddings, extending attention_mask (unmasked) and labels
    (ignored, -100) by one position accordingly.
    """
    soft_embeds = soft_embeds.unsqueeze(1).to(input_embeds.dtype)
    input_embeds = torch.cat([soft_embeds, input_embeds], dim=1)

    pad = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device,
                     dtype=attention_mask.dtype)
    attention_mask = torch.cat([pad, attention_mask], dim=1)

    if labels is not None:
        ignore = torch.full((labels.shape[0], 1), -100,
                            device=labels.device, dtype=labels.dtype)
        labels = torch.cat([ignore, labels], dim=1)

    return input_embeds, attention_mask, labels
