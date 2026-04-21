"""
Embedded Historical Retrieval Module
Paper Section 3.2.2, Equations 11-15:
  - Maintains a database of historical flight embeddings
  - Retrieves top-k similar records via cosine similarity
  - Fuses retrieved embeddings with current flight embedding

Key: This module is differentiable (Section 3.2.2 para on gradient flow).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoricalRetrievalModule(nn.Module):
    """
    Retrieves and fuses historically similar flight embeddings.
    
    D_history = {h1, h2, ..., hM} (Eq. 11)
    similarity = cosine(h_current, hi) (Eq. 12)
    μi = softmax(similarity) (Eq. 13)
    h_retrieved = Σ μi · h_similar,i (Eq. 14)
    h_f = α·h_current + (1-α)·h_retrieved (Eq. 15)
    """

    def __init__(
        self,
        embedding_dim: int,
        db_size: int = 50000,
        top_k: int = 5,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.db_size = db_size
        self.top_k = top_k

        # α is learnable (initialized to provided value)
        self.alpha = nn.Parameter(torch.tensor(alpha))

        # Historical database (stored as buffer — not a parameter)
        self.register_buffer(
            "history_db",
            torch.randn(db_size, embedding_dim) * 0.01,
        )
        self.register_buffer("db_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("db_filled", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def preseed_from_extreme_cases(self, encoder, dataloader, device, max_batches: int = 50):
        """
        Pre-seed the historical database with embeddings from extreme delay flights.
        Paper Section 3.2.2: "the historical database additionally incorporates
        extreme-weather cases observed over the past five years."
        
        Call this BEFORE training starts.
        """
        encoder.eval()
        n_seeded = 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            dynamic = batch["dynamic"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"]

            # Only seed extreme delay cases (>180 min)
            extreme_mask = targets > 180
            if extreme_mask.sum() == 0:
                continue

            sub_dynamic = dynamic[extreme_mask]
            sub_mask = mask[extreme_mask]
            h_dynamic, _ = encoder(sub_dynamic, sub_mask)

            # FIX: store h_current (last valid position) not h_global (mean pool).
            # forward() is always called with h_current, so the pre-seeded embeddings
            # must live in the same subspace or cosine similarity is meaningless.
            lengths = sub_mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0)
            batch_idx = torch.arange(h_dynamic.size(0), device=h_dynamic.device)
            h_current = h_dynamic[batch_idx, lengths]  # (B, D)

            self.update_database(h_current)
            n_seeded += extreme_mask.sum().item()

        print(f"  Pre-seeded historical DB with {n_seeded} extreme-delay embeddings "
              f"(filled: {self.db_filled.item()}/{self.db_size})")
        encoder.train()

    @torch.no_grad()
    def update_database(self, embeddings: torch.Tensor):
        """
        Add new embeddings to the historical database.
        Called after each day's predictions (paper: "After each day's predictions
        are completed, the learned dynamic representations are stored and indexed").
        """
        batch_size = embeddings.shape[0]
        ptr = self.db_ptr.item()

        if ptr + batch_size <= self.db_size:
            self.history_db[ptr:ptr + batch_size] = embeddings.detach()
            self.db_ptr += batch_size
        else:
            # Wrap around (circular buffer)
            remaining = self.db_size - ptr
            self.history_db[ptr:] = embeddings[:remaining].detach()
            overflow = batch_size - remaining
            if overflow > 0:
                self.history_db[:overflow] = embeddings[remaining:remaining + overflow].detach()
            self.db_ptr = torch.tensor(overflow % self.db_size, dtype=torch.long,
                                       device=self.db_ptr.device)

        self.db_filled = torch.clamp(self.db_filled + batch_size, max=self.db_size)

    def forward(self, h_current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_current: (batch, embedding_dim) — current flight embedding

        Returns:
            h_f: (batch, embedding_dim) — fused embedding
        """
        filled = self.db_filled.item()

        if filled < self.top_k:
            # Not enough history yet — return current embedding
            return h_current

        # Use only filled portion of database
        db = self.history_db[:filled]  # (M, D)

        # Eq. 12: Cosine similarity
        # h_current: (B, D), db: (M, D)
        h_norm = F.normalize(h_current, p=2, dim=-1)  # (B, D)
        db_norm = F.normalize(db, p=2, dim=-1)  # (M, D)
        similarity = torch.mm(h_norm, db_norm.t())  # (B, M)

        # Select top-k
        top_k = min(self.top_k, filled)
        topk_sim, topk_idx = similarity.topk(top_k, dim=-1)  # (B, k)

        # Eq. 13: Softmax weights
        mu = F.softmax(topk_sim, dim=-1)  # (B, k)

        # Gather top-k embeddings
        # topk_idx: (B, k) -> gather from db: (M, D)
        topk_embeddings = db[topk_idx]  # (B, k, D)

        # Eq. 14: Weighted sum
        h_retrieved = torch.einsum("bk,bkd->bd", mu, topk_embeddings)  # (B, D)

        # Eq. 15: Fusion with learnable alpha
        alpha = torch.sigmoid(self.alpha)  # constrain to (0, 1)
        h_f = alpha * h_current + (1 - alpha) * h_retrieved

        return h_f
