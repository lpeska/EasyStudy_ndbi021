"""Model training for FastCompare study initialization. CPU-friendly, no separate framework."""
import numpy as np
from scipy.sparse import csr_matrix

def train_embeddings(loader, factors=64, epochs=5, batch_size=256, sae_dim=256,
                     sae_k=16, sae_epochs=25, variant='abs', seed=42):
    """Fit ELSA via elsarec and SAE via Compresso; return aligned item codes.

    ELSA is fitted with the upstream implementation to avoid an unverified
    approximation of its published objective. Only NumPy embeddings are cached.
    """
    import torch
    from elsa import ELSA
    from compresso import TopKSAEConfig, TopKSAETrainer
    df = loader.ratings_df
    n_items = len(loader.items_df)
    users, user_ids = np.unique(df.user.to_numpy(), return_inverse=True)
    item_indices = df.item.to_numpy(dtype=int)
    x = csr_matrix((np.ones(len(df), dtype=np.float32), (user_ids, item_indices)),
                   shape=(len(users), n_items), dtype=np.float32)
    x.data[:] = 1
    torch.manual_seed(seed)
    model = ELSA(n_items=n_items, device=torch.device('cpu'), n_dims=factors)
    model.fit(x, batch_size=batch_size, epochs=epochs)
    dense = torch.nn.functional.normalize(model.get_items_embeddings().detach(), dim=-1).cpu().numpy().astype(np.float32)
    if variant not in ('abs', 'topk'):
        raise ValueError('variant must be abs or topk')
    config_kwargs = dict(hidden_dim=sae_dim, k=sae_k, batch_size=batch_size, epochs=sae_epochs)
    if variant == 'abs':
        config_kwargs['sparsify_score_mode'] = 'abs'
    # For regular TopK use the Compresso package's default activation policy.
    config = TopKSAEConfig(**config_kwargs)
    trainer = TopKSAETrainer(config)
    srp = trainer.fit_transform(dense)
    codes = srp.to_dense()

    if hasattr(codes, "detach"):
        codes = codes.detach().cpu().numpy()

    codes = np.asarray(codes, dtype=np.float32)
    if codes.shape != (n_items, sae_dim):
        raise ValueError(f'Unexpected SAE shape: {codes.shape}')
    popularity = np.bincount(item_indices, minlength=n_items).astype(np.float32)
    return codes, popularity
