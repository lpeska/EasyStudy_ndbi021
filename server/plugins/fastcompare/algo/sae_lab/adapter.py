"""FastCompare adapters for SAE preference elicitation and recommendation."""
import numpy as np
from plugins.fastcompare.algo.algorithm_base import (AlgorithmBase, PreferenceElicitationBase,
                                                      Parameter, ParameterType)
from plugins.fastcompare.algo.sae_lab.core import select_groups, recommend
from plugins.fastcompare.algo.sae_lab.models import train_embeddings

class SAEChoiceElicitation(PreferenceElicitationBase):
    def __init__(self, loader, factors=64, epochs=5, sae_dim=256, sae_k=16,
                 sae_epochs=25, variant='abs', group_size=4, questions=10, **kwargs):
        self.loader = loader
        self.factors, self.epochs = int(factors), int(epochs)
        self.sae_dim, self.sae_k, self.sae_epochs = int(sae_dim), int(sae_k), int(sae_epochs)
        self.variant, self.group_size, self.questions = variant, int(group_size), int(questions)
    def fit(self):
        self.codes, self.popularity = train_embeddings(self.loader, self.factors, self.epochs,
            sae_dim=self.sae_dim, sae_k=self.sae_k, sae_epochs=self.sae_epochs, variant=self.variant)
        self.groups = select_groups(self.codes, self.popularity, self.group_size)[:self.questions]
        if not self.groups:
            raise ValueError('No signed dimensions with enough representatives. Use AbsTopK, more data, or lower group size.')
        self.loader = None
    def get_initial_data(self, movie_indices_to_ignore=None):
        return np.array([], dtype=int)
    @classmethod
    def name(cls): return 'SAE choice groups'
    @classmethod
    def parameters(cls):
        return [
            Parameter('factors', ParameterType.INT, 64, help='Number of latent dimensions in the ELSA item representations used to train the SAE.'),
            Parameter('epochs', ParameterType.INT, 5, help='Number of ELSA training epochs before fitting the SAE.'),
            Parameter('sae_dim', ParameterType.INT, 256, help='Number of SAE latent neurons/features in each movie code.'),
            Parameter('sae_k', ParameterType.INT, 16, help='Target number of active SAE features per movie (TopK/AbsTopK sparsity).'),
            Parameter('sae_epochs', ParameterType.INT, 25, help='Number of epochs used to train the sparse autoencoder.'),
            Parameter('variant', ParameterType.STRING, 'abs', help="SAE activation variant: 'abs' uses signed AbsTopK; 'topk' uses ordinary TopK. Two-sided questions require both signs."),
            Parameter('group_size', ParameterType.INT, 4, help='Number of randomly sampled movies displayed on each side. A neuron needs at least this many nonzero activations of both signs.'),
            Parameter('questions', ParameterType.INT, 10, help='Number of eligible neurons queried, ranked by the smaller of their positive/negative active movie counts.'),
        ]


def blend_profiles(codes, explicit, later_items, gamma=0.5):
    """Blend normalized explicit elicitation and later positive-item profiles."""
    z = np.asarray(codes, dtype=float)
    gamma = float(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must be between 0 and 1')
    e = np.asarray(explicit, dtype=float)
    if e.shape != (z.shape[1],):
        raise ValueError('Explicit vector dimension does not match SAE codes')
    ids = sorted({int(i) for i in later_items if 0 <= int(i) < len(z)})
    later = z[ids].mean(axis=0) if ids else np.zeros(z.shape[1])
    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.zeros_like(v)
    e, later = normalize(e), normalize(later)
    if np.any(e) and np.any(later):
        return gamma * e + (1 - gamma) * later
    return e if np.any(e) else later


class SAEProfileRecommender(AlgorithmBase):
    def __init__(self, loader, gamma=0.5, **kwargs):
        self.gamma = float(gamma)
        if not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be between 0 and 1')
        self.n_items = len(loader.items_df)
    def fit(self): pass
    def predict(self, selected_items, filter_out_items, k):
        from flask import session
        from plugins.fastcompare import get_cache_path
        from plugins.fastcompare.algo.sae_lab.adapter import SAEChoiceElicitation
        elicitation = SAEChoiceElicitation.__new__(SAEChoiceElicitation)
        elicitation.load(get_cache_path(session['user_study_guid'], SAEChoiceElicitation.name()), '')
        z = np.asarray(elicitation.codes)
        explicit = session.get('sae_explicit', [0.0] * z.shape[1])
        # selected_items is the accumulated standard FastCompare history.
        # Remove the items selected in SAE elicitation to isolate LATER feedback.
        elicitation_liked = set(map(int, session.get('sae_liked', [])))
        later_items = set(map(int, selected_items)) - elicitation_liked
        profile = blend_profiles(z, explicit, later_items, self.gamma)
        seen = set(map(int, filter_out_items)) | set(map(int, session.get('sae_displayed', [])))
        # recommend() accepts an explicit and an implicit vector; pass the
        # already blended profile as explicit and a zero implicit vector.
        return recommend(z, profile, np.zeros_like(profile), seen, k, alpha=0.0)
    @classmethod
    def name(cls): return 'SAE profile'
    @classmethod
    def parameters(cls):
        return [
            Parameter('gamma', ParameterType.FLOAT, 0.5,
                      help='Weight of explicit SAE elicitation preferences versus later liked movie selections, from 0 (later only) to 1 (elicitation only). Both profiles are normalized; if only one exists, it is used alone.'),
        ]
