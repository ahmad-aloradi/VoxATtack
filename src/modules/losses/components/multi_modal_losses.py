from typing import Any, Dict, Callable, Union, List, Tuple, Optional, Literal
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.constants import LOSS_TYPES, EMBEDS
from src import utils

log = utils.get_pylogger(__name__)


@dataclass
class LossWeights:
    """Configurable weights for different loss components"""
    ensemble: float = 1.0
    fusion: float = 1.0
    audio: float = 0.2
    text: float = 0.2
    contrastive: float = 0.1
    consistency: float = 0.1
    confidence: float = 0.1

    @classmethod
    def from_dict(cls, weights_dict: Dict[str, float]) -> 'LossWeights':
        """Create LossWeights from a dictionary"""
        return cls(**{k: v for k, v in weights_dict.items() if hasattr(cls, k)})


class MultiModalLoss(nn.Module):
    """
    Loss function for multi-modal fusion with multiple regularization terms.
    Supports different classifier architectures with adaptive loss components.
    """
    def __init__(
        self,
        classification_loss: Callable,
        classifier_name: Literal['normalized', 'robust'],
        weights: Optional[LossWeights] = None,
        confidence_target: float = 0.9,
        contrastive_temprature: float = 1.0,
        weight_scheduler: Optional[Callable] = None,
        return_dict: bool = True
    ):
        super().__init__()
        # Core configuration
        self.classification_loss = classification_loss
        self.classifier_name = classifier_name
        self.weights = weights or LossWeights()
        self.confidence_target = confidence_target
        self.weight_scheduler = weight_scheduler
        self.contrastive_temprature = contrastive_temprature
        self.return_dict = return_dict
        
        # Validation
        if classifier_name not in ['normalized', 'robust']:
            raise ValueError(f"Invalid classifier name: {classifier_name}. Expected 'normalized' or 'robust'")
            
        # Performance optimization
        self.eps = 1e-8
        self._current_epoch = 0
        self.unsqueeze = classification_loss.__class__.__name__ == 'LogSoftmaxWrapper'
        
        # Pre-compute constants
        self.embed_keys = {EMBEDS[modality].lower(): modality.lower() for modality in ["TEXT", "AUDIO", "FUSION"]}
                          
        # Define classifier-specific configurations
        self.key_configs = {
            'normalized': {
                'logits_patterns': [(r'(audio|text|fusion)_logits', lambda m: m.group(1))],
                'embedding_prefixes': list(self.embed_keys.keys()),
                'active_weights': {'audio', 'text', 'fusion', 'contrastive', 'consistency'}
            },
            'robust': {
                'logits_patterns': [ (r'(ensemble|audio|text|fusion)_logits', lambda m: m.group(1))],
                'embedding_prefixes': list(self.embed_keys.keys()) + [r'(audio|text|fusion)_features'],
                'confidence_pattern': r'(audio|text)_confidence',
                'active_weights': {'ensemble', 'audio', 'text', 'fusion', 'contrastive', 'consistency', 'confidence'}
            }
        }

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for weight scheduling."""
        self._current_epoch = epoch
        if self.weight_scheduler is not None:
            new_weights = self.weight_scheduler(epoch)
            if isinstance(new_weights, dict):
                self.weights = LossWeights.from_dict(new_weights)
            elif isinstance(new_weights, LossWeights):
                self.weights = new_weights

    def cosine_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity matrix with gradient clipping for stability."""
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        similarity_matrix = torch.clamp(similarity_matrix, min=-1.0 + self.eps, max=1.0 - self.eps)
        return similarity_matrix / self.contrastive_temprature

    def contrastive_loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss with improved stability."""
        # Early return for small batches
        if embeddings.size(0) <= 1:
            return torch.tensor(0.0, device=embeddings.device)
            
        similarity_matrix = self.cosine_similarity_matrix(embeddings)
        
        # Create masks for positive and negative pairs
        batch_size = targets.size(0)
        labels_matrix = targets.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = labels_matrix == labels_matrix.T
        positive_mask.fill_diagonal_(False)
        negative_mask = ~positive_mask
        negative_mask.fill_diagonal_(False)
        
        # Check for valid samples (with both positive and negative pairs)
        valid_samples = (positive_mask.sum(dim=1) > 0) & (negative_mask.sum(dim=1) > 0)
        if not valid_samples.any():
            return torch.tensor(0.0, device=embeddings.device)
            
        # Calculate loss using log-sum-exp for numerical stability
        pos_similarities = similarity_matrix.masked_fill(~positive_mask, -1e9)
        neg_similarities = similarity_matrix.masked_fill(~negative_mask, -1e9)
        
        pos_logits = torch.logsumexp(pos_similarities, dim=1)
        neg_logits = torch.logsumexp(neg_similarities, dim=1)
        
        # Apply loss only to valid samples
        per_sample_loss = (-pos_logits + neg_logits) * valid_samples.float()
        n_valid = valid_samples.sum().item()
        
        return per_sample_loss.sum() / max(n_valid, 1)

    def consistency_loss(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        if len(predictions) < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        # 1. Compute log-probabilities safely with log_softmax
        log_probs = [F.log_softmax(pred, dim=-1) for pred in predictions]  # (batch, num_classes)
        
        # 2. Convert to probabilities (clamped to avoid underflow)
        probs = [log_prob.exp().clamp(min=1e-8) for log_prob in log_probs]  # (batch, num_classes)
        stacked_probs = torch.stack(probs, dim=1)  # (batch, num_modalities, num_classes)
        
        # 3. Compute mean probability distribution
        mean_probs = stacked_probs.mean(dim=1, keepdim=True)  # (batch, 1, num_classes)
        mean_probs = mean_probs.clamp(min=1e-8)  # Avoid log(0)
        
        # 4. Compute KL divergence safely
        log_mean_probs = torch.log(mean_probs)  # (batch, 1, num_classes)
        
        # Expand to match stacked_probs shape
        log_mean_probs_expanded = log_mean_probs.expand_as(stacked_probs)
        
        # KL(p_i || mean_p) = sum(p_i * (log(p_i) - log(mean_p)))
        kl_divs = F.kl_div(
            input=log_mean_probs_expanded,  # log(mean_p)
            target=stacked_probs,           # p_i
            reduction='none'
        ).sum(dim=-1)  # Sum over classes
        
        return kl_divs.mean()  # Average over batch and modalities

    def confidence_loss(self, confidences: List[torch.Tensor]) -> torch.Tensor:
        """Regularize prediction confidences to prevent overconfidence."""
        if not confidences:
            return torch.tensor(0.0, device=confidences[0].device)
            
        stacked_conf = torch.cat(confidences, dim=1)
        target_conf = torch.full_like(stacked_conf, self.confidence_target)
        
        return F.binary_cross_entropy(stacked_conf, target_conf, reduction='mean')

    def _prepare_inputs(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        """Prepare inputs by handling unsqueezing."""
        # Handle target unsqueezing
        targets_unsqueezed = targets.unsqueeze(1) if self.unsqueeze and targets.dim() == 1 else targets
        
        # Handle outputs unsqueezing
        outputs_unsqueezed = {}
        for k, v in outputs.items():
            needs_unsqueeze = (
                self.unsqueeze and 
                ('logits' in k.lower() or k == EMBEDS["CLASS"]) and 
                v.dim() == 2
            )
            outputs_unsqueezed[k] = v.unsqueeze(1) if needs_unsqueeze else v
            
        return outputs_unsqueezed, targets_unsqueezed

    def _extract_keys(self, outputs: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, Dict]:
        """Extract relevant keys based on classifier type."""
        config = self.key_configs[self.classifier_name]
        
        logits_keys = {}
        embedding_keys = {}
        confidence_keys = {}
        
        # Process keys
        for key in outputs:
            key_lower = key.lower()
            
            # Extract logits keys
            for pattern, name_fn in config.get('logits_patterns', []):
                if callable(name_fn):
                    import re
                    match = re.match(pattern, key_lower)
                    if match:
                        logits_keys[name_fn(match)] = key
                elif key_lower == pattern:
                    logits_keys[name_fn] = key
            
            # Extract embedding keys
            if key_lower in self.embed_keys:
                embedding_keys[self.embed_keys[key_lower]] = key
            
            # Extract confidence keys (robust classifier only)
            if self.classifier_name == 'robust' and key_lower.endswith('_confidence'):
                modality = key_lower[:-11]  # Remove '_confidence'
                if modality in ['audio', 'text']:
                    confidence_keys[modality] = key
                    
        # Determine active weights
        active_weights = set(config['active_weights'])
        if 'consistency' in active_weights and len(logits_keys) < 2:
            active_weights.remove('consistency')
        if 'confidence' in active_weights and not confidence_keys:
            active_weights.remove('confidence')
            
        return logits_keys, embedding_keys, confidence_keys, active_weights

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss based on classifier type and outputs."""
        # Prepare inputs
        outputs_unsqueezed, targets_unsqueezed = self._prepare_inputs(outputs, targets)
        
        # Extract keys and determine active weights
        logits_keys, embedding_keys, confidence_keys, active_weights = self._extract_keys(outputs)
        
        # Initialize losses container
        losses = {}
        
        # Compute classification losses for all modalities
        for modality, key in logits_keys.items():
            weight = getattr(self.weights, modality, 0)
            if modality in active_weights and weight > 0:
                losses[f"{modality}_loss"] = self.classification_loss(
                    outputs_unsqueezed[key], 
                    targets_unsqueezed
                )
        
        # Compute contrastive loss if applicable
        if (embedding_keys and "contrastive" in active_weights and 
                getattr(self.weights, "contrastive", 0) > 0):
            embed_key = embedding_keys.get("fusion")
            if embed_key:
                losses["contrastive_loss"] = self.contrastive_loss(
                    outputs[embed_key], 
                    targets
                )
        
        # Compute consistency loss if applicable
        if (len(logits_keys) >= 2 and "consistency" in active_weights and 
                getattr(self.weights, "consistency", 0) > 0):
            # Get all logits except for ensemble_logits
            logits_list = [outputs[key] for key in logits_keys.values() if 'ensemble_logits' not in key]
            losses["consistency_loss"] = self.consistency_loss(logits_list)
        
        # Compute confidence loss if applicable
        if (confidence_keys and "confidence" in active_weights and 
                getattr(self.weights, "confidence", 0) > 0):
            confidence_list = [outputs[key] for key in confidence_keys.values()]
            losses["confidence_loss"] = self.confidence_loss(confidence_list)
        
        # Combine all losses with weights
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        for loss_name, loss_value in losses.items():
            weight_attr = loss_name.split("_")[0]
            weight = getattr(self.weights, weight_attr, 0.0)
            if weight > 0:
                total_loss = total_loss + weight * loss_value
        
        # Handle potential NaN/Inf values
        if not torch.isfinite(total_loss):
            # Try to use reliable fallback losses
            for fallback in ["main_loss", "fusion_loss"]:
                if fallback in losses and torch.isfinite(losses[fallback]):
                    total_loss = losses[fallback]
                    break
            else:
                # Last resort fallback
                total_loss = torch.tensor(1.0, device=targets.device, requires_grad=True)
        
        # Add total loss to dictionary
        losses["loss"] = total_loss
        
        return losses if self.return_dict else total_loss
