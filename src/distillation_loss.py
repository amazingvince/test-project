"""
Distillation Loss for Chess Policy Learning.

Implements forward KL divergence loss for distilling Stockfish's
policy into an LLM, with support for soft targets and hard labels.

Supports optional Liger Kernel integration for memory-efficient
KL divergence and JSD (Jensen-Shannon Divergence) computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Try to import Liger Kernel for optimized loss computation
try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSD
    from liger_kernel.ops.jsd import LigerJSD
    from liger_kernel.ops.kl_div import LigerKLDivLoss
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False


class ChessDistillationLoss(nn.Module):
    """
    Combined distillation loss for chess policy learning.

    Combines:
    1. Soft target loss (Forward KL divergence or JSD with Stockfish distribution)
    2. Hard target loss (Cross-entropy with played move)

    The soft targets teach the model the relative quality of ALL moves,
    while hard targets ground it to actual game/puzzle moves.

    Supports Liger Kernel for memory-efficient computation (up to 80% savings).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        use_liger: bool = False,
        loss_type: str = "kl",
    ):
        """
        Initialize distillation loss.

        Args:
            alpha: Weight for soft loss (1-alpha for hard loss)
                   alpha=1.0 means pure distillation (default, recommended)
                   alpha=0.5 means balanced soft + hard loss
                   alpha=0.0 means pure supervised learning
            temperature: Temperature for softening student predictions
                        (teacher probs are already soft from Stockfish)
            label_smoothing: Label smoothing for hard targets
            use_liger: Whether to use Liger Kernel for optimized loss computation
            loss_type: Type of soft loss - "kl" (Forward KL) or "jsd" (Jensen-Shannon)
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.use_liger = use_liger and LIGER_AVAILABLE
        self.loss_type = loss_type

        # Initialize soft loss function based on type and Liger availability
        if loss_type == "jsd":
            if self.use_liger:
                self.jsd_loss = LigerJSD()
            # Fallback JSD is implemented as a method
        else:  # Default: KL divergence
            if self.use_liger:
                self.kl_loss = LigerKLDivLoss(reduction='batchmean')
            else:
                self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Cross-entropy for hard targets
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='mean'
        )

    def _compute_jsd_fallback(
        self,
        student_log_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon Divergence without Liger Kernel.

        JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)
        """
        student_probs = student_log_probs.exp()
        m = 0.5 * (student_probs + teacher_probs)
        m_log = m.clamp(min=1e-10).log()

        kl_student = F.kl_div(m_log, student_probs, reduction='batchmean')
        kl_teacher = F.kl_div(m_log, teacher_probs, reduction='batchmean')

        return 0.5 * (kl_student + kl_teacher)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Raw logits from student model [batch, vocab_size]
            teacher_probs: Soft probability distribution from Stockfish [batch, vocab_size]
            hard_targets: Ground truth move indices [batch] (optional)
            mask: Boolean mask for valid positions [batch] (optional)

        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
        batch_size = student_logits.size(0)

        # Apply temperature scaling to student
        if self.temperature != 1.0:
            student_logits_scaled = student_logits / self.temperature
        else:
            student_logits_scaled = student_logits

        # Compute student log probabilities
        student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)

        # Ensure teacher probs are valid (no zeros for numerical stability)
        teacher_probs_safe = teacher_probs.clamp(min=1e-10)
        teacher_probs_safe = teacher_probs_safe / teacher_probs_safe.sum(dim=-1, keepdim=True)

        # Compute soft loss based on loss type
        if self.loss_type == "jsd":
            # Jensen-Shannon Divergence
            if self.use_liger:
                # Liger JSD expects log probabilities for both inputs
                teacher_log_probs = teacher_probs_safe.clamp(min=1e-10).log()
                soft_loss = self.jsd_loss(student_log_probs, teacher_log_probs)
            else:
                soft_loss = self._compute_jsd_fallback(student_log_probs, teacher_probs_safe)
        else:
            # Forward KL divergence (default)
            # KL(P_teacher || P_student) = sum(P_teacher * log(P_teacher / P_student))
            # PyTorch KLDivLoss expects log_probs for input, probs for target
            soft_loss = self.kl_loss(student_log_probs, teacher_probs_safe)

        # Scale by temperature^2 (standard in distillation)
        if self.temperature != 1.0:
            soft_loss = soft_loss * (self.temperature ** 2)

        # Compute hard loss if targets provided
        if hard_targets is not None and self.alpha < 1.0:
            hard_loss = self.ce_loss(student_logits, hard_targets)
        else:
            hard_loss = torch.tensor(0.0, device=student_logits.device)

        # Combine losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item() if hard_targets is not None else 0.0,
            'alpha': self.alpha,
            'loss_type': self.loss_type,
            'using_liger': self.use_liger,
        }

        return total_loss, loss_dict


class SequenceDistillationLoss(nn.Module):
    """
    Distillation loss for sequence-level training (full response generation).
    
    This is used when the student generates a full response including
    thinking and the move, and we want to distill knowledge at the
    move token positions.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 1.0,
        move_token_weight: float = 2.0,
    ):
        """
        Initialize sequence distillation loss.
        
        Args:
            alpha: Weight for soft loss
            temperature: Temperature for softening
            move_token_weight: Extra weight for move token positions
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.move_token_weight = move_token_weight
    
    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_probs_at_move: Optional[torch.Tensor] = None,
        move_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute sequence-level distillation loss.
        
        Args:
            student_logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len] with -100 for ignored positions
            teacher_probs_at_move: [batch, vocab_size] soft targets for move
            move_positions: [batch] position of move token in sequence
        
        Returns:
            Tuple of (loss, loss_dict)
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        device = student_logits.device
        
        # Standard cross-entropy loss on full sequence
        # Flatten for cross_entropy
        logits_flat = student_logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        ce_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction='mean'
        )
        
        # If no teacher probs, just return CE loss
        if teacher_probs_at_move is None or move_positions is None:
            return ce_loss, {'total_loss': ce_loss.item(), 'ce_loss': ce_loss.item()}
        
        # Extract logits at move positions for distillation
        # move_positions: [batch] indices into seq_len
        batch_indices = torch.arange(batch_size, device=device)
        move_logits = student_logits[batch_indices, move_positions]  # [batch, vocab_size]
        
        # Soft loss at move positions
        move_log_probs = F.log_softmax(move_logits / self.temperature, dim=-1)
        teacher_probs_safe = teacher_probs_at_move.clamp(min=1e-10)
        teacher_probs_safe = teacher_probs_safe / teacher_probs_safe.sum(dim=-1, keepdim=True)
        
        soft_loss = F.kl_div(
            move_log_probs,
            teacher_probs_safe,
            reduction='batchmean'
        )
        
        if self.temperature != 1.0:
            soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combine
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * soft_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'soft_loss': soft_loss.item(),
        }
        
        return total_loss, loss_dict


def create_soft_target_tensor(
    move_probs: Dict[str, float],
    vocab_size: int,
    move_to_token_id: Dict[str, int],
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert move probability dict to tensor for distillation loss.
    
    Args:
        move_probs: Dict mapping UCI moves to probabilities
        vocab_size: Size of model vocabulary
        move_to_token_id: Mapping from UCI moves to token IDs
        device: Target device
    
    Returns:
        Tensor of shape [vocab_size] with probabilities
    """
    probs = torch.zeros(vocab_size, device=device)
    
    for move_uci, prob in move_probs.items():
        if move_uci in move_to_token_id:
            token_id = move_to_token_id[move_uci]
            if 0 <= token_id < vocab_size:
                probs[token_id] = prob
    
    # Normalize
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        # Fallback to uniform if no valid moves found
        probs = torch.ones(vocab_size, device=device) / vocab_size
    
    return probs


def compute_distillation_metrics(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    hard_targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute metrics for monitoring distillation quality.
    
    Args:
        student_logits: [batch, vocab_size]
        teacher_probs: [batch, vocab_size]
        hard_targets: [batch] ground truth indices
    
    Returns:
        Dict with metrics
    """
    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=-1)
        
        # Agreement with teacher's top-1
        teacher_top1 = teacher_probs.argmax(dim=-1)
        student_top1 = student_probs.argmax(dim=-1)
        top1_agreement = (teacher_top1 == student_top1).float().mean().item()
        
        # Student probability mass on teacher's top-3
        teacher_top3 = teacher_probs.topk(3, dim=-1).indices
        student_mass_on_top3 = student_probs.gather(1, teacher_top3).sum(dim=-1).mean().item()
        
        # KL divergence
        kl_div = F.kl_div(
            student_probs.log().clamp(min=-100),
            teacher_probs,
            reduction='batchmean'
        ).item()
        
        metrics = {
            'top1_agreement': top1_agreement,
            'student_mass_on_teacher_top3': student_mass_on_top3,
            'kl_divergence': kl_div,
        }
        
        # If hard targets provided
        if hard_targets is not None:
            hard_accuracy = (student_top1 == hard_targets).float().mean().item()
            metrics['hard_accuracy'] = hard_accuracy
            
            # Probability assigned to correct move
            correct_probs = student_probs.gather(1, hard_targets.unsqueeze(1)).squeeze(1)
            metrics['prob_on_correct'] = correct_probs.mean().item()
        
        return metrics


class AdaptiveDistillationLoss(nn.Module):
    """
    Adaptive distillation loss that adjusts alpha based on move quality.
    
    For positions where the target move is the best move, we emphasize
    distillation. For positions where the target is suboptimal, we
    reduce distillation weight to avoid pushing the model toward
    the suboptimal move.
    """
    
    def __init__(
        self,
        base_alpha: float = 0.5,
        alpha_for_best: float = 0.7,
        alpha_for_bad: float = 0.2,
        cp_loss_threshold: int = 50,
        temperature: float = 1.0,
    ):
        """
        Initialize adaptive distillation loss.
        
        Args:
            base_alpha: Default alpha
            alpha_for_best: Alpha when target is best move
            alpha_for_bad: Alpha when target is a bad move
            cp_loss_threshold: CP loss above which move is "bad"
            temperature: Softmax temperature
        """
        super().__init__()
        self.base_alpha = base_alpha
        self.alpha_for_best = alpha_for_best
        self.alpha_for_bad = alpha_for_bad
        self.cp_loss_threshold = cp_loss_threshold
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        hard_targets: torch.Tensor,
        target_cp_losses: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute adaptive loss.
        
        Args:
            student_logits: [batch, vocab_size]
            teacher_probs: [batch, vocab_size]
            hard_targets: [batch]
            target_cp_losses: [batch] CP loss of each target move
        
        Returns:
            Tuple of (loss, loss_dict)
        """
        batch_size = student_logits.size(0)
        device = student_logits.device
        
        # Compute per-sample alpha based on move quality
        alphas = torch.full((batch_size,), self.base_alpha, device=device)
        
        # Best moves get high alpha (emphasize distillation)
        best_mask = target_cp_losses == 0
        alphas[best_mask] = self.alpha_for_best
        
        # Bad moves get low alpha (reduce distillation)
        bad_mask = target_cp_losses > self.cp_loss_threshold
        alphas[bad_mask] = self.alpha_for_bad
        
        # Compute losses per sample
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs_safe = teacher_probs.clamp(min=1e-10)
        teacher_probs_safe = teacher_probs_safe / teacher_probs_safe.sum(dim=-1, keepdim=True)
        
        # Per-sample KL divergence
        soft_losses = F.kl_div(
            student_log_probs,
            teacher_probs_safe,
            reduction='none'
        ).sum(dim=-1)  # [batch]
        
        if self.temperature != 1.0:
            soft_losses = soft_losses * (self.temperature ** 2)
        
        # Per-sample CE loss
        hard_losses = F.cross_entropy(
            student_logits,
            hard_targets,
            reduction='none'
        )  # [batch]
        
        # Combine with per-sample alpha
        total_losses = alphas * soft_losses + (1 - alphas) * hard_losses
        total_loss = total_losses.mean()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'soft_loss': soft_losses.mean().item(),
            'hard_loss': hard_losses.mean().item(),
            'mean_alpha': alphas.mean().item(),
            'best_move_ratio': best_mask.float().mean().item(),
            'bad_move_ratio': bad_mask.float().mean().item(),
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the loss functions
    print("Testing distillation losses...")
    print(f"Liger Kernel available: {LIGER_AVAILABLE}")

    batch_size = 4
    vocab_size = 1000

    # Create dummy data
    student_logits = torch.randn(batch_size, vocab_size)
    teacher_probs = F.softmax(torch.randn(batch_size, vocab_size) * 2, dim=-1)
    hard_targets = torch.randint(0, vocab_size, (batch_size,))

    # Test basic KL loss (PyTorch)
    print("\n=== ChessDistillationLoss (KL, PyTorch) ===")
    loss_fn = ChessDistillationLoss(alpha=0.5, temperature=1.0, use_liger=False, loss_type="kl")
    loss, loss_dict = loss_fn(student_logits, teacher_probs, hard_targets)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    # Test JSD loss (fallback)
    print("\n=== ChessDistillationLoss (JSD, fallback) ===")
    loss_fn_jsd = ChessDistillationLoss(alpha=0.5, temperature=1.0, use_liger=False, loss_type="jsd")
    loss_jsd, loss_dict_jsd = loss_fn_jsd(student_logits, teacher_probs, hard_targets)
    print(f"Total loss: {loss_jsd.item():.4f}")
    print(f"Loss dict: {loss_dict_jsd}")

    # Test Liger KL if available
    if LIGER_AVAILABLE:
        print("\n=== ChessDistillationLoss (KL, Liger) ===")
        loss_fn_liger = ChessDistillationLoss(alpha=0.5, temperature=1.0, use_liger=True, loss_type="kl")
        loss_liger, loss_dict_liger = loss_fn_liger(student_logits, teacher_probs, hard_targets)
        print(f"Total loss: {loss_liger.item():.4f}")
        print(f"Loss dict: {loss_dict_liger}")

        print("\n=== ChessDistillationLoss (JSD, Liger) ===")
        loss_fn_liger_jsd = ChessDistillationLoss(alpha=0.5, temperature=1.0, use_liger=True, loss_type="jsd")
        loss_liger_jsd, loss_dict_liger_jsd = loss_fn_liger_jsd(student_logits, teacher_probs, hard_targets)
        print(f"Total loss: {loss_liger_jsd.item():.4f}")
        print(f"Loss dict: {loss_dict_liger_jsd}")

    # Test with different alphas
    print("\n=== Alpha sweep ===")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        loss_fn = ChessDistillationLoss(alpha=alpha)
        loss, _ = loss_fn(student_logits, teacher_probs, hard_targets)
        print(f"Alpha={alpha}: loss={loss.item():.4f}")

    # Test metrics
    print("\n=== Distillation Metrics ===")
    metrics = compute_distillation_metrics(student_logits, teacher_probs, hard_targets)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test adaptive loss
    print("\n=== AdaptiveDistillationLoss ===")
    adaptive_loss = AdaptiveDistillationLoss()
    cp_losses = torch.tensor([0, 20, 100, 300], dtype=torch.float)
    loss, loss_dict = adaptive_loss(student_logits, teacher_probs, hard_targets, cp_losses)
    print(f"Adaptive loss dict: {loss_dict}")
