"""
Distillation losses and metrics used by policy distillation training.

The main entrypoint is `ChessDistillationLoss`, which computes a forward KL
divergence between the student logits and a teacher probability distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Try to import Liger Kernel for optimized loss computation
try:
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Args:
            student_logits: Raw logits from student model [batch, vocab_size]
            teacher_probs: Soft probability distribution from Stockfish [batch, vocab_size]
            hard_targets: Ground truth move indices [batch] (optional).

        Returns:
            Tuple of (total_loss, loss_dict with components)
        """
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
