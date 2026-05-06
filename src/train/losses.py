import torch
import torch.nn.functional as F


def slide_classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def attention_entropy_regularizer(attention: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    entropy = -torch.sum(attention * torch.log(attention + eps))
    return entropy


def top_bottom_margin_loss(instance_prob: torch.Tensor, attention: torch.Tensor, target_class: int) -> torch.Tensor:
    if instance_prob.size(0) < 4:
        return torch.tensor(0.0, device=instance_prob.device)
    order = torch.argsort(attention, descending=True)
    top_idx = order[: max(1, int(0.05 * len(order)))]
    bottom_idx = order[-max(1, int(0.05 * len(order))):]
    top_score = instance_prob[top_idx, target_class].mean()
    bottom_score = instance_prob[bottom_idx, target_class].mean()
    return torch.relu(0.2 - (top_score - bottom_score))
