import torch
import torch.nn as nn
import torch.nn.functional as F



def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def laplace_loss(x: torch.Tensor):
    x = torch.narrow(x, 1, 0, 1)
    laplace_filter = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32)
    laplace_filter = laplace_filter.unsqueeze(0).unsqueeze(0).to(x.device)
    filtered = F.conv2d(x, weight=laplace_filter, padding=1)
    loss = torch.mean(torch.abs(filtered))
    return loss

def lap_loss(x, target):
    r"""Laplacian Loss.
    Args:
        pred (Tensor): Predicted alpha matte.
        truth (Tensor): Ground truth alpha matte.
    Returns:
        Tensor: Laplacian loss.
    """
    x = torch.narrow(x, 1, 0, 1)
    target = target.float()
    target = torch.narrow(target, 0, 0, 1)
    # Laplacian filter to compute second derivatives.
    laplace_filter = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                              dtype=torch.float32,
                              device=x.device)
    laplace_filter = laplace_filter.unsqueeze(0).unsqueeze(0).to(x.device)
    # Compute second derivatives of predicted and true alphas.
    pred_d2 = F.conv2d(x, laplace_filter, padding=1)
    truth_d2 = F.conv2d(target, laplace_filter, padding=1)
    # Compute the mean square error between second derivatives.
    return torch.mean(torch.abs(pred_d2 - truth_d2))


def sobel_loss(y_true, y_pred):
    y_true = torch.narrow(y_true, 1, 0, 1)
    y_pred = y_pred.float()
    y_pred = torch.narrow(y_pred, 0, 0, 1)
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
    if y_pred.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()
    y_true_sobel_x = F.conv2d(y_true, sobel_x, padding=1)
    y_pred_sobel_x = F.conv2d(y_pred, sobel_x, padding=1)
    y_true_sobel_y = F.conv2d(y_true, sobel_y, padding=1)
    y_pred_sobel_y = F.conv2d(y_pred, sobel_y, padding=1)
    loss = (torch.abs(y_true_sobel_x - y_pred_sobel_x) + torch.abs(y_true_sobel_y - y_pred_sobel_y)).mean()
    return loss





