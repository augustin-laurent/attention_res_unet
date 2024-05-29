import torch

from tqdm import tqdm

from loss_functions import dice_coeff

@torch.inference_mode()
def evaluate(model, val_loader, device, amp):
    model.eval()
    num_val_batches = len(val_loader)
    dice_score = 0

    with torch.autocast(device.type if device.type != "cuda" else "cpu", enabled=amp):
        for batch in tqdm(val_loader, total=num_val_batches, desc="Validation", unit="batch", leave=False):
            images, true_masks = batch["image"], batch["mask"]
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            masks_pred = model(images)
            
            assert true_masks.min() >= 0 and true_masks.max() <= 1, "For binary segmentation masks, the values should be 0 or 1."
            masks_pred = (torch.nn.functional.sigmoid(masks_pred.squeeze(1)) >= 0.5).float()
            print(masks_pred.shape, true_masks.shape)
            dice_score += dice_coeff(masks_pred, true_masks, reduce_batch_first=False)
            
    model.train()
    return dice_score / max(num_val_batches, 1)