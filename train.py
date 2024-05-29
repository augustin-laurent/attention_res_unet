import logging
import os
import random
import sys
import wandb

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

from pathlib import Path
from tqdm import tqdm

from model import R2AttU_Net
from evaluate import evaluate
from loss_functions import dice_loss
from load_data import SegDataset

PATH_TO_DIR = "/users/m2ida/m2ida/dataset_segm/"

DIR_IMG = Path(PATH_TO_DIR + "train/" +"HE_cell/")
DIR_MASK = Path(PATH_TO_DIR + "train/"  +"ERG_cell/")
DIR_IMG_VAL = Path(PATH_TO_DIR + "eval/" + "HE_eval/")
DIR_MASK_VAL = Path(PATH_TO_DIR + "eval/" +"ERG_eval/")
DIR_SAVE = Path("checkpoints")

def train_model(model, device, epochs: int = 5, batch_size: int = 16, learning_rate: float = 1e-4, val_percent: float = 0.1, save_checkpoints: bool = False, img_scale: float = 0.5, amp: bool = False, weight_decay: float = 1e-8, gradiant_clipping: float = 1.0):
    try:
        dataset = SegDataset(DIR_IMG, DIR_MASK, img_scale)
    except (AssertionError, RuntimeError, FileNotFoundError) as e:
        print(e)
        return
    
    try:
        val_dataset = SegDataset(DIR_IMG_VAL, DIR_MASK_VAL, img_scale)
    except (AssertionError, RuntimeError, FileNotFoundError) as e:
        print(e)
        return

    n_train = len(dataset)
    n_val = len(val_dataset)

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count() - 2, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    tracker = wandb.init(project="Attention_Res_Unet", resume="allow", anonymous="must")
    tracker.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_percent=val_percent, img_scale=img_scale, amp=amp))

    logging.info(
        f"""Starting training:\n
        Device: {device}\n
        Epochs: {epochs}\n
        Batch size: {batch_size}\n
        Learning rate: {learning_rate}\n
        Training size: {n_train}\n
        Validation size: {n_val}\n
        Checkpoints: {save_checkpoints}\n
        Image scaling: {img_scale}\n
        Mixed precision: {amp}\n
        Weight decay: {weight_decay}\n
        Gradient clipping: {gradiant_clipping}\n
        """
    )

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

                assert images.shape[1] == model.img_ch, f"Network has been defined with {model.img_ch} input channels, but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly."

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != "cuda" or "mps" else "cpu", enabled=amp):
                    # Canal useless we can drop them here
                    masks_pred = model(images)
                    print(masks_pred.shape, true_masks.shape)
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(torch.nn.functional.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradiant_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                tracker.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]["lr"], "step": global_step, "epoch": epoch})
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                division_step = 1 if n_train % batch_size == 0 else (n_train // batch_size) + 1
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace(".", "/")
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms["Weights/" + tag] = wandb.Histogram(value.detach().cpu().numpy())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms["Gradients/" + tag] = wandb.Histogram(value.grad.detach().cpu().numpy())
                        
                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info("Validation Dice Score : {}, {}".format(val_score).format(val_score * max(len(val_loader), 1)))
                        try:
                            tracker.log({
                                "Learning rate": optimizer.param_groups[0]["lr"],
                                "Validation Dice": val_score,
                                "Images": wandb.Image(images[0].cpu()),
                                "Masks": {
                                    "True": wandb.Image(true_masks[0].cpu()),
                                    "Pred": wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                "Step": global_step,
                                "Epoch": epoch,
                                **histograms
                            })
                        except Exception as e:
                            print(e)
                            pass
                
        if save_checkpoints:
            Path(DIR_SAVE).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["mask_values"] = dataset.mask_values
            torch.save(state_dict, str(DIR_SAVE / f"model_{epoch}.pth"))
            logging.info(f"Checkpoint {epoch} saved !")

    tracker.finish(quiet=True)

if __name__ == "__main__":
    wandb.login()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logging.info(f"Using device {device}")
    model = R2AttU_Net(img_ch=3, output_ch=1)
    model = model.to(memory_format=torch.channels_last)
    logging.info(f"Network:\n"
                 f"\t{model.img_ch} input channels\n"
                 f"\t{model.output_ch} output channels (classes)\n")
    model.to(device=device)
    try:
        train_model(model= model, epochs=20, batch_size=16, learning_rate=1e-4, device=device, img_scale=0.5, val_percent=0.1, amp=True, save_checkpoints=True)
    except torch.cuda.OutOfMemoryError:
        logging.error("Detected OOM error."
                      "Enabling checkpoint to reduce memory usage, keep in mind this slow down training."
                      "Enabling AMP can also help to reduce memory usage.")
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(model=model, epochs=20, batch_size=16, learning_rate=1e-4, device=device, img_scale=0.512, val_percent=0.1, amp=False)