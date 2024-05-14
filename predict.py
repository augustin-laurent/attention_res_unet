import numpy as np
from sklearn.metrics import f1_score
from scipy.spatial import distance
from torchvision import transforms
from PIL import Image
from predict import predict_img, R2AttU_Net
import torch

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)

def evaluate_model(net, device, images, masks, threshold=0.5):
    net.eval()
    f1_scores = []
    dice_scores = []
    
    for i in range(len(images)):
        img = Image.open(images[i])
        true_mask = np.array(Image.open(masks[i]))

        pred_mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=0.512,
                                out_threshold=threshold,
                                device=device)
        pred_mask = (pred_mask > threshold).astype(np.uint8)

        f1 = f1_score(true_mask.flatten(), pred_mask.flatten())
        dice = dice_score(true_mask, pred_mask)

        f1_scores.append(f1)
        dice_scores.append(dice)

    return f1_scores, dice_scores

# Usage:
net = R2AttU_Net(img_ch=4, output_ch=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
state_dict = torch.load("checkpoint/model_12.pth", map_location=device)
net.load_state_dict(state_dict)
f1_scores, dice_scores = evaluate_model(net, device, image_files, mask_files)
print('Average F1 score:', np.mean(f1_scores))
print('Average Dice score:', np.mean(dice_scores))