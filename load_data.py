import torch

import numpy as np

from PIL import Image
from os import listdir
from os.path import splitext, isfile, join

from torch.utils.data import Dataset

import logging

def load_image(filename: str) -> Image:
    """
    This function loads an image from a file.

    Parameters:
        filename (str): The path to the image file. The file extension determines how the image is loaded.

    Returns:
        PIL.Image: The loaded image.

    The function supports three types of file extensions:
    - '.npy': The image data is loaded using numpy's `load` function and then converted to a PIL Image.
    - '.pt' or '.pth': The image data is loaded using PyTorch's `load` function, converted to a numpy array, and then converted to a PIL Image.
    - Any other extension: The image is opened directly using PIL's `open` function and converted to RGB format. This is because the images in the dataset are in RGB format, not RGBA, and only 3 channels are needed as all values in the 4th channel are 255.
    """
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        # Image on my dataset are in RGB format not RGBA so 3 channels is only needed since all values in the 4th channel are 255
        return Image.open(filename).convert("RGB")

class SegDataset(Dataset):
    def __init__(self, images_path, masks_path, scale=1.0):
        self.images_path = images_path
        self.masks_path = masks_path
        assert 0 < scale <= 1, "Scale must be between 0 and 1"
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(images_path) if isfile(join(images_path, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f"No input file found in {images_path}, make sure you put your images there")
        logging.info(f"Creating dataset with {len(self.ids)} examples")
    
    def __len__(self) -> int:
        """
        This method returns the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.

        The method works by returning the length of the `ids` list, which contains the IDs of all items in the dataset.
        """
        return len(self.ids)
    
    @staticmethod
    def preprocess_image(img: Image, scale: float, is_mask: bool) -> np.ndarray:
        """
        This function preprocesses an image by resizing it and normalizing its pixel values.

        Parameters:
            img (PIL.Image): The image to preprocess.
            scale (float): The scale factor to resize the image. Must be non-zero.
            is_mask (bool): A boolean indicating whether the image is a mask or not.

        Returns:
            numpy.ndarray: The preprocessed image as a numpy array.

            If `is_mask` is True, the image is resized using nearest neighbor sampling (NEAREST) and returned as a numpy array.

            If `is_mask` is False, the image is resized using bicubic interpolation (BICUBIC), then transposed if it has more than two dimensions. If pixel values are greater than 1, they are normalized by dividing by 255.0. The image is then returned as a numpy array.

        Raises:
            AssertionError: If the scale factor is too small and the resized images would have no pixels.
        """
        width, height = img.size
        new_width, new_height = int(width * scale), int(height * scale)
        assert new_height > 0 and new_width > 0, "Scale is too small, resized images would have no pixel"
        # Nearest for mask, details is not important since it's black or white, Bicubic is good but Lanczos is better but slower
        pil_img = img.resize((new_width, new_height), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        if is_mask:
            pil_img = pil_img.convert("L")
            data = np.asarray(pil_img)
            data = data[np.newaxis, ...]
            return data

        else:
            data = np.asarray(pil_img)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            else:
                data = data.transpose((2, 0, 1))
            if (data > 1).any():
                data = data / 255.0

            return data
        
    def __getitem__(self, idx: int) -> dict:
        """
        This method retrieves an item from the dataset at a given index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the preprocessed image and mask tensors.

        The method works as follows:
        - It first retrieves the ID of the item at the given index.
        - It then finds the corresponding image and mask files in the dataset directories.
        - It asserts that exactly one image and one mask file are found for the ID.
        - It loads the image and mask files using the `load_image` function.
        - It asserts that the image and mask have the same size.
        - It preprocesses the image and mask using the `preprocess` method.
        - It returns a dictionary containing the preprocessed image and mask as PyTorch tensors.

        Raises:
            AssertionError: If no image or multiple images are found for the ID, if no mask or multiple masks are found for the ID, or if the image and mask sizes do not match.
        """
        name = self.ids[idx]
        mask_file = list(self.masks_path.glob(name + ".*"))
        img_file = list(self.images_path.glob(name + ".*"))

        assert len(img_file) == 1, f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert len(mask_file) == 1, f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, f"Image and mask {name} should be the same size, but are {img.size} and {mask.size}"

        img = self.preprocess_image(img, self.scale, is_mask=False)
        mask = self.preprocess_image(mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous()
        }
    