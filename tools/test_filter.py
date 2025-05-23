import torch
import os
from PIL import Image
import torch.nn.functional as F

def local_variance_filter_torch(image, window_size):
    """
    Apply a local variance filter to an image using PyTorch.

    Parameters:
        image (torch.Tensor): Input grayscale image (H, W).
        window_size (int): Size of the sliding window.

    Returns:
        torch.Tensor: Image filtered by local variance.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    # Convert to 4D tensor (N, C, H, W) for convolution
    image = image.unsqueeze(0).unsqueeze(0)

    # Create a mean filter kernel
    kernel = torch.ones((1, 1, window_size, window_size), device=image.device) / (window_size ** 2)

    # Compute local mean
    local_mean = F.conv2d(image, kernel, padding=window_size // 2)

    # Compute local squared mean
    local_squared_mean = F.conv2d(image ** 2, kernel, padding=window_size // 2)

    # Compute local variance
    local_variance = local_squared_mean - local_mean ** 2

    # Normalize the local variance to the range [-2, 2]
    local_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min()) * 16 - 8

    # Print a sample of local variance at a specific patch
    patch_x, patch_y = 50, 50  # Coordinates of the patch
    patch_size = 5  # Size of the patch
    patch = local_variance[0, 0, patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    print("Sample local variance patch:")
    print(patch)

    # Apply the sigmoid function
    local_variance = torch.sigmoid(local_variance) * 255

    # Remove extra dimensions
    return local_variance.squeeze(0).squeeze(0)

# Example usage
if __name__ == "__main__":
    # Create a random grayscale image
    import torchvision.transforms as transforms

    folder = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/data/KITTI15/kitti15/testing"
    args = lambda: None  # Create a simple object to hold attributes
    args.left_img_path = os.path.join(folder, "image_2/000008_11.png")

    # Load the image using PIL
    pil_image = Image.open(args.left_img_path).convert("RGB")  # Convert to RGB

    # Convert the PIL image to a PyTorch tensor
    transform = transforms.ToTensor()
    image = transform(pil_image)  # Shape: (C, H, W)

    # Apply local variance filter to each channel
    window_size = 5
    filtered_channels = []
    for c in range(image.shape[0]):  # Iterate over channels
        filtered_channel = local_variance_filter_torch(image[c], window_size)
        filtered_channels.append(filtered_channel)

    # Stack the filtered channels back together
    filtered_image = torch.stack(filtered_channels)

    # Convert the filtered image back to a PIL image
    filtered_pil_image = transforms.ToPILImage()(filtered_image)

    

    # Save the original image
    original_output_path = "original_image.png"
    pil_image.save(original_output_path)
    print(f"Original image saved to {original_output_path}")

    # Save the filtered image
    output_path = "filtered_image.png"
    filtered_pil_image.save(output_path)
    print(f"Filtered image saved to {output_path}")
