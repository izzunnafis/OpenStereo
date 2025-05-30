import torch
import os
from PIL import Image
import torch.nn.functional as F

def local_variance_filter(image, window_size):
    """
    Apply a local variance filter to an image using PyTorch.

    Parameters:
        image (torch.Tensor): Input grayscale image (H, W).
        window_size (int): Size of the sliding window.

    Returns:
        torch.Tensor: Image filtered by local variance.
    """

    kernel = torch.ones((3, 1, window_size, window_size), device=image.device) / (window_size ** 2)
    local_mean = F.conv2d(image, kernel, padding=window_size // 2, groups=image.shape[1])
    local_squared_mean = F.conv2d(image ** 2, kernel, padding=window_size // 2, groups=image.shape[1])
    local_variance = local_squared_mean - local_mean ** 2

    local_variance_min = local_variance.amin(dim=(2,3), keepdim=True)
    local_variance_max = local_variance.amax(dim=(2,3), keepdim=True)
    # local_variance = (local_variance - local_variance_min) / (local_variance_max - local_variance_min) * 16 - 8
    local_variance = (local_variance - local_variance_min) / (local_variance_max - local_variance_min) * 4 - 2

    local_variance = torch.sigmoid(local_variance)# * 255

    return local_variance

def edge_detection(image, kernel_type='sobel'):
    """
    Apply edge detection to an image using PyTorch.

    Parameters:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        kernel_type (str): Type of edge detection kernel ('sobel' or 'prewitt').

    Returns:
        torch.Tensor: Edge-detected image tensor of shape (B, C, H, W).
    """
    if kernel_type == 'sobel':
        kernel_x = torch.tensor([[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]], dtype=image.dtype, device=image.device)
        kernel_y = torch.tensor([[[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]], dtype=image.dtype, device=image.device)
    elif kernel_type == 'prewitt':
        kernel_x = torch.tensor([[[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]]], dtype=image.dtype, device=image.device)
        kernel_y = torch.tensor([[[-1, -1, -1],
                                    [ 0,  0,  0],
                                    [ 1,  1,  1]]], dtype=image.dtype, device=image.device)
    else:
        raise ValueError("Unsupported kernel_type. Use 'sobel' or 'prewitt'.")

    # Expand kernels to match input channels
    channels = image.shape[1]
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)

    edge_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
    edge_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
    edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    # Optionally normalize to [0, 1]
    edge_magnitude = (edge_magnitude - edge_magnitude.amin(dim=(2,3), keepdim=True)) / \
                        (edge_magnitude.amax(dim=(2,3), keepdim=True) - edge_magnitude.amin(dim=(2,3), keepdim=True) + 1e-8)
    return edge_magnitude

# Example usage
if __name__ == "__main__":
    # Create a random grayscale image
    import torchvision.transforms as transforms

    folder = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/data/KITTI15/kitti15/testing"
    args = lambda: None  # Create a simple object to hold attributes
    args.left_img_path = os.path.join(folder, "image_2/000008_11.png")

    # Load the image using PIL
    pil_image = Image.open(args.left_img_path)  # Convert to RGB

    # Convert the PIL image to a PyTorch tensor
    transform = transforms.ToTensor()
    image = transform(pil_image)  # Shape: (C, H, W)

    # Apply local variance filter to each channel
    image = image.unsqueeze(0)  # Add batch dimension

    window_size = 5
    filtered_image = local_variance_filter(image, window_size)


    filtered_image = filtered_image.squeeze(0)  # Remove batch dimension
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
