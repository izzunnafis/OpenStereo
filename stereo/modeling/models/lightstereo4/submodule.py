import torch
import os
from PIL import Image
import torch.nn.functional as F
from torch import nn

class LearnableVarianceFilter(torch.nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2

        self.kernel = torch.nn.Parameter(
            torch.ones((3, 1, window_size, window_size))/window_size**2
        )

        self.scale = torch.nn.Parameter(
            torch.tensor(4.0)
        )
        self.shift = torch.nn.Parameter(
            torch.tensor(-2.0)
        )


    def forward(self, image):
        B, C, H, W = image.shape
        local_mean = F.conv2d(image, self.kernel, padding=self.padding, groups=C)
        local_squared_mean = F.conv2d(image ** 2, self.kernel, padding=self.padding, groups=C)
        local_variance = local_squared_mean - local_mean ** 2

        v_min = local_variance.amin(dim=(2, 3), keepdim=True)
        v_max = local_variance.amax(dim=(2, 3), keepdim=True)
        local_variance = (local_variance - v_min) / (v_max - v_min + 1e-8) * self.scale + self.shift
        
        return torch.sigmoid(local_variance)  # Apply sigmoid to normalize the output


class LearnableCorrelationVolume(nn.Module):
    def __init__(self, in_channels, max_disp):
        super(LearnableCorrelationVolume, self).__init__()
        self.max_disp = max_disp
        self.weight = nn.Conv3d(1, 1, kernel_size=(in_channels, 1, 1), bias=False, groups=1)

        # Initialize weights like uniform mean
        nn.init.constant_(self.weight.weight, 1.0 / in_channels)

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()
        cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)

        for i in range(self.max_disp):
            if i > 0:
                product = left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]
                patch = product.unsqueeze(1)  # (B,1,C,H,W-i)
                cost_volume[:, i, :, i:] = self.weight(patch).squeeze(1).squeeze(1)
            else:
                product = left_feature * right_feature
                patch = product.unsqueeze(1)  # (B,1,C,H,W)
                cost_volume[:, i, :, :] = self.weight(patch).squeeze(1).squeeze(1)

        return cost_volume.contiguous()
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

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
                        (edge_magnitude.amax(dim=(2,3), keepdim=True) - edge_magnitude.amin(dim=(2,3), keepdim=True) + 1)
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
