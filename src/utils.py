import cv2 as cv
import numpy as np
from enum import Enum
from typing import Tuple, Optional, List


class ColorSpace(Enum):
    """Supported color spaces for texture mixing."""
    RGB = "RGB"
    LAB = "LAB"
    YUV = "YUV"


class ColorSpaceConfig:
    """Configuration for how to handle each color space in mixing."""
    
    @staticmethod
    def get_config(color_space: ColorSpace) -> dict:
        """Get processing configuration for a color space."""
        configs = {
            ColorSpace.RGB: {
                'channels': 3,
                'all_channels': True,  # Process all channels together
                'channel_names': ['R', 'G', 'B'],
            },
            ColorSpace.LAB: {
                'channels': 3,
                'all_channels': False,  # Process channels separately
                'channel_names': ['L', 'a', 'b'],
                'mixing_channels': [0],  # Only mix L channel
                'preserve_channels': [1, 2],  # Preserve a,b channels
            },
            ColorSpace.YUV: {
                'channels': 3,
                'all_channels': False,
                'channel_names': ['Y', 'U', 'V'],
                'mixing_channels': [0],  # Only mix Y (luminance)
                'preserve_channels': [1, 2],  # Preserve U,V (chrominance)
            },
        }
        return configs[color_space]


def convert_to_colorspace(image: np.ndarray, target_space: ColorSpace) -> np.ndarray:
    """
    Convert RGB image to target color space.
    
    Args:
        image: RGB image as numpy array (H, W, 3), float64 [0, 1]
        target_space: Target color space
        
    Returns:
        Converted image as numpy array (H, W, 3), float64 [0, 1]
    """
    if target_space == ColorSpace.RGB:
        return image
    
    # Convert to uint8 for OpenCV conversions
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    conversions = {
        ColorSpace.LAB: cv.COLOR_RGB2LAB,
        ColorSpace.YUV: cv.COLOR_RGB2YUV,
    }
    
    result = cv.cvtColor(img_uint8, conversions[target_space])
    
    # Normalize back to float64 [0, 1]
    return result.astype(np.float64) / 255.0


def convert_from_colorspace(image: np.ndarray, source_space: ColorSpace) -> np.ndarray:
    """
    Convert from color space back to RGB.
    
    Args:
        image: Image in source color space as numpy array (H, W, 3), float64 [0, 1]
        source_space: Source color space
        
    Returns:
        RGB image as numpy array (H, W, 3), float64 [0, 1]
    """
    if source_space == ColorSpace.RGB:
        return image
    
    # Convert to uint8
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    conversions = {
        ColorSpace.LAB: cv.COLOR_LAB2RGB,
        ColorSpace.YUV: cv.COLOR_YUV2RGB,
    }
    
    result = cv.cvtColor(img_uint8, conversions[source_space])
    
    return result.astype(np.float64) / 255.0


def prepare_textures_for_mixing(
    textures: List[np.ndarray],
    color_space: ColorSpace
) -> Tuple[List[np.ndarray], dict]:
    """
    Prepare textures for mixing in specified color space.
    
    Args:
        textures: List of RGB textures as numpy arrays (H, W, 3), float64 [0, 1]
        color_space: Color space to perform mixing in
        
    Returns:
        Tuple of (converted_textures, metadata)
    """
    config = ColorSpaceConfig.get_config(color_space)
    converted_textures = []
    
    for tex in textures:
        # Convert to target color space
        converted = convert_to_colorspace(tex, color_space)
        converted_textures.append(converted)
    
    metadata = {
        'color_space': color_space,
        'config': config,
        'original_shape': textures[0].shape
    }
    
    return converted_textures, metadata


def extract_mixing_channels(
    textures: List[np.ndarray],
    config: dict
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Extract channels to mix and channels to preserve.
    
    Args:
        textures: List of textures in target color space
        config: Color space configuration
        
    Returns:
        Tuple of (channels_to_mix, channels_to_preserve)
        channels_to_preserve is the first texture's preserved channels as reference
    """
    if config['all_channels']:
        return textures, None
    
    mixing_channels = config.get('mixing_channels', [0])
    preserve_channels = config.get('preserve_channels', [])
    
    textures_to_mix = []
    preserved_channels = None
    
    for i, tex in enumerate(textures):
        # Extract channels to mix
        if len(mixing_channels) == 1:
            mix_channel = tex[:, :, mixing_channels[0]:mixing_channels[0]+1]
        else:
            mix_channel = tex[:, :, mixing_channels]
        textures_to_mix.append(mix_channel)
        
        # Extract channels to preserve (use first texture as reference)
        if i == 0 and preserve_channels:
            preserved_channels = tex[:, :, preserve_channels]
    
    return textures_to_mix, preserved_channels


def reconstruct_from_channels(
    mixed_result: np.ndarray,
    preserved_channels: Optional[np.ndarray],
    config: dict
) -> np.ndarray:
    """
    Reconstruct full image from mixed and preserved channels.
    
    Args:
        mixed_result: Result after mixing (H, W, C_mix)
        preserved_channels: Preserved channels from first texture (H, W, C_preserve) or None
        config: Color space configuration
        
    Returns:
        Full reconstructed image (H, W, 3)
    """
    if config['all_channels'] or preserved_channels is None:
        return mixed_result
    
    mixing_channels = config.get('mixing_channels', [0])
    preserve_channels = config.get('preserve_channels', [])
    
    # Create output with correct number of channels
    h, w = mixed_result.shape[:2]
    output = np.zeros((h, w, config['channels']), dtype=np.float64)
    
    # Place mixed channels
    if len(mixing_channels) == 1:
        if len(mixed_result.shape) == 2:
            output[:, :, mixing_channels[0]] = mixed_result
        else:
            output[:, :, mixing_channels[0]] = mixed_result[:, :, 0]
    else:
        for i, ch in enumerate(mixing_channels):
            output[:, :, ch] = mixed_result[:, :, i]
    
    # Place preserved channels
    if preserved_channels is not None:
        for i, ch in enumerate(preserve_channels):
            output[:, :, ch] = preserved_channels[:, :, i]
    
    return output


def read_texture(path: str, target_size: int = 256):
    """
    Read and resize a texture image.
    
    Args:
        path: Path to the texture image
        target_size: Target size for width and height (default: 256)
    
    Returns:
        RGB image as numpy array
    """
    texture = cv.imread(path)
    texture = cv.cvtColor(texture, cv.COLOR_BGR2RGB)
    texture = cv.resize(texture, (target_size, target_size), interpolation=cv.INTER_AREA)
    return texture
