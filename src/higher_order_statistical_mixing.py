import time
import numpy as np
import torch
import pyrtools as pt
from .sliced_wasserstein import (
    sliced_wasserstein_2_barycenter,
    sliced_wasserstein_2_projection
)
from .utils import (
    ColorSpace,
    prepare_textures_for_mixing,
    extract_mixing_channels,
    reconstruct_from_channels,
    convert_from_colorspace
)


def extract_non_overlapping_blocks(coeffs_rgb: list[torch.Tensor], 
                                     block_size: int) -> tuple[torch.Tensor, list]:
    h, w = coeffs_rgb[0].shape
    
    # Calculate number of blocks that fit
    n_blocks_h = h // block_size
    n_blocks_w = w // block_size

    blocks = []
    positions = []
    
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            block_i = i * block_size
            block_j = j * block_size
            
            block_features = []
            for c in range(3):
                block = coeffs_rgb[c][block_i:block_i+block_size, block_j:block_j+block_size]
                block_features.append(block.flatten())
            
            block_vec = torch.cat(block_features)
            blocks.append(block_vec)
            positions.append((block_i, block_j))
    
    blocks = torch.stack(blocks)
    return blocks, positions


def reconstruct_from__nonoverlapping_blocks(blocks: torch.Tensor, 
                                               positions: list, 
                                               shape: tuple, 
                                               block_size: int,
                                               device='cpu') -> list[torch.Tensor]:
    h, w = shape
    coeffs_out = [torch.zeros((h, w), dtype=torch.float32, device=device) for _ in range(3)]
    
    block_len = block_size * block_size
    
    for idx, (i, j) in enumerate(positions):
        block_vec = blocks[idx]
        
        for c in range(3):
            block_c = block_vec[c * block_len:(c + 1) * block_len]
            block_2d = block_c.reshape(block_size, block_size)
            coeffs_out[c][i:i+block_size, j:j+block_size] = block_2d
    
    return coeffs_out

def higher_order_texture_mixing(textures: list[np.ndarray], 
                                    rhos: np.ndarray,
                                    height: int = 4,
                                    order: int = 3,
                                    n_iter: int = 10,
                                    K: int = 128,
                                    step_size: float = 0.1,
                                    barycenter_iter: int = 50,
                                    proj_iter: int = 15,
                                    block_size: int = 12,
                                    device: str = 'cuda',
                                    color_space: ColorSpace = ColorSpace.RGB,
                                    verbose: bool = True) -> np.ndarray:
    """
    Higher-order texture mixing using sliced Wasserstein barycenters.
    Implements the algorithm from Section 4.3 of the paper.
    """
    J = len(textures)
    rhos_gpu = torch.tensor(rhos, dtype=torch.float32, device=device)
    
    # Normalize textures
    textures = [tex.astype(np.float64) / 255.0 if tex.max() > 1 else tex.astype(np.float64) 
                for tex in textures]
    
    # Convert to target color space
    converted_textures, metadata = prepare_textures_for_mixing(textures, color_space)
    config = metadata['config']
    
    if verbose:
        print(f"Mixing in {color_space.value} color space")
        print(f"Block size: {block_size}x{block_size}")
        print(f"Mixing channels: {[config['channel_names'][i] for i in config.get('mixing_channels', range(config['channels']))]}")
    
    # Extract channels to mix and preserve
    textures_to_mix, preserved_channels = extract_mixing_channels(converted_textures, config)
    
    P, Q, _ = textures_to_mix[0].shape
    
    if verbose:
        print(f"Computing steerable pyramids for {J} textures of size {P}x{Q}...")
        start_time = time.time()
    
    # Build pyramids
    pyramids = []
    for j, tex in enumerate(textures_to_mix):
        pyr_channels = []
        num_channels = tex.shape[2] if len(tex.shape) == 3 else 1
        for c in range(num_channels):
            if num_channels == 1:
                pyr_c = pt.pyramids.SteerablePyramidFreq(tex[:, :, 0], height=height, order=order-1, is_complex=False)
            else:
                pyr_c = pt.pyramids.SteerablePyramidFreq(tex[:, :, c], height=height, order=order-1, is_complex=False)
            pyr_channels.append(pyr_c)
        pyramids.append(pyr_channels)
    
    pyr_keys = list(pyramids[0][0].pyr_coeffs.keys())
    
    # Identify oriented bands vs residuals
    oriented_keys = [k for k in pyr_keys if isinstance(k, tuple) and len(k) == 2]
    
    if verbose:
        print(f"  Pyramid decomposition complete ({time.time() - start_time:.2f}s)")
        print(f"  {len(pyr_keys)} total subbands ({len(oriented_keys)} oriented bands)")
        print(f"\nComputing first-order barycenters on GPU...")
        start_time = time.time()
    
    # STEP 1: Compute first-order barycenters Y_ℓ (marginal distributions)
    Y_ell = {}
    coeff_shapes = {}
    
    for l_idx, key in enumerate(oriented_keys):
        Y_list_cpu = []
        coeff_shape = pyramids[0][0].pyr_coeffs[key].shape
        coeff_shapes[key] = coeff_shape
        
        for j in range(J):
            coeffs_rgb = []
            for c in range(len(pyramids[j])):
                coeff = pyramids[j][c].pyr_coeffs[key]
                coeffs_rgb.append(coeff.flatten())
            coeff_flat = np.stack(coeffs_rgb, axis=1)
            Y_list_cpu.append(coeff_flat)
        
        # Convert to torch tensors
        Y_list_gpu = [torch.from_numpy(y).float().to(device) for y in Y_list_cpu]
        
        Y_ell[key] = sliced_wasserstein_2_barycenter(
            Y_list_gpu, rhos_gpu, K=K, step_size=step_size, n_iter=barycenter_iter, device=device
        )
        
        if verbose and (l_idx + 1) % 5 == 0:
            print(f"  Processed {l_idx + 1}/{len(oriented_keys)} first-order barycenters")
    
    if verbose:
        print(f"  First-order barycenters complete ({time.time() - start_time:.2f}s)")
        print(f"\nComputing joint distribution barycenters (block_size={block_size}x{block_size}) on GPU...")
        start_time = time.time()
    
    # STEP 2: Compute joint distribution barycenters C_ℓ^N (spatial correlations)
    C_ell = {}
    
    for l_idx, key in enumerate(oriented_keys):
        h, w = coeff_shapes[key]
        
        # Skip if subband is too small for blocks
        if h < block_size or w < block_size:
            if verbose:
                print(f"  Skipping {key}: too small ({h}x{w}) for {block_size}x{block_size} blocks")
            continue
        
        C_list_gpu = []
        
        for j in range(J):
            # Move coefficients to GPU as torch tensors
            coeffs_rgb_gpu = []
            for c in range(len(pyramids[j])):
                coeff = pyramids[j][c].pyr_coeffs[key]
                coeffs_rgb_gpu.append(torch.from_numpy(coeff).float().to(device))
            
            blocks_gpu, _ = extract_non_overlapping_blocks(coeffs_rgb_gpu, block_size)
            
            if len(blocks_gpu) > 0:
                C_list_gpu.append(blocks_gpu)
        
        # Compute barycenter of joint distributions
        if len(C_list_gpu) > 0 and len(C_list_gpu[0]) > 0:
            C_ell[key] = sliced_wasserstein_2_barycenter(
                C_list_gpu, rhos_gpu, K=K, step_size=step_size, n_iter=barycenter_iter, device=device
            )
        
        if verbose and (l_idx + 1) % 5 == 0:
            print(f"  Processed {l_idx + 1}/{len(oriented_keys)} joint barycenters")
    
    if verbose:
        print(f"  Joint barycenters complete ({time.time() - start_time:.2f}s)")
        print("\nComputing pixel barycenter on GPU...")
        start_time = time.time()
    
    # STEP 3: Compute pixel barycenter (on mixing channels only)
    pixel_Y_list_gpu = [torch.from_numpy(tex.reshape(-1, tex.shape[2])).float().to(device) for tex in textures_to_mix]
    Y_pixels = sliced_wasserstein_2_barycenter(
        pixel_Y_list_gpu, rhos_gpu, K=K, step_size=step_size, n_iter=barycenter_iter, device=device
    )
    
    if verbose:
        print(f"  Pixel barycenter complete ({time.time() - start_time:.2f}s)")
        print(f"\nStarting iterative synthesis ({n_iter} iterations)...")
    
    # Initialize with white noise
    f_k = np.random.randn(P, Q, len(pyramids[0])).astype(np.float64) * 0.2 + 0.5
    f_k = np.clip(f_k, 0, 1)
    
    # STEP 4: Iterative synthesis
    for k in range(n_iter):
        iter_start = time.time()
        
        # Build pyramid of current image
        pyr_k_channels = []
        for c in range(len(pyramids[0])):
            pyr_k_c = pt.pyramids.SteerablePyramidFreq(f_k[:, :, c], height=height, order=order-1, is_complex=False)
            pyr_k_channels.append(pyr_k_c)
        
        # Project coefficients - TWO STAGE PROCESS
        c_k_channels = [{} for _ in range(len(pyramids[0]))]
        
        for key in oriented_keys:
            original_shape = coeff_shapes[key]
            h, w = original_shape
            
            # STAGE 1: First-order projection (equation 13 in paper)
            # Extract current coefficients
            coeffs_rgb = []
            for c in range(len(pyramids[0])):
                coeff = pyr_k_channels[c].pyr_coeffs[key]
                coeffs_rgb.append(coeff.flatten())
            
            coeff_flat_cpu = np.stack(coeffs_rgb, axis=1)
            coeff_flat_gpu = torch.from_numpy(coeff_flat_cpu).float().to(device)
            
            # Project to first-order barycenter Y_ℓ
            coeff_proj_gpu = sliced_wasserstein_2_projection(
                coeff_flat_gpu, Y_ell[key], K=K, step_size=step_size, n_iter=proj_iter, device=device
            )
            
            # STAGE 2: Higher-order projection (joint distributions)
            if key in C_ell and h >= block_size and w >= block_size:
                # Reshape projected coefficients back to 2D for block extraction
                coeffs_proj_rgb_gpu = []
                for c in range(len(pyramids[0])):
                    coeff_c = coeff_proj_gpu[:, c].reshape(original_shape)
                    coeffs_proj_rgb_gpu.append(coeff_c)
                
                # Extract blocks from PROJECTED coefficients (not original!)
                blocks_k_gpu, block_positions = extract_non_overlapping_blocks(coeffs_proj_rgb_gpu, block_size)
                
                if len(blocks_k_gpu) > 0:
                    # Project blocks to joint distribution barycenter C_ℓ^N
                    blocks_proj_gpu = sliced_wasserstein_2_projection(
                        blocks_k_gpu, C_ell[key], K=K, step_size=step_size, n_iter=proj_iter, device=device
                    )
                    
                    # Reconstruct coefficients from projected blocks
                    coeffs_final_rgb_gpu = reconstruct_from__nonoverlapping_blocks(
                        blocks_proj_gpu, block_positions, original_shape, block_size, device=device
                    )
                    
                    # Store final coefficients
                    for c in range(len(pyramids[0])):
                        c_k_channels[c][key] = coeffs_final_rgb_gpu[c].cpu().numpy()
                else:
                    # If block extraction failed, use first-order projection only
                    coeff_proj_cpu = coeff_proj_gpu.cpu().numpy()
                    for c in range(len(pyramids[0])):
                        c_k_channels[c][key] = coeff_proj_cpu[:, c].reshape(original_shape)
            else:
                # If no joint distribution available, use first-order projection only
                coeff_proj_cpu = coeff_proj_gpu.cpu().numpy()
                for c in range(len(pyramids[0])):
                    c_k_channels[c][key] = coeff_proj_cpu[:, c].reshape(original_shape)
        
        # Reconstruct image - preserve lowpass/highpass residuals
        f_tilde_k_channels = []
        for c in range(len(pyramids[0])):
            # Preserve all original coefficients, then update with projected ones
            new_coeffs = pyr_k_channels[c].pyr_coeffs.copy()
            new_coeffs.update(c_k_channels[c])
            pyr_k_channels[c].pyr_coeffs = new_coeffs
            
            recon_c = pyr_k_channels[c].recon_pyr()
            f_tilde_k_channels.append(recon_c)
        
        f_tilde_k = np.stack(f_tilde_k_channels, axis=-1)
        
        # Project pixels (equation 14 in paper)
        pixels_flat_gpu = torch.from_numpy(f_tilde_k.reshape(-1, len(pyramids[0]))).float().to(device)
        pixels_proj_gpu = sliced_wasserstein_2_projection(
            pixels_flat_gpu, Y_pixels, K=K, step_size=step_size, n_iter=proj_iter, device=device
        )
        
        f_k = pixels_proj_gpu.cpu().numpy().reshape(P, Q, len(pyramids[0]))
        f_k = np.clip(f_k, 0, 1)
        
        if verbose:
            print(f"  Iteration {k+1}/{n_iter} complete ({time.time() - iter_start:.2f}s)")
    
    if verbose:
        print("\nSynthesis complete!")
    
    # Reconstruct full image with preserved channels
    f_k = reconstruct_from_channels(f_k, preserved_channels, config)
    
    # Convert back to RGB
    f_k_rgb = convert_from_colorspace(f_k, color_space)
    
    result = (f_k_rgb * 255).astype(np.uint8)
    return result