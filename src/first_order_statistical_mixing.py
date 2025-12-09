import time
import numpy as np
import torch
import pyrtools as pt

from .sliced_wasserstein import sliced_wasserstein_2_barycenter, sliced_wasserstein_2_projection
from .utils import (
    ColorSpace,
    prepare_textures_for_mixing,
    extract_mixing_channels,
    reconstruct_from_channels,
    convert_from_colorspace
)


def first_order_texture_mixing(textures: list[np.ndarray], 
                                rhos: np.ndarray,
                                height: int = 4,
                                order: int = 3,
                                n_iter=10,                  
                                K=128,                       
                                step_size=0.1,              
                                barycenter_iter=50,       
                                proj_iter=15,             
                                device='cpu',
                                color_space: ColorSpace = ColorSpace.RGB,
                                verbose: bool = True) -> np.ndarray:
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
        print(f"Mixing channels: {[config['channel_names'][i] for i in config.get('mixing_channels', range(config['channels']))]}")
    
    # Extract channels to mix and preserve
    textures_to_mix, preserved_channels = extract_mixing_channels(converted_textures, config)
    
    P, Q, _ = textures_to_mix[0].shape
    
    if verbose:
        print(f"Computing steerable pyramids for {J} textures of size {P}x{Q}...")
        start_time = time.time()
    
    # Build steerable pyramids 
    # NOTE: this is a CPU operation via pyrtools, a possible improvement is to move this to the GPU
    pyramids = []
    for j, tex in enumerate(textures_to_mix):
        pyr_channels = []
        # Handle single channel (grayscale) or multi-channel
        num_channels = tex.shape[2] if len(tex.shape) == 3 else 1
        for c in range(num_channels):
            if num_channels == 1:
                pyr_c = pt.pyramids.SteerablePyramidFreq(tex[:, :, 0], height=height, order=order-1, is_complex=False)
            else:
                pyr_c = pt.pyramids.SteerablePyramidFreq(tex[:, :, c], height=height, order=order-1, is_complex=False)
            pyr_channels.append(pyr_c)
        pyramids.append(pyr_channels)
    
    pyr_keys = list(pyramids[0][0].pyr_coeffs.keys())
    
    # Identify which keys are oriented bands (to be projected) vs residuals (to be preserved)
    # Oriented bands typically have tuple keys like (scale, orientation)
    oriented_keys = [k for k in pyr_keys if isinstance(k, tuple) and len(k) == 2]
    
    if verbose:
        print(f"  Pyramid decomposition complete ({time.time() - start_time:.2f}s)")
        print(f"  {len(pyr_keys)} total subbands ({len(oriented_keys)} oriented bands)")
        print("\nComputing coefficient barycenters on GPU...")
        start_time = time.time()
    
    # Compute barycenters for each oriented subband only
    Y_ell = {}
    num_channels = pyramids[0][0].pyr_coeffs[oriented_keys[0]].shape[-1] if len(pyramids[0][0].pyr_coeffs[oriented_keys[0]].shape) > 2 else len(pyramids[0])
    
    for l_idx, key in enumerate(oriented_keys):
        Y_list_cpu = []
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
            print(f"  Processed {l_idx + 1}/{len(oriented_keys)} oriented subbands")
    
    if verbose:
        print(f"  Coefficient barycenters complete ({time.time() - start_time:.2f}s)")
        print("\nComputing pixel barycenter on GPU...")
        start_time = time.time()
    
    # Compute pixel barycenter (on mixing channels only)
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
    
    # Iterative synthesis
    for k in range(n_iter):
        iter_start = time.time()
        
        # Build pyramid of current image
        pyr_k_channels = []
        for c in range(len(pyramids[0])):
            pyr_k_c = pt.pyramids.SteerablePyramidFreq(f_k[:, :, c], height=height, order=order-1, is_complex=False)
            pyr_k_channels.append(pyr_k_c)
        
        # Project coefficients - ONLY oriented bands, preserve residuals
        c_k_channels = [{} for _ in range(len(pyramids[0]))]
        for key in oriented_keys:
            coeffs_rgb = []
            original_shape = None
            for c in range(len(pyramids[0])):
                coeff = pyr_k_channels[c].pyr_coeffs[key]
                if original_shape is None:
                    original_shape = coeff.shape
                coeffs_rgb.append(coeff.flatten())
            
            coeff_flat_cpu = np.stack(coeffs_rgb, axis=1)
            coeff_flat_gpu = torch.from_numpy(coeff_flat_cpu).float().to(device)
            
            # Use safe projection wrapper to preserve ordering
            coeff_proj_gpu = sliced_wasserstein_2_projection(
                coeff_flat_gpu, Y_ell[key], K=K, step_size=step_size, n_iter=proj_iter, device=device
            )
            
            coeff_proj_cpu = coeff_proj_gpu.cpu().numpy()
            for c in range(len(pyramids[0])):
                c_k_channels[c][key] = coeff_proj_cpu[:, c].reshape(original_shape)
        
        # Reconstruct image - CRITICAL FIX: preserve all coefficients
        f_tilde_k_channels = []
        for c in range(len(pyramids[0])):
            # FIX Issue #1: Use .copy() and .update() to preserve lowpass/highpass residuals
            new_coeffs = pyr_k_channels[c].pyr_coeffs.copy()
            new_coeffs.update(c_k_channels[c])
            pyr_k_channels[c].pyr_coeffs = new_coeffs
            
            recon_c = pyr_k_channels[c].recon_pyr()
            f_tilde_k_channels.append(recon_c)
        
        f_tilde_k = np.stack(f_tilde_k_channels, axis=-1)
        
        # Project pixels - use safe projection
        pixels_flat_gpu = torch.from_numpy(f_tilde_k.reshape(-1, len(pyramids[0]))).float().to(device)
        pixels_proj_gpu = sliced_wasserstein_2_projection(
            pixels_flat_gpu, Y_pixels, K=K, step_size=step_size, n_iter=proj_iter, device=device
        )
        
        pixels_proj_cpu = pixels_proj_gpu.cpu().numpy()
        f_k = pixels_proj_cpu.reshape(P, Q, len(pyramids[0]))
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