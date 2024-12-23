"""
This code is adapted from the core prediction algorithm of WASSFAST.
Original source: https://gitlab.com/fibe/wassfast
Original copyright (C) 2020 Filippo Bergamasco
Modified by [Feng Wang], [2024]

Adaptation rationale:
1. Converted original NumPy implementation to PyTorch for GPU acceleration
4. Added support for concurrent processing of multiple wave fields (ensemble members of ETKF)

Original paper citation:
Bergamasco, F., Benetazzo, A., Yoo, J., Torsello, A., Barbariol, F., Jeong, J. Y., ... & Cavaleri, L. (2021). 
Toward real-time optical estimation of ocean waves' space-time fields. 
Computers & Geosciences, 147, 104666.

License notice:
This code is released under GNU General Public License v3.0
Original copyright (C) 2020 Filippo Bergamasco

Note:
The main functions have been reimplemented in PyTorch while maintaining the 
original algorithm's mathematical foundations. The adaptation focuses on 
enabling efficient batch processing and GPU acceleration while preserving 
the accuracy of the wave field predictions.
"""



import torch
from torchvision.transforms import GaussianBlur

def tukeywin_m_torch(N, p):
    x = torch.linspace(0, 1, N)
    mask = torch.where((x >= 0) & (x <= p/2), 0.5 * (1 + torch.cos(2*torch.pi/p * (x - p/2))), 1.0)
    mask = torch.where((x > (1 - p/2)) & (x <= 1), 0.5 * (1 + torch.cos(2*torch.pi/p * (x - 1 + p/2))), mask)
    return mask

def compute_mask_torch(ZI, tukey_p=0.08, gauss_sigma=2.0):
    assert ZI.shape[-2] == ZI.shape[-1]
    N = ZI.shape[-1]
    mask = (~torch.isnan(ZI)).float()
    
    gaussian_blur = GaussianBlur(kernel_size=(5, 5), sigma=(gauss_sigma, gauss_sigma))
    mask = gaussian_blur(mask)
    
    mask[mask < 0.99] = 0
    mask = gaussian_blur(mask)
    
    maskborder = tukeywin_m_torch(N, tukey_p).view(1, 1, N, 1).repeat(1, 1, 1, N)
    maskborder = maskborder.to(mask.device)  
    mask = mask * maskborder
    return mask

def mat_points_predicate_torch(ensemble_tensor, config, dt=0.083, _xd=-1.0, _yd=1.0, device=torch.device('cpu')):
    KX_ab_tensor = torch.tensor(config.KX_ab, dtype=torch.float64, device=device)  
    KY_ab_tensor = torch.tensor(config.KY_ab, dtype=torch.float64, device=device)  
    
    df = compute_phase_diff_torch(KX_ab_tensor, KY_ab_tensor, _xd, _yd, dt, device=device)
    spec_current = elevation_to_spectrum_torch(ensemble_tensor)
    spec_pred = spec_current * torch.exp(df * 1j)
    ZIp_pred = spectrum_to_elevation_torch(spec_pred)
    ZIp_pred_masked = ZIp_pred * compute_mask_torch(ZIp_pred)
    # return ZIp_pred
    return ZIp_pred_masked

def compute_phase_diff_torch(KX_ab, KY_ab, xsign, ysign, dt, device=torch.device('cuda'), depth=None, current_vector=[-8, -6]):
    if depth is None:
        depth = torch.tensor(float('inf'), dtype=torch.float64, device=device)  
    depth_tensor = torch.full_like(KX_ab, depth, dtype=torch.float64, device=device)  
    Kmag = torch.sqrt(KX_ab ** 2 + KY_ab ** 2)
    temp = xsign * KX_ab + ysign * KY_ab
    # 将接近零的值设置为零
    zero_threshold = 1e-12
    temp[torch.abs(temp) < zero_threshold] = 0.0
    Ksign = torch.sign(temp)

    omega_sq = torch.where(depth_tensor == float('inf'), 9.8 * Kmag, 9.8 * Kmag * torch.tanh(Kmag * depth_tensor))
    ph_diff = Ksign * (torch.sqrt(omega_sq) + KX_ab * current_vector[0] + KY_ab * current_vector[1]) * dt

    ph_diff = torch.triu(ph_diff) - torch.tril(ph_diff)
    ones_tensor = torch.ones_like(KX_ab)
    ph_diff = ph_diff * (torch.triu(-ones_tensor) + torch.tril(ones_tensor))
    return ph_diff

def elevation_to_spectrum_torch(ZI):
    spec_scale = 1.0 / (ZI.shape[1] * ZI.shape[2])
    spec = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(ZI, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1))
    return spec * spec_scale

def spectrum_to_elevation_torch(spec):
    spec_scale = spec.shape[1] * spec.shape[2]
    ele = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(spec * spec_scale, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)))
    return ele