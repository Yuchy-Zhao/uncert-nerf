import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3, lambda_uncertainty=False):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.lambda_uncertainty = lambda_uncertainty

    def uncert_loss(self, x, y, uncert, alphas, w=0.01):
        term1 = torch.mean(((x - y) ** 2) / (2 * (uncert+1e-9).unsqueeze(-1)), dim=1)
        term2 = 0.5 * torch.log(uncert+1e-9)
        term3 = w * torch.mean(alphas, dim=1)
        return term1 + term2 + term3

    def forward(self, results, target, **kwargs):
        d = {}
        # RGB Loss
        d['rgb'] = torch.mean((results['rgb_coarse']-target['rgb'])**2, dim=1)
        if 'rgb_fine' in results:
            if self.lambda_uncertainty:
                if results['uncert_fine'].min() > 0:
                    d['rgb'] += self.uncert_loss(results['rgb_fine'], target['rgb'], results['uncert_fine'], results['alphas_fine'])
                else:
                    d['rgb'] += torch.mean((results['rgb_fine']-target['rgb'])**2, dim=1)
            else:
                d['rgb'] += torch.mean((results['rgb_fine']-target['rgb'])**2, dim=1)
        
        # Loss of Depth
        if kwargs.get('use_depth', True):
            d['depth'] = torch.abs(results['depth']-target['depth'])
        
        # Opacity Loss
        if self.lambda_opacity > 0:
            o = results['opacity']+1e-10
            # # encourage opacity to be either 0 or 1 to avoid floater
            d['opacity'] = self.lambda_opacity*(-o*torch.log(o))
        
        # Distortion Loss
        if self.lambda_distortion > 0:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])

        return d
