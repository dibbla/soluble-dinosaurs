# This class gives the Optimal Transport (OT) path for the flow matching
import torch

class OTFlowMatching:
    def __init__(
        self,
        interpolation="linear",
    ):
        self.interpolation = interpolation

    def get_interpolation(self, noise_sample, data_sample, t):
        t = t.view(-1, 1, 1, 1)
        if self.interpolation == "linear":
            return (1 - t) * noise_sample + t * data_sample
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")
        
    def loss(self, model, noise_sample, data_sample):
        t = torch.rand(noise_sample.size(0), device=noise_sample.device)        
        interpolated_samples = self.get_interpolation(noise_sample, data_sample, t)
        model_pred = model(interpolated_samples, t)
        loss = torch.nn.functional.mse_loss(model_pred, -noise_sample + data_sample)
        return loss
    
    def generate(self, model, steps=100, noise=None):
        if noise is None:
            noise = torch.randn(1, 3, 256, 256).to(next(model.parameters()).device)
        t_vals = torch.linspace(0, 1, steps).to(noise.device).unsqueeze(1)
        dt = t_vals[1] - t_vals[0]
        for t in t_vals[:-1]:
            noise = noise + model(noise, t) * dt
        return noise