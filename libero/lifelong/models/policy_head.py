import robomimic.utils.tensor_utils as TensorUtils
import math
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class DeterministicHead(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024, num_layers=2):

        super().__init__()
        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if self.action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class GMMHead(nn.Module):
    def __init__(
        self,
        # network_kwargs
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        min_std=0.0001,
        num_modes=5,
        activation="softplus",
        low_eval_noise=False,
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.output_size = output_size
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_size * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)

        self.low_eval_noise = low_eval_noise
        self.loss_coef = loss_coef

        if activation == "softplus":
            self.actv = F.softplus
        else:
            self.actv = torch.exp

    def forward_fn(self, x):
        # x: (B, input_size)
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_size)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training or not self.low_eval_noise:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_size
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * 1e-4
        return means, stds, logits

    def forward(self, x):
        if x.ndim == 3:
            means, scales, logits = TensorUtils.time_distributed(x, self.forward_fn)
        elif x.ndim < 3:
            means, scales, logits = self.forward_fn(x)

        compo = D.Normal(loc=means, scale=scales)
        compo = D.Independent(compo, 1)
        mix = D.Categorical(logits=logits)
        gmm = D.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        return gmm

    def loss_fn(self, gmm, target, reduction="mean"):
        log_probs = gmm.log_prob(target)
        loss = -log_probs
        if reduction == "mean":
            return loss.mean() * self.loss_coef
        elif reduction == "none":
            return loss * self.loss_coef
        elif reduction == "sum":
            return loss.sum() * self.loss_coef
        else:
            raise NotImplementedError


class DiffusionDist:
    def __init__(self, head, cond):
        self.head = head
        self.cond = cond

    def sample(self):
        B = self.cond.shape[0]
        device = self.cond.device
        action = torch.randn((B, self.head.output_size), device=device)
        for t in range(self.head.num_train_timesteps - 1, -1, -1):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.head.forward_fn(action, t_tensor, self.cond)
            action = self.head.step(pred_noise, t_tensor, action)
        return torch.tanh(action)

# Sinusoidal position embedding for time steps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.float()
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Simple MLP-based denoising model
class SimpleDenoise(nn.Module):
    def __init__(
        self,
        action_dim,
        cond_dim,
        time_emb_dim=64,
        hidden_size=512,
        num_layers=3,
    ):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        layers = []
        input_size = action_dim + cond_dim + time_emb_dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.Mish(),
            ]
        layers += [nn.Linear(hidden_size, action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, noisy_a, t, cond):
        device = next(self.parameters()).device
        noisy_a = noisy_a.to(device)
        t = t.to(device)
        cond = cond.to(device)
        temb = self.time_emb(t)
        inp = torch.cat([noisy_a, cond, temb], dim=-1)
        return self.net(inp)

# Simple DDPM scheduler implemented in pure torch
class DDPMSchedulerTorch:
    def __init__(
        self,
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise ValueError("Unsupported beta_schedule")
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(original_samples.device)[timesteps]
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, *((1,) * (len(original_samples.shape) - 1)))
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, *((1,) * (len(original_samples.shape) - 1)))
        return sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise

    def step(self, model_output, timestep, sample):
        t = timestep
        prev_t = t - 1

        beta_t = self.betas.to(sample.device)[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(sample.device)[t].unsqueeze(1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas.to(sample.device)[t].unsqueeze(1)
        model_output = model_output.to(sample.device).view_as(sample)

        # Standard DDPM formula for mu_t
        posterior_mean = sqrt_recip_alphas_t * (sample - beta_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = self.posterior_variance.to(sample.device)[t].unsqueeze(1)

        noise = torch.randn_like(sample) if prev_t.min() >= 0 else torch.zeros_like(sample)
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise

class DiffusionHead(nn.Module):
    def __init__(
        self,
        # network_kwargs (similar to GMMHead)
        input_size,
        output_size,
        hidden_size=512,
        num_layers=3,
        # diffusion-specific
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        min_std=0.0001,  # Unused but for API parity
        low_eval_noise=False,  # Can adapt for evaluation
        # loss_kwargs
        loss_coef=1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.min_std = min_std  # Unused but for parity
        self.low_eval_noise = low_eval_noise  # Can use for inference noise scaling
        self.loss_coef = loss_coef
        self.num_train_timesteps = num_train_timesteps

        # Pure torch scheduler
        self.scheduler = DDPMSchedulerTorch(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        # Denoising model
        self.forward_fn = SimpleDenoise(
            action_dim=output_size,
            cond_dim=input_size,
            time_emb_dim=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).forward

    def step(self, model_output, timestep, sample):
        return self.scheduler.step(model_output, timestep, sample)

    def forward(self, x):
        # Handle sequence dimension if present (match GMM's time_distributed)
        if x.ndim == 3:
            x = TensorUtils.join_dimensions(x, 0, 1)  # (B*T, input_size)
        return DiffusionDist(self, x)

    def loss_fn(self, dist, target, reduction="mean"):
        # Extract conditioning from dist (matches API: dist from forward(x))
        cond = dist.cond
        # Handle sequence dimension if present
        if target.ndim == 3:
            target = TensorUtils.join_dimensions(target, 0, 1)  # (B*T, output_size)
        B = target.shape[0]
        device = target.device

        # Standard DDPM loss computation
        t = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(target).to(device)
        noisy_target = self.scheduler.add_noise(target, noise, t)
        pred_noise = self.forward_fn(noisy_target, t, cond)
        pred_noise = pred_noise.to(device)
        loss = F.mse_loss(pred_noise, noise, reduction="none").mean(-1)  # Mean over action dims

        # Apply reduction
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise NotImplementedError
        return loss * self.loss_coef