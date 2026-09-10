import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """NAFNet-style two-branch gated block; SCA includes global pooling."""
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1))
        self.project = nn.Conv2d(hidden // 2, channels, 1)
        self.ffn = nn.Sequential(nn.Conv2d(channels, hidden, 1), SimpleGate(), nn.Conv2d(hidden // 2, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        z = self.gate(self.depthwise(self.expand(self.norm1(x))))
        y = x + self.beta * self.project(z * self.sca(z))
        return y + self.gamma * self.ffn(self.norm2(y))


class FiLM(nn.Module):
    def __init__(self, channels, condition_dim=128):
        super().__init__()
        self.map = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.map.bias)

    def forward(self, x, condition):
        scale, shift = self.map(condition).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class BandStem(nn.Module):
    def __init__(self, blocks=1, expansion=2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), *[NAFBlock(16, expansion) for _ in range(blocks)])
        self.embedding = nn.Embedding(4, 16)
        nn.init.normal_(self.embedding.weight, std=.02)

    def forward(self, x):
        b, _, h, w = x.shape
        z = self.stem(x.reshape(b * 4, 1, h, w)).reshape(b, 4, 16, h, w)
        return z + self.embedding.weight[None, :, :, None, None]


class SpectralAttention(nn.Module):
    """Exactly four 16-D band tokens at each LR spatial coordinate."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)
        self.attn = nn.MultiheadAttention(16, 4, dropout=0, batch_first=True)

    def forward(self, features, return_weights=False):
        b, bands, d, h, w = features.shape
        if (bands, d) != (4, 16):
            raise ValueError("Spectral attention requires [B,4,16,H,W]")
        tokens = features.permute(0, 3, 4, 1, 2).reshape(b * h * w, 4, 16)
        norm = self.norm(tokens)
        out, weights = self.attn(norm, norm, norm, need_weights=return_weights, average_attn_weights=False)
        out = (tokens + out).reshape(b, h, w, 4, 16).permute(0, 3, 4, 1, 2)
        return (out, weights) if return_weights else out


def partition(x, window):
    b, h, w, c = x.shape
    return x.reshape(b, h // window, window, w // window, window, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def unpartition(x, window, h, w, batch):
    return x.reshape(batch, h // window, w // window, window, window, -1).permute(0, 1, 3, 2, 4, 5).reshape(batch, h, w, -1)


class SwinBlock(nn.Module):
    """Window attention with relative position bias, shift mask and padding mask."""
    def __init__(self, width=64, heads=4, window=8, shift=0, mlp_ratio=2):
        super().__init__()
        self.width, self.heads, self.window, self.shift = width, heads, window, shift
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, width * mlp_ratio), nn.GELU(), nn.Linear(width * mlp_ratio, width))
        self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0) + window - 1
        self.register_buffer("relative_index", relative[..., 0] * (2 * window - 1) + relative[..., 1])
        nn.init.trunc_normal_(self.relative_bias, std=.02)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window
        ph, pw = (-h) % ws, (-w) % ws
        hp, wp = h + ph, w + pw
        shift = self.shift if min(h, w) > ws else 0
        tokens = x.permute(0, 2, 3, 1)
        z = F.pad(self.norm1(tokens), (0, 0, 0, pw, 0, ph))
        valid = torch.zeros(1, hp, wp, 1, device=x.device, dtype=torch.bool)
        valid[:, :h, :w] = True
        if shift:
            z = torch.roll(z, (-shift, -shift), (1, 2))
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        windows = partition(z, ws)
        n = ws * ws
        qkv = self.qkv(windows).reshape(-1, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        bias = self.relative_bias[self.relative_index.reshape(-1)].reshape(n, n, self.heads).permute(2, 0, 1)
        key_valid = partition(valid, ws).squeeze(-1)
        allowed = key_valid[:, None, :].expand(-1, n, -1).clone()
        if shift:
            regions = torch.zeros(1, hp, wp, 1, device=x.device)
            slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
            for i, hs in enumerate(slices):
                for j, vs in enumerate(slices):
                    regions[:, hs, vs] = i * 3 + j
            labels = partition(regions, ws).squeeze(-1)
            allowed &= labels[:, :, None] == labels[:, None, :]
        # Padded queries are discarded; give them a finite self-attention row.
        diagonal = torch.eye(n, device=x.device, dtype=torch.bool)[None]
        allowed |= (~key_valid)[:, :, None] & diagonal
        mask = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
        mask = mask.repeat(b, 1, 1)[:, None] + bias[None].to(q.dtype)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0)
        attended = self.proj(attended.transpose(1, 2).reshape(-1, n, c))
        z = unpartition(attended, ws, hp, wp, b)
        if shift:
            z = torch.roll(z, (shift, shift), (1, 2))
        y = tokens + z[:, :h, :w]
        y = y + self.mlp(self.norm2(y))
        return y.permute(0, 3, 1, 2).contiguous()


class DegradationEncoder(nn.Module):
    """Availability is explicit; missing metadata never masquerades as measurement."""
    def __init__(self, metadata_dim=8):
        super().__init__()
        self.metadata_dim = metadata_dim
        layers, previous = [], 4
        for channels in (16, 32, 48, 64):
            layers.extend([nn.Conv2d(previous, channels, 3, stride=2, padding=1), nn.GELU()])
            previous = channels
        self.image = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.metadata = nn.Sequential(nn.Linear(2 * metadata_dim, 64), nn.GELU())
        self.combine = nn.Linear(128, 128)

    def forward(self, x, metadata=None, availability=None):
        shape = (x.shape[0], self.metadata_dim)
        if metadata is None:
            metadata, availability = x.new_zeros(shape), x.new_zeros(shape)
        if availability is None or metadata.shape != shape or availability.shape != shape:
            raise ValueError("Metadata requires values and matching availability [B,metadata_dim]")
        if not torch.isfinite(availability).all() or ((availability < 0) | (availability > 1)).any():
            raise ValueError("Metadata availability must be finite in [0,1]")
        safe = torch.where(availability > 0, metadata, torch.zeros_like(metadata))
        if not torch.isfinite(safe).all():
            raise ValueError("Available metadata must be finite")
        return self.combine(torch.cat([self.image(x), self.metadata(torch.cat([safe, availability], 1))], 1))


class ResidualDecoder(nn.Module):
    def __init__(self, rich_heads=False, conditioned=True, expansion=2):
        super().__init__()
        self.rich_heads = rich_heads
        self.up1 = nn.Sequential(nn.Conv2d(64, 192, 3, padding=1), nn.PixelShuffle(2))
        self.refine1 = NAFBlock(48, expansion)
        self.up2 = nn.Sequential(nn.Conv2d(48, 128, 3, padding=1), nn.PixelShuffle(2))
        self.refine2 = NAFBlock(32, expansion)
        self.film1 = FiLM(48) if conditioned else None
        self.film2 = FiLM(32) if conditioned else None
        channels = 48 if rich_heads else 32
        self.heads = nn.ModuleList([nn.Conv2d(channels, 1, 3, padding=1) for _ in range(4)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features, band_features, condition=None):
        z = self.refine1(self.up1(features))
        if self.film1 is not None:
            z = self.film1(z, condition)
        z = self.refine2(self.up2(z))
        if self.film2 is not None:
            z = self.film2(z, condition)
        residuals = []
        for i, head in enumerate(self.heads):
            own = F.interpolate(band_features[:, i], size=z.shape[-2:], mode="bilinear", align_corners=False) if self.rich_heads else None
            residuals.append(head(torch.cat([z, own], 1) if own is not None else z))
        return torch.cat(residuals, 1), z
