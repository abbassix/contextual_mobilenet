import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class ContextualConv1d(nn.Module):
    """
    Custom 1D convolution layer with optional global context conditioning.
    Uses unfold + matrix multiplication instead of nn.Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        c_dim: Optional[int] = None,
    ):
        super().__init__()

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.c_dim = c_dim

        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups

        self.weight = nn.Parameter(torch.randn(
            out_channels, self.group_in_channels, kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.c_weight = (
            nn.Parameter(torch.randn(out_channels, c_dim)) if c_dim is not None else None
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, _, L = x.shape
        print(f'{x.shape=}')
        k, s, p, d = self.kernel_size, self.stride, self.padding, self.dilation

        x_padded = F.pad(x, (p, p))
        # print(f'{x_padded.shape=}')
        patches = x_padded.unfold(dimension=2, size=k, step=s)  # (N, C_in, L_out, K)
        # print(f'{patches=}')
        L_out = patches.shape[2]
        unfolded = patches.permute(0, 2, 1, 3).reshape(N, L_out, -1)  # (N, L_out, C_in * K)
        # print(f'{unfolded.shape=}')

        outputs = []

        for g in range(self.groups):
            weight_g = self.weight[
                g * self.group_out_channels : (g + 1) * self.group_out_channels
            ].reshape(self.group_out_channels, -1)
            # print(f'{weight_g.shape=}')
            
            start = g * self.group_in_channels * k
            end = (g + 1) * self.group_in_channels * k
            input_group = unfolded[:, :, start:end]
            # print(f'{input_group.shape=}')
            
            if self.c_dim is not None and c is not None:
                if c.shape[-1] != self.c_dim:
                    raise ValueError(f"Expected c.shape[-1] = {self.c_dim}, got {c.shape[-1]}")
                c_expanded = c.view(N, 1, self.c_dim).expand(N, L_out, self.c_dim)
                # print(f'{c_expanded.shape=}')
                c_weight_g = self.c_weight[
                    g * self.group_out_channels : (g + 1) * self.group_out_channels
                ]
                # print(f'{c_weight_g.shape=}')
                weight_g = torch.cat([weight_g, c_weight_g], dim=1)
                # print(f'{weight_g.shape=}')
                input_group = torch.cat([input_group, c_expanded], dim=-1)
                # print(f'{input_group.shape=}')

            output_g = torch.matmul(input_group, weight_g.T)
            # print(f'{output_g.shape=}')
            outputs.append(output_g)

        out = torch.cat(outputs, dim=-1)
        if self.bias is not None:
            out += self.bias.view(1, 1, -1)

        return out.permute(0, 2, 1)  # (N, out_channels, L_out)


class ContextualConv2d(nn.Module):
    """
    Custom 2D convolution layer with optional global context conditioning.
    Uses im2col (unfold) + matrix multiplication instead of nn.Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        c_dim: Optional[int] = None,
    ):
        super().__init__()

        self.kernel_size = self._to_pair(kernel_size)
        self.stride = self._to_pair(stride)
        self.padding = self._to_pair(padding)
        self.dilation = self._to_pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.padding_mode = padding_mode
        self.c_dim = c_dim

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups

        self.weight = nn.Parameter(torch.randn(
            out_channels, self.group_in_channels, *self.kernel_size
        ))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.c_weight = (
            nn.Parameter(torch.randn(out_channels, c_dim)) if c_dim is not None else None
        )

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, _, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation

        if self.padding_mode == 'zeros':
            x_padded = F.pad(x, (pw, pw, ph, ph), mode='constant', value=0)
        else:
            x_padded = F.pad(x, (pw, pw, ph, ph), mode=self.padding_mode)

        patches = x_padded.unfold(2, kh, sh).unfold(3, kw, sw)
        patches = patches[..., ::dh, ::dw]

        out_h, out_w = patches.shape[2], patches.shape[3]
        unfolded = patches.permute(0, 2, 3, 1, 4, 5).reshape(N, out_h * out_w, -1)

        outputs = []

        for g in range(self.groups):
            weight_g = self.weight[
                g * self.group_out_channels : (g + 1) * self.group_out_channels
            ].reshape(self.group_out_channels, -1)

            start = g * self.group_in_channels * kh * kw
            end = (g + 1) * self.group_in_channels * kh * kw
            input_group = unfolded[:, :, start:end]

            if self.c_dim is not None and c is not None:
                if c.shape[-1] != self.c_dim:
                    raise ValueError(f"Expected c.shape[-1] = {self.c_dim}, got {c.shape[-1]}")
                c_expanded = c.view(N, 1, self.c_dim).expand(N, out_h * out_w, self.c_dim)
                c_weight_g = self.c_weight[
                    g * self.group_out_channels : (g + 1) * self.group_out_channels
                ]
                weight_g = torch.cat([weight_g, c_weight_g], dim=1)
                input_group = torch.cat([input_group, c_expanded], dim=-1)

            output_g = torch.matmul(input_group, weight_g.T)
            outputs.append(output_g)

        out = torch.cat(outputs, dim=-1)
        if self.bias is not None:
            out += self.bias.view(1, 1, -1)

        return out.permute(0, 2, 1).reshape(N, self.out_channels, out_h, out_w)

    @staticmethod
    def _to_pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        return (value, value) if isinstance(value, int) else value
