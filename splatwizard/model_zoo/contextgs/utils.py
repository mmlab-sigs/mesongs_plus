import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.uniform import Uniform


from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import torchac
import math
import multiprocessing

anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
use_clamp = True
use_multiprocessor = False  # Always False plz. Not yet implemented for True.

def torch_unique_with_indices(tensor, dim=0,):
    """Return the unique elements of a tensor and their indices."""
    with torch.no_grad():
        unique, inverse_indices, counts = torch.unique(tensor, return_inverse=True, dim=dim, return_counts=True)
        # indices = torch.scatter_reduce(
        #     torch.zeros_like(unique, dtype=torch.long, device=tensor.device), 
        #     dim=0,
        #     index=inverse_indices,
        #     src=torch.arange(tensor.size(0), device=tensor.device),
        #     reduce="amin",
        #     include_self=False,
        # )
        # indices = torch.scatter_reduce(
        #     input=torch.arange(tensor.size(0), device=tensor.device), 
        #     dim=dim,
        #     index=inverse_indices,
        #     reduce="amin",
        # )
        # indices = torch.zeros_like(inverse_indices)
        # indices.scatter_(dim=dim, index=inverse_indices, src=torch.arange(tensor.size(0), device=tensor.device))[:unique.shape[0]]
        indices = torch.scatter_reduce(
            input=torch.zeros_like(unique[:,0], device=tensor.device, dtype=torch.double),
            src=torch.arange(tensor.size(0), device=tensor.device, dtype=torch.double), 
            dim=dim,
            index=inverse_indices,
            reduce="amin",
            include_self=False,
        ).long()
        return unique, inverse_indices, indices, counts

def get_binary_vxl_size(binary_vxl):
    # binary_vxl: {0, 1}
    # assert torch.unique(binary_vxl).mean() == 0.5
    ttl_num = binary_vxl.numel()

    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = pos_num / ttl_num  #  + 1e-6
    Pg = torch.clamp(Pg, min=1e-6, max=1-1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # Pg
    # print('binary_vxl:', Pg.item(), ttl_bit.item(), ttl_num, pos_num.item(), neg_num.item())
    return Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num


def multiprocess_encoder(lower, symbol, file_name, chunk_num=10):
    def enc_func(l, s, f, b_l, i):
        byte_stream = torchac.encode_float_cdf(l, s, check_input_bounds=True)
        with open(f, 'wb') as fout:
            fout.write(byte_stream)
        bit_len = len(byte_stream) * 8
        b_l[i] = bit_len
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    b_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        symbol_m = symbol[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=enc_func, args=(lower_m, symbol_m, file_name_m, b_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    ttl_bit_len = sum(list(b_list))
    return ttl_bit_len


def multiprocess_deoder(lower, file_name, chunk_num=10):
    def dec_func(l, f, o_l, i):
        with open(f, 'rb') as fin:
            byte_stream_d = fin.read()
        o = torchac.decode_float_cdf(l, byte_stream_d).to(torch.float32)
        o_l[i] = o
    encoding_len = lower.shape[0]
    chunk_len = int(math.ceil(encoding_len / chunk_num))
    processes = []
    manager = multiprocessing.Manager()
    output_list = manager.list([None] * chunk_num)
    for m_id in range(chunk_num):
        lower_m = lower[m_id * chunk_len:(m_id + 1) * chunk_len]
        file_name_m = file_name.replace('.b', f'_{m_id}.b')
        process = multiprocessing.Process(target=dec_func, args=(lower_m, file_name_m, output_list, m_id))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    output_list = torch.cat(list(output_list), dim=0).cuda()
    return output_list


def encoder_gaussian(x, mean, scale, Q, file_name=None):
    if file_name is not None: assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert x.shape == mean.shape == scale.shape == Q.shape
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(x.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    del samples
    del mean
    del scale
    del GD
    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    assert (x_int_round_idx.to(torch.int32) == x_int_round - min_value).all()
    # if x_int_round_idx.max() >= lower.shape[-1] - 1:  x_int_round_idx.max() exceed 65536 but to int6, that's why error
        # assert False

    if not use_multiprocessor:
        byte_stream = torchac.encode_float_cdf(lower.cpu(), x_int_round_idx.cpu(), check_input_bounds=True)
        if file_name is not None:
            with open(file_name, 'wb') as fout:
                fout.write(byte_stream)
        bit_len = len(byte_stream)*8
    else:
        bit_len = multiprocess_encoder(lower.cpu(), x_int_round_idx.cpu(), file_name)
    torch.cuda.empty_cache()
    return byte_stream, bit_len, min_value, max_value


def decoder_gaussian(mean, scale, Q, file_name=None, min_value=-100, max_value=100, bstream=None):
    if file_name is not None: assert file_name.endswith('.b')
    else: assert bstream is not None
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert mean.shape == scale.shape == Q.shape
    samples = torch.tensor(range(min_value, max_value+ 1 + 1)).to(
        torch.float).to(mean.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    if not use_multiprocessor:
        if file_name is not None:
            with open(file_name, 'rb') as fin:
                byte_stream_d = fin.read()
        else:
            byte_stream_d = bstream
        sym_out = torchac.decode_float_cdf(lower.cpu(), byte_stream_d).to(mean.device).to(torch.float32)
    else:
        sym_out = multiprocess_deoder(lower.cpu(), file_name, chunk_num=10).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    torch.cuda.empty_cache()
    return x


def encoder(x, p, file_name):
    x = x.detach().cpu()
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(((x+1)/2)).to(torch.int16)
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    # Number of bits taken by the stream
    bit_len = len(byte_stream) * 8
    # Write to a file.
    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len

def decoder(p, file_name):
    dvc = p.device
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    # Encode to bytestream.
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    # Read from a file.
    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()
    # Decode from bytestream.
    sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)
    sym_out = (sym_out * 2 - 1).to(torch.float32)
    return sym_out.to(dvc)


class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q):
        if use_clamp:
            input_min = -15_000 * Q
            input_max = +15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Quantize_anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, min_v, max_v):
        interval = ((max_v - min_v) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v
        # quantized_v = (anchors - min_v) // interval
        quantized_v = torch.div(anchors - min_v, interval, rounding_mode='floor')
        quantized_v = torch.clamp(quantized_v, 0, 2 ** anchor_round_digits - 1)
        anchors_q = quantized_v * interval + min_v
        return anchors_q, quantized_v
    
    @staticmethod
    def backward(ctx, grad_output, tmp):  # tmp is for quantized_v:)
        return grad_output, None, None

# class Entropy_gaussian_clamp(nn.Module):
#     def __init__(self, Q=1):
#         super(Entropy_gaussian_clamp, self).__init__()
#         self.Q = Q
#     def forward(self, x, mean, scale, Q=None):
#         if Q is None:
#             Q = self.Q
#         if use_clamp:
#             x_mean = x.mean()
#             x_min = x_mean - 15_000 * Q
#             x_max = x_mean + 15_000 * Q
#             x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
#         scale = torch.clamp(scale, min=1e-9)
#         m1 = torch.distributions.normal.Normal(mean, scale)
#         lower = m1.cdf(x - 0.5*Q)
#         upper = m1.cdf(x + 0.5*Q)
#         likelihood = torch.abs(upper - lower)
#         likelihood = Low_bound.apply(likelihood)
#         bits = -torch.log2(likelihood)
#         return bits
#
#
# class Entropy_gaussian(nn.Module):
#     def __init__(self, Q=1):
#         super(Entropy_gaussian, self).__init__()
#         self.Q = Q
#     def forward(self, x, mean, scale, Q=None, x_mean=None):
#         if Q is None:
#             Q = self.Q
#         if use_clamp:
#             if x_mean is None:
#                 x_mean = x.mean()
#             x_min = x_mean - 15_000 * Q
#             x_max = x_mean + 15_000 * Q
#             x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
#         scale = torch.clamp(scale, min=1e-9)
#         m1 = torch.distributions.normal.Normal(mean, scale)
#         lower = m1.cdf(x - 0.5*Q)
#         upper = m1.cdf(x + 0.5*Q)
#         likelihood = torch.abs(upper - lower)
#         likelihood = Low_bound.apply(likelihood)
#         bits = -torch.log2(likelihood)
#         return bits
#
#
# class Entropy_bernoulli(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x, p):
#         # p = torch.sigmoid(p)
#         p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
#         pos_mask = (1 + x) / 2.0  # 1 -> 1, -1 -> 0
#         neg_mask = (1 - x) / 2.0  # -1 -> 1, 1 -> 0
#         pos_prob = p
#         neg_prob = 1 - p
#         param_bit = -torch.log2(pos_prob) * pos_mask + -torch.log2(neg_prob) * neg_mask
#         return param_bit
#
#
# class Entropy_factorized(nn.Module):
#     def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6,
#                  tail_mass=1e-9, optimize_integer_offset=True, Q=1):
#         super(Entropy_factorized, self).__init__()
#         self.filters = tuple(int(t) for t in filters)
#         self.init_scale = float(init_scale)
#         self.likelihood_bound = float(likelihood_bound)
#         self.tail_mass = float(tail_mass)
#         self.optimize_integer_offset = bool(optimize_integer_offset)
#         self.Q = Q
#         if not 0 < self.tail_mass < 1:
#             raise ValueError(
#                 "`tail_mass` must be between 0 and 1")
#         filters = (1,) + self.filters + (1,)
#         scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
#         self._matrices = nn.ParameterList([])
#         self._bias = nn.ParameterList([])
#         self._factor = nn.ParameterList([])
#         for i in range(len(self.filters) + 1):
#             init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
#             self.matrix = nn.Parameter(torch.FloatTensor(
#                 channel, filters[i + 1], filters[i]))
#             self.matrix.data.fill_(init)
#             self._matrices.append(self.matrix)
#             self.bias = nn.Parameter(
#                 torch.FloatTensor(channel, filters[i + 1], 1))
#             noise = np.random.uniform(-0.5, 0.5, self.bias.size())
#             noise = torch.FloatTensor(noise)
#             self.bias.data.copy_(noise)
#             self._bias.append(self.bias)
#             if i < len(self.filters):
#                 self.factor = nn.Parameter(
#                     torch.FloatTensor(channel, filters[i + 1], 1))
#                 self.factor.data.fill_(0.0)
#                 self._factor.append(self.factor)
#
#     def _logits_cumulative(self, logits, stop_gradient):
#         for i in range(len(self.filters) + 1):
#             matrix = nnf.softplus(self._matrices[i])
#             if stop_gradient:
#                 matrix = matrix.detach()
#             # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
#             logits = torch.matmul(matrix, logits)
#             bias = self._bias[i]
#             if stop_gradient:
#                 bias = bias.detach()
#             logits += bias
#             if i < len(self._factor):
#                 factor = nnf.tanh(self._factor[i])
#                 if stop_gradient:
#                     factor = factor.detach()
#                 logits += factor * nnf.tanh(logits)
#         return logits
#
#     def forward(self, x, Q=None):
#         # x: [N, C], quantized
#         if Q is None:
#             Q = self.Q
#         else:
#             Q = Q.permute(1, 0).contiguous()
#         x = x.permute(1, 0).contiguous()  # [C, N]
#         # print('dqwdqwdqwdqwfqwf:', x.shape, Q.shape)
#         lower = self._logits_cumulative(x - 0.5*(1/Q), stop_gradient=False)
#         upper = self._logits_cumulative(x + 0.5*(1/Q), stop_gradient=False)
#         sign = -torch.sign(torch.add(lower, upper))
#         sign = sign.detach()
#         likelihood = torch.abs(
#             nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
#         likelihood = Low_bound.apply(likelihood)
#         bits = -torch.log2(likelihood)  # [C, N]
#         bits = bits.permute(1, 0).contiguous()
#         return bits
#
#
# class Low_bound(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         x = torch.clamp(x, min=1e-6)
#         return x
#
#     @staticmethod
#     def backward(ctx, g):
#         x, = ctx.saved_tensors
#         grad1 = g.clone()
#         grad1[x < 1e-6] = 0
#         pass_through_if = np.logical_or(
#             x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
#         t = torch.Tensor(pass_through_if+0.0).cuda()
#         return grad1 * t


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        #b = np.random.uniform(-1,1)
        b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g