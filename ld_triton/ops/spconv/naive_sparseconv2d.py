
import torch


# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _naive_sparseconv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,                 
                features: torch.Tensor,
                indices: torch.Tensor,
                H,
                W,
                batch_size,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                stride: int = (1, 1),
                padding: int = (0, 0),
                dilation: int = (1, 1),
                memory_format: str = 'channel_last'):
        assert memory_format == 'channel_last'
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        num_points = len(features)
        
        if memory_format == 'channel_first':
            K, C, R, S = weight.shape
        elif memory_format == 'channel_last':
            K, R, S, C = weight.shape
        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        gather_idx = {}
        scatter_idx = {}
        out_indices = list()
        # print(f'indices: {indices}')
        for r in range(R):
            for s in range(S):
                gather_idx[(r, s)] = []
                scatter_idx[(r, s)] = []

                for i_in in range(num_points):
                    p_in = indices[i_in]
                    bs, h, w = p_in
                    if ((h + pad_h - r * dil_h) % str_h == 0 and 
                        (w + pad_w - s * dil_w) % str_w == 0):

                        p = ((h + pad_h - r * dil_h) // str_h).item()
                        q = ((w + pad_w - s * dil_w) // str_w).item()
                        if p >= 0 and p < P and q >= 0 and q < Q:
                            p_out = [bs, p, q]
                            gather_idx[(r, s)].append(i_in)
                            if p_out not in out_indices:
                                scatter_idx[(r, s)].append(len(out_indices))
                                out_indices.append(p_out)
                            else:
                                idx = out_indices.index(p_out)
                                scatter_idx[(r, s)].append(idx)

        # print(f'gather_idx: {gather_idx}')
        # print(f'scatter_idx: {scatter_idx}')
        out_indices = torch.tensor(out_indices, dtype=indices.dtype, device=indices.device)
        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)
        for r, s in gather_idx.keys():
            if memory_format == 'channel_first':
                w = weight[:, :, r, s]
            elif memory_format == 'channel_last':
                w = weight[:, r, s, :]
            else:
                raise ValueError(f"memory_format: {memory_format} is not supported, only support 'channel_first' or 'channel_last'")
            p_in = features[gather_idx[(r, s)]]
            p_out = p_in @ w.t()
            out_features[scatter_idx[(r, s)]] += p_out
 
        if bias is not None:
            out_features += bias

        ctx.save_for_backward(features, weight)
        ctx.indices = indices
        ctx.H = H
        ctx.W = W
        ctx.out_indices = out_indices
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.P = P
        ctx.Q = Q
        ctx.R = R
        ctx.S = S
        ctx.memory_format = memory_format
        if bias is None:
            ctx.bias_requires_grad = False
        else:
            ctx.bias_requires_grad = bias.requires_grad
        
        return out_features, out_indices, P, Q
    
    @staticmethod
    def backward(ctx, dout_features: torch.Tensor, *args):
        # print(f'---: dout_features: {dout_features}')
        features, weight = ctx.saved_tensors
        indices = ctx.indices.tolist()
        H = ctx.H
        W = ctx.W
        P = ctx.P
        Q = ctx.Q
        R = ctx.R
        S = ctx.S

        dout_indices = ctx.out_indices.tolist()
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        memory_format = ctx.memory_format

        dfeatures = None
        if features.requires_grad:
            dfeatures = torch.zeros_like(features)
            num_points = len(dout_features)
            gather_idx = {}
            scatter_idx = {}
            for r in range(R):
                for s in range(S):
                    gather_idx[(r, s)] = []
                    scatter_idx[(r, s)] = []
                    
                    for i_dout in range(num_points):
                        p_dout = dout_indices[i_dout]

                        bs, p, q = p_dout
                        h = p * str_h + r * dil_h - pad_h
                        w = q * str_w + s * dil_w - pad_w

                        if h >= 0 and h < H and w >= 0 and w < W:
                            p_in = [bs, h, w]
                            if p_in in indices:
                                idx = indices.index(p_in)
                                gather_idx[(r, s)].append(i_dout)
                                scatter_idx[(r, s)].append(idx)

            # print(f'gather_idx: {gather_idx}')
            # print(f'scatter_idx: {scatter_idx}')

            for r, s in gather_idx.keys():
                if memory_format == 'channel_first':
                    w = weight[:, :, r, s]
                elif memory_format == 'channel_last':
                    w = weight[:, r, s, :]
                else:
                    raise ValueError(f"memory_format: {memory_format} is not supported, only support 'channel_first' or 'channel_last'")

                p_dout = dout_features[gather_idx[(r, s)]]
                p_din = p_dout @ w
                dfeatures[scatter_idx[(r, s)]] += p_din

        dweight = None
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)
            num_points = len(dout_features)
            dout_gather_idx = {}
            input_gather_idx = {}
            for r in range(R):
                for s in range(S):
                    dout_gather_idx[(r, s)] = []
                    input_gather_idx[(r, s)] = []
                    
                    for i_dout in range(num_points):
                        p_dout = dout_indices[i_dout]

                        bs, p, q = p_dout
                        h = p * str_h + r * dil_h - pad_h
                        w = q * str_w + s * dil_w - pad_w

                        if h >= 0 and h < H and w >= 0 and w < W:
                            p_in = [bs, h, w]
                            if p_in in indices:
                                idx = indices.index(p_in)
                                dout_gather_idx[(r, s)].append(i_dout)
                                input_gather_idx[(r, s)].append(idx)

            for r, s in dout_gather_idx.keys():
                p_dout = dout_features[dout_gather_idx[(r, s)]]
                p_in = features[input_gather_idx[(r, s)]]

                # print(f'p_dout.t(): {p_dout.t().shape}')
                # print(f'p_in: {p_in.shape}')
                if memory_format == 'channel_first':
                    dweight[:, :, r, s] += p_dout.t() @ p_in
                elif memory_format == 'channel_last':
                    dweight[:, r, s, :] += p_dout.t() @ p_in
                else:
                    raise ValueError(f"memory_format: {memory_format} is not supported, only support 'channel_first' or 'channel_last'")

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, dweight, dbias, None, None, None, None


naive_sparseconv2d = _naive_sparseconv2d.apply
