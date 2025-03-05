
import torch


# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _naive_sparseconv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,                 
                features: torch.Tensor,
                indices: torch.Tensor,
                HW_0, 
                HW_1, 
                HW_2,
                batch_size,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                stride: tuple = (1, 1),
                padding: tuple = (0, 0),
                dilation: tuple = (1, 1)):
        str_0, str_1, str_2 = stride
        pad_0, pad_1, pad_2 = padding
        dil_0, dil_1, dil_2 = dilation
        num_points = len(features)
        
        K, RS_0, RS_1, RS_2, C = weight.shape

        PQ_0 = (HW_0 + 2 * pad_0 - dil_0 * (RS_0 - 1) - 1) // str_0 + 1
        PQ_1 = (HW_1 + 2 * pad_1 - dil_1 * (RS_1 - 1) - 1) // str_1 + 1
        PQ_2 = (HW_2 + 2 * pad_2 - dil_2 * (RS_2 - 1) - 1) // str_2 + 1

        gather_idx = {}
        scatter_idx = {}
        out_indices = list()
        # print(f'indices: {indices}')
        for rs_0 in range(RS_0):
            for rs_1 in range(RS_0):
                for rs_2 in range(RS_2):
                    gather_idx[(rs_0, rs_1, rs_2)] = []
                    scatter_idx[(rs_0, rs_1, rs_2)] = []

                    for i_in in range(num_points):
                        p_in = indices[i_in]
                        bs, hw_0, hw_1, hw_2 = p_in
                        if ((hw_0 + pad_0 - rs_0 * dil_0) % str_0 == 0 and 
                            (hw_1 + pad_1 - rs_1 * dil_1) % str_1 == 0 and
                            (hw_2 + pad_2 - rs_2 * dil_2) % str_2 == 0):

                            pq_0 = ((hw_0 + pad_0 - rs_0 * dil_0) // str_0).item()
                            pq_1 = ((hw_1 + pad_1 - rs_1 * dil_1) // str_1).item()
                            pq_2 = ((hw_2 + pad_2 - rs_2 * dil_2) // str_2).item()
                            if pq_0 >= 0 and pq_0 < PQ_0 and pq_1 >= 0 and pq_1 < PQ_1 and pq_2 >= 0 and pq_2 < PQ_2:
                                p_out = [bs, pq_0, pq_1, pq_2]
                                gather_idx[(rs_0, rs_1, rs_2)].append(i_in)
                                if p_out not in out_indices:
                                    scatter_idx[(rs_0, rs_1, rs_2)].append(len(out_indices))
                                    out_indices.append(p_out)
                                else:
                                    idx = out_indices.index(p_out)
                                    scatter_idx[(rs_0, rs_1, rs_2)].append(idx)

        out_indices = torch.tensor(out_indices, dtype=indices.dtype, device=indices.device)
        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)
        for rs_0, rs_1, rs_2 in gather_idx.keys():
            w = weight[:, rs_0, rs_1, rs_2, :]
            p_in = features[gather_idx[(rs_0, rs_1, rs_2)]]
            p_out = p_in @ w.t()
            out_features[scatter_idx[(rs_0, rs_1, rs_2)]] += p_out
 
        if bias is not None:
            out_features += bias

        ctx.save_for_backward(features, weight)
        ctx.indices = indices
        ctx.HW_0 = HW_0
        ctx.HW_1 = HW_1
        ctx.HW_2 = HW_2
        ctx.out_indices = out_indices
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.PQ_0 = PQ_0
        ctx.PQ_1 = PQ_1
        ctx.PQ_2 = PQ_2
        ctx.RS_0 = RS_0
        ctx.RS_1 = RS_1
        ctx.RS_2 = RS_2
        if bias is None:
            ctx.bias_requires_grad = False
        else:
            ctx.bias_requires_grad = bias.requires_grad
        
        return out_features, out_indices, PQ_0, PQ_1, PQ_2
    
    @staticmethod
    def backward(ctx, dout_features: torch.Tensor, *args):
        # print(f'---: dout_features: {dout_features}')
        features, weight = ctx.saved_tensors
        indices = ctx.indices.tolist()
        HW_0 = ctx.HW_0
        HW_1 = ctx.HW_1
        HW_2 = ctx.HW_2
        PQ_0 = ctx.PQ_0
        PQ_1 = ctx.PQ_1
        PQ_2 = ctx.PQ_2
        RS_0 = ctx.RS_0
        RS_1 = ctx.RS_1
        RS_2 = ctx.RS_2

        dout_indices = ctx.out_indices.tolist()
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        str_0, str_1, str_2 = stride
        pad_0, pad_1, pad_2 = padding
        dil_0, dil_1, dil_2 = dilation

        dfeatures = None
        if features.requires_grad:
            dfeatures = torch.zeros_like(features)
            num_points = len(dout_features)
            gather_idx = {}
            scatter_idx = {}
            for rs_0 in range(RS_0):
                for rs_1 in range(RS_1):
                    for rs_2 in range(RS_2):
                        gather_idx[(rs_0, rs_1, rs_2)] = []
                        scatter_idx[(rs_0, rs_1, rs_2)] = []
                        
                        for i_dout in range(num_points):
                            p_dout = dout_indices[i_dout]

                            bs, pq_0, pq_1, pq_2 = p_dout
                            hw_0 = pq_0 * str_0 + rs_0 * dil_0 - pad_0
                            hw_1 = pq_1 * str_1 + rs_1 * dil_1 - pad_1
                            hw_2 = pq_2 * str_2 + rs_2 * dil_2 - pad_2
                            if hw_0 >= 0 and hw_0 < HW_0 and hw_1 >= 0 and hw_1 < HW_1 and hw_2 >= 0 and hw_2 < HW_2:
                                p_in = [bs, hw_0, hw_1, hw_2]
                                if p_in in indices:
                                    idx = indices.index(p_in)
                                    gather_idx[(rs_0, rs_1, rs_2)].append(i_dout)
                                    scatter_idx[(rs_0, rs_1, rs_2)].append(idx)

            # print(f'gather_idx: {gather_idx}')
            # print(f'scatter_idx: {scatter_idx}')

            for rs_0, rs_1, rs_2 in gather_idx.keys():
                w = weight[:, rs_0, rs_1, rs_2, :]
                p_dout = dout_features[gather_idx[(rs_0, rs_1, rs_2)]]
                p_din = p_dout @ w
                dfeatures[scatter_idx[(rs_0, rs_1, rs_2)]] += p_din

        dweight = None
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)
            num_points = len(dout_features)
            dout_gather_idx = {}
            input_gather_idx = {}
            for rs_0 in range(RS_0):
                for rs_1 in range(RS_1):
                    for rs_2 in range(RS_2):
                        dout_gather_idx[(rs_0, rs_1, rs_2)] = []
                        input_gather_idx[(rs_0, rs_1, rs_2)] = []
                        
                        for i_dout in range(num_points):
                            p_dout = dout_indices[i_dout]

                            bs, pq_0, pq_1, pq_2 = p_dout
                            hw_0 = pq_0 * str_0 + rs_0 * dil_0 - pad_0
                            hw_1 = pq_1 * str_1 + rs_1 * dil_1 - pad_1
                            hw_2 = pq_2 * str_2 + rs_2 * dil_2 - pad_2
                            if hw_0 >= 0 and hw_0 < HW_0 and hw_1 >= 0 and hw_1 < HW_1 and hw_2 >= 0 and hw_2 < HW_2:
                                p_in = [bs, hw_0, hw_1, hw_2]
                                if p_in in indices:
                                    idx = indices.index(p_in)
                                    dout_gather_idx[(rs_0, rs_1, rs_2)].append(i_dout)
                                    input_gather_idx[(rs_0, rs_1, rs_2)].append(idx)

            for rs_0, rs_1, rs_2 in dout_gather_idx.keys():
                p_dout = dout_features[dout_gather_idx[(rs_0, rs_1, rs_2)]]
                p_in = features[input_gather_idx[(rs_0, rs_1, rs_2)]]

                # print(f'p_dout.t(): {p_dout.t().shape}')
                # print(f'p_in: {p_in.shape}')
                dweight[:, rs_0, rs_1, rs_2, :] += p_dout.t() @ p_in

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, None, dweight, dbias, None, None, None, None


naive_sparseconv3d = _naive_sparseconv3d.apply
