
import torch
import time


# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/nn/SubmConv2D_cn.html#submconv2d
# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _naive_submconv3d(torch.autograd.Function):
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
                stride: int = (1, 1, 1),
                padding: int = (0, 0, 0),
                dilation: int = (1, 1, 1),
                memory_format: str = 'channel_last'):
        _fwd_debug = True
        assert memory_format == 'channel_last'
        # why? Adapted from spconv
        str_hw_0, str_hw_1, str_hw_2 = 1, 1, 1

        if memory_format == 'channel_first':
            K, C, RS_0, RS_1, RS_2 = weight.shape
        elif memory_format == 'channel_last':
            K, RS_0, RS_1, RS_2, C = weight.shape

        assert RS_0 % 2 == 1, "subm only support odd ksize(RS_0={RS_0})"
        assert RS_1 % 2 == 1, "subm only support odd ksize(RS_1={RS_1})"
        assert RS_2 % 2 == 1, "subm only support odd ksize(RS_2={RS_2})"

        dil_hw_0, dil_hw_1, dil_hw_2 = dilation
        pad_hw_0 = (RS_0 // 2) * dil_hw_0
        pad_hw_1 = (RS_1 // 2) * dil_hw_1
        pad_hw_2 = (RS_2 // 2) * dil_hw_2
        
        num_points = len(features)

        PQ_0 = HW_0
        PQ_1 = HW_1
        PQ_2 = HW_2

        gather_idx = {}
        scatter_idx = {}
        out_indices_list = indices.tolist()
        out_indices_dict = {tuple(value): index for index, value in enumerate(out_indices_list)}
        # print(f'indices: {indices}')
        center_rs_0 = RS_0 // 2
        center_rs_1 = RS_1 // 2
        center_rs_2 = RS_2 // 2
        if _fwd_debug:
            start_cpu = time.time()
        RS = RS_0 * RS_1 * RS_2

        for rs in range(RS // 2 + 1):
            rs_0 = rs // (RS_1 * RS_2)
            rs_1 = (rs % (RS_1 * RS_2)) // RS_2
            rs_2 = rs % RS_2
            gather_idx[(rs_0, rs_1, rs_2)] = []
            scatter_idx[(rs_0, rs_1, rs_2)] = []
            if rs_0 == center_rs_0 and rs_1 == center_rs_1 and rs_2 == center_rs_2:
                gather_idx[(rs_0, rs_1, rs_2)] = [i for i in range(num_points)]
                scatter_idx[(rs_0, rs_1, rs_2)] = [i for i in range(num_points)]
            else:
                gather_idx[(rs_0, rs_1, rs_2)] = [-1 for _ in range(num_points)]
                scatter_idx[(rs_0, rs_1, rs_2)] = [-1 for _ in range(num_points)]
                gather_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)] = [-1 for _ in range(num_points)]
                scatter_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)] = [-1 for _ in range(num_points)]
            for i_in in range(num_points):
                if rs_0 == center_rs_0 and rs_1 == center_rs_1 and rs_2 == center_rs_2:
                    pass
                else:
                    p_in = out_indices_list[i_in]
                    bs, hw_0, hw_1, hw_2 = p_in
                    center_hw_0 = hw_0 + (center_rs_0 - rs_0) * dil_hw_0
                    center_hw_1 = hw_1 + (center_rs_1 - rs_1) * dil_hw_1
                    center_hw_2 = hw_2 + (center_rs_2 - rs_2) * dil_hw_2
                    center_p = tuple([bs, center_hw_0, center_hw_1, center_hw_2])
                    # print(f'center_p: {center_p}')
                    if ((hw_0 + pad_hw_0 - rs_0 * dil_hw_0) % str_hw_0 == 0 and 
                        (hw_1 + pad_hw_1 - rs_1 * dil_hw_1) % str_hw_1 == 0 and
                        (hw_2 + pad_hw_2 - rs_2 * dil_hw_2) % str_hw_2 == 0 and
                        center_p in out_indices_dict):
                        pq_0 = (hw_0 + pad_hw_0 - rs_0 * dil_hw_0) // str_hw_0
                        pq_1 = (hw_1 + pad_hw_1 - rs_1 * dil_hw_1) // str_hw_1
                        pq_2 = (hw_2 + pad_hw_2 - rs_2 * dil_hw_2) // str_hw_2
                        if pq_0 >= 0 and pq_0 < PQ_0 and pq_1 >= 0 and pq_1 < PQ_1 and pq_2 >= 0 and pq_2 < PQ_2:
                            assert center_hw_0 == pq_0
                            assert center_hw_1 == pq_1
                            assert center_hw_2 == pq_2
                            p_out = tuple([bs, pq_0, pq_1, pq_2])
                            idx = out_indices_dict[p_out]
                            gather_idx[(rs_0, rs_1, rs_2)][i_in] = i_in
                            gather_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)][i_in] = idx

                            scatter_idx[(rs_0, rs_1, rs_2)][i_in] = idx
                            scatter_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)][i_in] = i_in
            gather_idx[(rs_0, rs_1, rs_2)] = [x for x in gather_idx[(rs_0, rs_1, rs_2)] if x != -1]
            gather_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)] = [x for x in gather_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)] if x != -1]
            scatter_idx[(rs_0, rs_1, rs_2)] = [x for x in scatter_idx[(rs_0, rs_1, rs_2)] if x != -1]
            scatter_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)] = [x for x in scatter_idx[(RS_0 - 1 - rs_0, RS_1 - 1 - rs_1, RS_2 - 1 - rs_2)] if x != -1]


        if _fwd_debug:
            end_cpu = time.time()
            cpu_time = (end_cpu - start_cpu) * 1000
            print(f'cpu_time: {cpu_time}ms')

        out_indices = indices
        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)
        if _fwd_debug:
            start_cuda = torch.cuda.Event(enable_timing=True)
            end_cuda = torch.cuda.Event(enable_timing=True)
            start_cuda.record()
        for rs_0, rs_1, rs_2 in gather_idx.keys():
            if memory_format == 'channel_first':
                w = weight[:, :, rs_0, rs_1, rs_2]
            elif memory_format == 'channel_last':
                w = weight[:, rs_0, rs_1, rs_2, :]
            else:
                raise ValueError(f"memory_format: {memory_format} is not supported, only support 'channel_first' or 'channel_last'")
            p_in = features[gather_idx[(rs_0, rs_1, rs_2)]]
            p_out = p_in @ w.t()
            out_features[scatter_idx[(rs_0, rs_1, rs_2)]] += p_out
 
        if bias is not None:
            out_features += bias
        if _fwd_debug:
            end_cuda.record()
            torch.cuda.synchronize()
            cuda_time = start_cuda.elapsed_time(end_cuda)
            print(f'cuda_time: {cuda_time}ms')
            print(f'cuda_time and cpu_time: {cuda_time + cpu_time}')

        ctx.save_for_backward(features, weight)
        ctx.indices = indices
        ctx.HW_0 = HW_0
        ctx.HW_1 = HW_1
        ctx.HW_2 = HW_2
        ctx.out_indices = out_indices
        ctx.str_hw_0 = str_hw_0
        ctx.str_hw_1 = str_hw_1
        ctx.str_hw_2 = str_hw_2
        ctx.pad_hw_0 = pad_hw_0
        ctx.pad_hw_1 = pad_hw_1
        ctx.pad_hw_2 = pad_hw_2
        ctx.dil_hw_0 = dil_hw_0
        ctx.dil_hw_1 = dil_hw_1
        ctx.dil_hw_2 = dil_hw_2
        ctx.PQ_0 = PQ_0
        ctx.PQ_1 = PQ_1
        ctx.PQ_2 = PQ_2
        ctx.RS_0 = RS_0
        ctx.RS_1 = RS_1
        ctx.RS_2 = RS_2
        ctx.memory_format = memory_format
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
        str_hw_0 = ctx.str_hw_0
        str_hw_1 = ctx.str_hw_1
        str_hw_2 = ctx.str_hw_2
        pad_hw_0 = ctx.pad_hw_0
        pad_hw_1 = ctx.pad_hw_1
        pad_hw_2 = ctx.pad_hw_2
        dil_hw_0 = ctx.dil_hw_0
        dil_hw_1 = ctx.dil_hw_1
        dil_hw_2 = ctx.dil_hw_2
        memory_format = ctx.memory_format

        dfeatures = None
        if features.requires_grad:
            dfeatures = torch.zeros_like(features)
            num_points = len(dout_features)
            gather_idx = {}
            scatter_idx = {}
            center_rs_0 = RS_0 // 2
            center_rs_1 = RS_1 // 2
            center_rs_2 = RS_2 // 2

            gather_idx[(center_rs_0, center_rs_1, center_rs_2)] = []
            scatter_idx[(center_rs_0, center_rs_1, center_rs_2)] = []
            center_idx = []
            
            for i_dout in range(num_points):
                p_dout = dout_indices[i_dout]

                bs, pq_0, pq_1, pq_2 = p_dout
                hw_0 = pq_0 * str_hw_0 + center_rs_0 * dil_hw_0 - pad_hw_0
                hw_1 = pq_1 * str_hw_1 + center_rs_1 * dil_hw_1 - pad_hw_1
                hw_2 = pq_2 * str_hw_2 + center_rs_2 * dil_hw_2 - pad_hw_2

                if hw_0 >= 0 and hw_0 < HW_0 and hw_1 >= 0 and hw_1 < HW_1 and hw_2 >= 0 and hw_2 < HW_2:
                    p_in = [bs, hw_0, hw_1, hw_2]
                    center_idx.append(p_in)
                    if p_in in indices:
                        idx = indices.index(p_in)
                        gather_idx[(center_rs_0, center_rs_1, center_rs_2)].append(i_dout)
                        scatter_idx[(center_rs_0, center_rs_1, center_rs_2)].append(idx) 
            
            # print(f'center_idx: {center_idx}')

            for rs_0 in range(RS_0):
                for rs_1 in range(RS_1):
                    for rs_2 in range(RS_2):
                        if rs_0 == center_rs_0 and rs_1 == center_rs_1 and rs_2 == center_rs_2:
                            continue

                        gather_idx[(rs_0, rs_1, rs_2)] = []
                        scatter_idx[(rs_0, rs_1, rs_2)] = []

                        for i_dout in range(num_points):
                            p_dout = dout_indices[i_dout]

                            bs, pq_0, pq_1, pq_2 = p_dout
                            hw_0 = pq_0 * str_hw_0 + rs_0 * dil_hw_0 - pad_hw_0
                            hw_1 = pq_1 * str_hw_1 + rs_1 * dil_hw_1 - pad_hw_1
                            hw_2 = pq_2 * str_hw_2 + rs_2 * dil_hw_2 - pad_hw_2
                            
                            center_hw_0 = hw_0 + (center_rs_0 - rs_0) * dil_hw_0
                            center_hw_1 = hw_1 + (center_rs_1 - rs_1) * dil_hw_1
                            center_hw_2 = hw_2 + (center_rs_2 - rs_2) * dil_hw_2
                            center_p = [bs, center_hw_0, center_hw_1, center_hw_2]

                            if hw_0 >= 0 and hw_0 < HW_0 and hw_1 >= 0 and hw_1 < HW_1 and hw_2 >= 0 and hw_2 < HW_2 and center_p in center_idx:
                                p_in = [bs, hw_0, hw_1, hw_2]
                                if p_in in indices:
                                    idx = indices.index(p_in)
                                    gather_idx[(rs_0, rs_1, rs_2)].append(i_dout)
                                    scatter_idx[(rs_0, rs_1, rs_2)].append(idx)

            # print(f'gather_idx: {gather_idx}')
            # print(f'scatter_idx: {scatter_idx}')

            for rs_0, rs_1, rs_2 in gather_idx.keys():
                if memory_format == 'channel_first':
                    w = weight[:, :, rs_0, rs_1, rs_2]
                elif memory_format == 'channel_last':
                    w = weight[:, rs_0, rs_1, rs_2, :]
                else:
                    raise ValueError(f"memory_format: {memory_format} is not supported, only support 'channel_first' or 'channel_last'")

                p_dout = dout_features[gather_idx[(rs_0, rs_1, rs_2)]]
                p_din = p_dout @ w
                dfeatures[scatter_idx[(rs_0, rs_1, rs_2)]] += p_din

        dweight = None
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)
            num_points = len(dout_features)
            dout_gather_idx = {}
            input_gather_idx = {}
            
            center_rs_0 = RS_0 // 2
            center_rs_1 = RS_1 // 2
            center_rs_2 = RS_2 // 2

            dout_gather_idx[(center_rs_0, center_rs_1, center_rs_2)] = []
            input_gather_idx[(center_rs_0, center_rs_1, center_rs_2)] = []
            center_idx = []
            
            for i_dout in range(num_points):
                p_dout = dout_indices[i_dout]

                bs, pq_0, pq_1, pq_2 = p_dout
                hw_0 = pq_0 * str_hw_0 + center_rs_0 * dil_hw_0 - pad_hw_0
                hw_1 = pq_1 * str_hw_1 + center_rs_1 * dil_hw_1 - pad_hw_1
                hw_2 = pq_2 * str_hw_2 + center_rs_2 * dil_hw_2 - pad_hw_2

                if hw_0 >= 0 and hw_0 < HW_0 and hw_1 >= 0 and hw_1 < HW_1 and hw_2 >= 0 and hw_2 < HW_2:
                    p_in = [bs, hw_0, hw_1, hw_2]
                    if p_in in indices:
                        center_idx.append(p_in)
                        idx = indices.index(p_in)
                        dout_gather_idx[(center_rs_0, center_rs_1, center_rs_2)].append(i_dout)
                        input_gather_idx[(center_rs_0, center_rs_1, center_rs_2)].append(idx)

            # print(f'center_idx: {center_idx}')

            for rs_0 in range(RS_0):
                for rs_1 in range(RS_1):
                    for rs_2 in range(RS_2):
                        if rs_0 == center_rs_0 and rs_1 == center_rs_1 and rs_2 == center_rs_2:
                            continue

                        dout_gather_idx[(rs_0, rs_1, rs_2)] = []
                        input_gather_idx[(rs_0, rs_1, rs_2)] = []
                        
                        for i_dout in range(num_points):
                            p_dout = dout_indices[i_dout]
                            bs, pq_0, pq_1, pq_2 = p_dout

                            hw_0 = pq_0 * str_hw_0 + rs_0 * dil_hw_0 - pad_hw_0
                            hw_1 = pq_1 * str_hw_1 + rs_1 * dil_hw_1 - pad_hw_1
                            hw_2 = pq_2 * str_hw_2 + rs_2 * dil_hw_2 - pad_hw_2
                            
                            center_hw_0 = hw_0 + (center_rs_0 - rs_0) * dil_hw_0
                            center_hw_1 = hw_1 + (center_rs_1 - rs_1) * dil_hw_1
                            center_hw_2 = hw_2 + (center_rs_2 - rs_2) * dil_hw_2
                            center_p = [bs, center_hw_0, center_hw_1, center_hw_2]

                            if hw_0 >= 0 and hw_0 < HW_0 and hw_1 >= 0 and hw_1 < HW_1 and hw_2 >= 0 and hw_2 < HW_2 and center_p in center_idx:
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
                if memory_format == 'channel_first':
                    dweight[:, :, rs_0, rs_1, rs_2] += p_dout.t() @ p_in
                elif memory_format == 'channel_last':
                    dweight[:, rs_0, rs_1, rs_2, :] += p_dout.t() @ p_in
                else:
                    raise ValueError(f"memory_format: {memory_format} is not supported, only support 'channel_first' or 'channel_last'")

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, None, dweight, dbias, None, None, None, None


naive_submconv3d = _naive_submconv3d.apply
