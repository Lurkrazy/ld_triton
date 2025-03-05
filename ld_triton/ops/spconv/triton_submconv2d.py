
import torch
import triton
import triton.language as tl

from ld_triton.ops.spconv.triton_hash import LinearHashTableSplit

def build_subm_conv_hash_table(indices: torch.Tensor, H, W):
    kv_size = indices.shape[0]
    indices = indices.to(torch.int32)
    key = indices[:, 0] * H * W + \
            indices[:, 1] * W + \
            indices[:, 2]
    val = torch.arange(0, kv_size, dtype=torch.int32, device=indices.device)
    
    table = LinearHashTableSplit(kv_size, rate = 2.0)
    table.insert(key.to(torch.uint32), val.to(torch.uint32))
    return table


@triton.jit
def linear_hash_table_lookup_offset_impl(table_key_ptr, table_val_ptr, key, kv_size, hash_size, empty_key_uint, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    kv_mask = offsets < kv_size
    # hash
    key ^= key >> 16
    key *= 0x85ebca6b
    key ^= key >> 13
    key *= 0xc2b2ae35
    key ^= key >> 16
    # prev = tl.atomic_cas(pointer, cmp, val), if pointer address is illegal, the prev = 0,
    # when the key = 0, hash(key) = 0, it will cause some bug
    # key = 0 or key = 0xFFFFFFFF the hash(k) = 0
    # we shift the key by 1, so the key = 0 will not cause the hash(k) = 0
    key += 1

    key = tl.where(kv_mask, key, empty_key_uint).to(tl.uint32)
    
    slot = tl.where(kv_mask, key.to(tl.uint32) % hash_size.to(tl.uint32), hash_size)
    key_target = tl.load(table_key_ptr + slot, mask=kv_mask, other=empty_key_uint + 1)
    # find a slot, this slot has a key is the same as the current key, break
    mask_1 = (key_target == key) & kv_mask
    # find a slot, this slot has a key is the same as empty_key_uint, break
    mask_2 = (key_target == empty_key_uint) & kv_mask
    # find a slot, this slot has a key is diffrent of current key and empty_key_uint, continue to find the next slot
    mask_3 = (key_target != key) & (key_target != empty_key_uint) & (key_target != empty_key_uint+ 1 ) & kv_mask
    # find a slot, key is the same as the current key or empty_key_uint, record the slot is used
    mask = mask_1 | mask_2
    # there is no slot match mask_1 or mask_2 or mask_3 break
    broke_mask = mask_1 | mask_2 | mask_3

    val = tl.load(table_val_ptr + slot, mask=mask_1, other=empty_key_uint)

    is_broke = tl.sum(broke_mask.to(tl.int32), axis=0)
    total_mask = tl.where(mask, False, False)

    res_val = tl.full((BLOCK_SIZE, ), empty_key_uint, dtype=tl.uint32)
    res_val = tl.where(mask_1, val, res_val)
    res_slot = tl.full((BLOCK_SIZE, ), -1, dtype=tl.int32)
    res_slot = tl.where(mask_1, slot, res_slot)

    while is_broke != 0:
        prev_mask = mask
        # record the slot is used
        total_mask = total_mask | prev_mask
        slot = tl.where(kv_mask & ~total_mask, (slot + 1) % hash_size.to(tl.uint32), hash_size)
        # for an example
        # table key: 1, 2, 3, 4, 5, 6, 7, 8
        # find  key: 9, 9, 3, 4, 5, 6, 7, 8
        #           2, 3, 4, 5, 6, 7, 8, 9
        # key_target other value is empty_key_uint + 1
        key_target = tl.load(table_key_ptr + slot, mask=(kv_mask & ~total_mask), other=empty_key_uint + 1)
        # find a slot, this slot has a key is the same as the current key, break
        mask_1 = (key_target == key) & kv_mask & ~total_mask
        # find a slot, this slot has a key is the same as empty_key_uint, break
        mask_2 = (key_target == empty_key_uint) & kv_mask & ~total_mask
        # find a slot, this slot has a key is diffrent of current key and empty_key_uint, continue to find the next slot
        mask_3 = (key_target != key) & (key_target != empty_key_uint) & (key_target != empty_key_uint+ 1 ) & kv_mask & ~total_mask
        # find a slot, key is the same as the current key or empty_key_uint, record the slot is used
        mask = mask_1 | mask_2

        # find a slot, key is the same as the current key, store the value
        is_store = tl.sum(mask_1.to(tl.int32), axis=0)
        mask_broke = mask_1 | mask_2 | mask_3
        is_broke = tl.sum(mask_broke.to(tl.int32), axis=0)
        if is_store != 0:
            val = tl.load(table_val_ptr + slot, mask=mask_1, other=empty_key_uint)
            res_val = tl.where(mask_1, val, res_val)
            res_slot = tl.where(mask_1, slot, res_slot)
        # these code is used to debug
        is_store_2 = tl.sum(mask_2.to(tl.int32), axis=0)
        if is_store_2 != 0:
            res_slot = tl.where(mask_2, -1, res_slot)

    return res_slot, res_val


@triton.jit
def calc_subm_conv_indices_kernel(
    table_key_ptr, table_val_ptr, hash_size, empty_key_uint,
    indices_ptr,
    gather_idx_ptr, scatter_idx_ptr, indice_num_per_loc,
    num_indices_in, 
    RS,
    H, W,
    P, Q,
    R, S,
    str_h, str_w,
    pad_h, pad_w,
    dil_h, dil_w,
    BLOCK_SIZE: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1) 
    if pid_1 == RS // 2:
        offsets = pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        gather_idx_offs = pid_1 * num_indices_in + offsets
        scatter_idx_offs = pid_1 * num_indices_in + offsets
        mask = offsets < num_indices_in
        tl.store(gather_idx_ptr + gather_idx_offs, offsets, mask=mask)
        tl.store(scatter_idx_ptr + scatter_idx_offs, offsets, mask=mask)
    else:
        r = pid_1 // S
        s = pid_1 % S
        center_r = R // 2
        center_s = S // 2

        center_offsets = (pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * 3
        bs = tl.load(indices_ptr + center_offsets + 0, center_offsets + 0 < 3 * num_indices_in)
        h = tl.load(indices_ptr + center_offsets + 1, center_offsets + 1 < 3 * num_indices_in)
        w = tl.load(indices_ptr + center_offsets + 2, center_offsets + 2 < 3 * num_indices_in)
        center_h = h + (center_r - r) * dil_h
        center_w = w + (center_s - s) * dil_w
        center_p_idx = bs * P * Q + center_h * Q + center_w
        center_p_slot, _ = linear_hash_table_lookup_offset_impl(table_key_ptr, table_val_ptr, 
                                                                center_p_idx,
                                                                BLOCK_SIZE, hash_size, 
                                                                empty_key_uint, BLOCK_SIZE)
        mask_mod_h = (h + pad_h - r * dil_h) % str_h == 0
        mask_mod_w = (w + pad_w - s * dil_w) % str_w == 0
        mask_mod_center = center_p_slot != -1
        mask_mod = mask_mod_h & mask_mod_w & mask_mod_center
        is_mod_broke = tl.sum(mask_mod.to(tl.int32), axis=0)
        if is_mod_broke != 0:
            p = (h + pad_h - r * dil_h) // str_h
            q = (w + pad_w - s * dil_w) // str_w
            
            mask_p = p >= 0 and p < P
            mask_q = q >= 0 and q < Q
            mask_pq = mask_p & mask_q

            is_pq_broke = tl.sum(mask_pq.to(tl.int32), axis=0)
            if is_pq_broke != 0:
                p_out_idx = bs * P * Q+ p * Q + q
                p_out_slot, idx = linear_hash_table_lookup_offset_impl(table_key_ptr, table_val_ptr, 
                                                                                p_out_idx,
                                                                                BLOCK_SIZE, hash_size, 
                                                                                empty_key_uint, BLOCK_SIZE)
                idx = tl.where(p_out_slot != -1, idx, -1)
                i_in = pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                gather_idx_offs = pid_1 * num_indices_in + i_in
                scatter_idx_offs = pid_1 * num_indices_in + i_in
                
                rotate_gather_idx_offs = (RS - 1 - pid_1) * num_indices_in + i_in
                rotate_scatter_idx_offs = (RS - 1 - pid_1) * num_indices_in + i_in

                filter_mask = mask_mod & mask_pq & (p_out_slot != -1) & (i_in < num_indices_in)

                # in[store_mask]
                ss = tl.sum(filter_mask.to(tl.int32), axis=0)
                # store_mask = i_in < num_indices_in
                # store_i_in = tl.where(filter_mask, i_in, -1)
                # store_i_in = tl.sort(store_i_in, dim=0, descending=True)
                # store_i_in = tl.where(store_mask, store_i_in, 200)
                # store_idx = tl.where(filter_mask, idx, -1)
                # store_idx = tl.sort(store_idx, dim=0, descending=True)
                # x = tl.where(x_mask, x, -1)
                # x = tl.sort(input_tensor, dim=0, descending=True)

                # https://live.nvidia.cn/gtc-od/attachments/CNS20315.pdf

                tl.store(gather_idx_ptr + gather_idx_offs, i_in, mask=filter_mask)
                tl.store(gather_idx_ptr + rotate_gather_idx_offs, idx, mask=filter_mask)
                tl.store(scatter_idx_ptr + scatter_idx_offs, idx, mask=filter_mask)
                tl.store(scatter_idx_ptr + rotate_scatter_idx_offs, i_in, mask=filter_mask)

                tl.atomic_add(indice_num_per_loc + pid_1, ss)
                # tl.atomic_add(indice_num_per_loc + RS - 1 - pid_1, ss)


def get_indice_pairs(
    indices,
    H, W,
    P, Q,
    R, S,
    str_h, str_w,
    pad_h, pad_w,
    dil_h, dil_w
):
    # generate_subm_conv_inds function
    RS = R * S
    num_points = indices.shape[0]
    table = build_subm_conv_hash_table(indices, H, W)
    gather_idx = torch.full((RS, num_points), -1, dtype=torch.int32, device=indices.device)
    scatter_idx = torch.full((RS, num_points), -1, dtype=torch.int32, device=indices.device)
    indice_num_per_loc = torch.zeros((RS, ), dtype=torch.int32, device=indices.device)

    num_indices_in = indices.shape[0]

    if num_indices_in < 1024:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_indices_in, 32) * 32)
    else:
        BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(num_indices_in, BLOCK_SIZE), RS // 2 + 1)

    calc_subm_conv_indices_kernel[grid](
        table._table_key, table._table_val, table._hash_size, table._empty_key_uint,
        indices,
        gather_idx, scatter_idx, indice_num_per_loc,
        num_indices_in,
        RS,
        H, W,
        P, Q,
        R, S,
        str_h, str_w,
        pad_h, pad_w,
        dil_h, dil_w,
        BLOCK_SIZE,
    )

    return indices, gather_idx, scatter_idx, indice_num_per_loc


# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/nn/SubmConv2D_cn.html#submconv2d
# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _triton_submconv2d(torch.autograd.Function):
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
    ):
        # why? Adapted from spconv
        str_h, str_w = 1, 1
        K, R, S, C = weight.shape

        assert R % 2 == 1, "subm only support odd ksize(R={R})"
        assert S % 2 == 1, "subm only support odd ksize(S={S})"

        dil_h, dil_w = dilation
        pad_h = (R // 2) * dil_h
        pad_w = (S // 2) * dil_w
        
        num_points = len(features)

        P = H
        Q = W

        out_indices, gather_idx, scatter_idx, indice_num_per_loc = get_indice_pairs(
            indices,
            H, W,
            P, Q,
            R, S,
            str_h, str_w,
            pad_h, pad_w,
            dil_h, dil_w
        )

        RS = R * S
        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)
       
        w = weight[:, R // 2, S // 2, :]

        p_in = features[gather_idx[RS // 2]]
        p_out = p_in @ w.t()
        out_features[scatter_idx[RS // 2]] += p_out

        for rs in range(RS // 2):
            if indice_num_per_loc[rs] != 0:
                r = rs // S
                s = rs % S
                rotate_rs = RS - 1 - rs
                rotate_r = rotate_rs // S
                rotate_s = rotate_rs % S

                w = weight[:, r, s, :]

                idx = gather_idx[rs][gather_idx[rs] != -1]
                p_in = features[idx]

                p_out = p_in @ w.t()

                idx = scatter_idx[rs][scatter_idx[rs] != -1]
                out_features[idx] += p_out

                w = weight[:, rotate_r, rotate_s, :]

                idx = gather_idx[rotate_rs][gather_idx[rotate_rs] != -1]
                rotate_p_in = features[idx]

                rotate_p_out = rotate_p_in @ w.t()

                idx = scatter_idx[rotate_rs][scatter_idx[rotate_rs] != -1]
                out_features[idx] += rotate_p_out
 
        if bias is not None:
            out_features += bias

        ctx.save_for_backward(gather_idx, scatter_idx, indice_num_per_loc, features, weight)
        ctx.R = R
        ctx.S = S

        if bias is None:
            ctx.bias_requires_grad = False
        else:
            ctx.bias_requires_grad = bias.requires_grad
        
        return out_features, out_indices, P, Q
    
    @staticmethod
    def backward(ctx, dout_features: torch.Tensor, *args):
        fwd_gather_idx, fwd_scatter_idx, indice_num_per_loc, features, weight = ctx.saved_tensors
        R = ctx.R
        S = ctx.S

        dfeatures = None
        if features.requires_grad:
            dfeatures = torch.zeros_like(features)
            gather_idx = fwd_scatter_idx
            scatter_idx = fwd_gather_idx

            RS = R * S
            w = weight[:, R // 2, S // 2, :]
            p_dout = dout_features[gather_idx[RS // 2]]
            p_din = p_dout @ w
            dfeatures[scatter_idx[RS // 2]] += p_din

            for rs in range(RS // 2):
                if indice_num_per_loc[rs] != 0:
                    r = rs // S
                    s = rs % S

                    rotate_rs = RS - 1 - rs
                    rotate_r = rotate_rs // S
                    rotate_s = rotate_rs % S

                    w = weight[:, r, s, :]

                    idx = gather_idx[rs][gather_idx[rs] != -1]
                    p_dout = dout_features[idx]

                    p_din = p_dout @ w

                    idx = scatter_idx[rs][scatter_idx[rs] != -1]
                    dfeatures[idx] += p_din

                    w = weight[:, rotate_r, rotate_s, :]

                    idx = gather_idx[rotate_rs][gather_idx[rotate_rs] != -1]
                    rotate_p_dout = dout_features[idx]

                    rotate_p_din = rotate_p_dout @ w

                    idx = scatter_idx[rotate_rs][scatter_idx[rotate_rs] != -1]
                    dfeatures[idx] += rotate_p_din

        dweight = None
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)
            dout_gather_idx = fwd_scatter_idx
            input_gather_idx = fwd_gather_idx
            
            RS = R * S
            p_dout = dout_features[dout_gather_idx[RS // 2]]
            p_in = features[input_gather_idx[RS // 2]]
            dweight[:, R // 2, S // 2, :] += p_dout.t() @ p_in

            if not torch.all(indice_num_per_loc == 0):
                for rs in range(RS // 2):
                    if indice_num_per_loc[rs] != 0:
                        r = rs // S
                        s = rs % S

                        rotate_rs = RS - 1 - rs
                        rotate_r = rotate_rs // S
                        rotate_s = rotate_rs % S
                        
                        idx = dout_gather_idx[rs][dout_gather_idx[rs] != -1]
                        p_dout = dout_features[idx]

                        idx = input_gather_idx[rs][input_gather_idx[rs] != -1]
                        p_in = features[idx]

                        dweight[:, r, s, :] += p_dout.t() @ p_in

                        idx = dout_gather_idx[rotate_rs][dout_gather_idx[rotate_rs] != -1]
                        p_dout = dout_features[idx]
                        idx = input_gather_idx[rotate_rs][input_gather_idx[rotate_rs] != -1]
                        p_in = features[idx]
                        dweight[:, rotate_r, rotate_s, :] += p_dout.t() @ p_in

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, dweight, dbias, None, None, None, None


triton_submconv2d = _triton_submconv2d.apply
