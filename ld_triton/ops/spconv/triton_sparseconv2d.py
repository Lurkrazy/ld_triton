
import torch
import triton
import triton.language as tl

from ld_triton.ops.spconv.triton_hash import LinearHashTableSplit


@triton.jit
def build_conv_hash_table_kernel(
    table_key_ptr, table_val_ptr, hash_size, empty_key_uint,
    indices_out_ptr, 
    indice_pairs_for_uniq_ptr,
    num_act_in_real,
    P, Q,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. build table
    pid = tl.program_id(0)
    kv_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    key_ptrs = indice_pairs_for_uniq_ptr +  (kv_start + offsets) 
    kv_mask = (kv_start + offsets) < num_act_in_real
    key = tl.load(key_ptrs, mask=kv_mask, other=empty_key_uint)
    output_coord_offset = tl.where(kv_mask, key, empty_key_uint)
    val = tl.where(kv_mask, (kv_start + offsets).to(tl.uint32), empty_key_uint).to(tl.uint32)

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
    cmp = tl.full((BLOCK_SIZE, ), empty_key_uint, dtype=tl.uint32)
    # if table_key_ptr + slot address is illegal, the prev = 0
    prev = tl.atomic_cas(table_key_ptr + slot, cmp, key)
    prev = tl.where(kv_mask, prev, cmp)
    mask = tl.where(prev == cmp or prev == key, True, False) & kv_mask
    
    tl.store(table_val_ptr + slot, val, mask=mask)
    is_broke = tl.sum(mask.to(tl.uint64), axis=0)
    total_mask = tl.where(cmp, False, False)
    while is_broke != 0:
        prev_mask = mask
        total_mask = (total_mask | prev_mask) & kv_mask
        
        slot = tl.where(kv_mask & ~total_mask, (slot + 1) % hash_size.to(tl.uint32), hash_size)
        
        key = tl.where(total_mask, empty_key_uint, key).to(tl.uint32)
        prev = tl.atomic_cas(table_key_ptr + slot, cmp, key)

        # find a slot, this slot has no key or the key is the same as the current key, break
        mask = tl.where((prev == cmp or prev == key) & kv_mask & ~total_mask, True, False) 
        # find a slot, this slot already has a different key, continue to find the next slot
        # must prev != 0, if table_key_ptr + slot address is illegal, the prev = 0
        skip_mask = tl.where(((prev != cmp) & (prev != key) & (prev != 0)) & kv_mask & ~total_mask, True, False) 

        is_store = tl.sum(mask.to(tl.uint64), axis=0)
        is_broke = is_store + tl.sum(skip_mask.to(tl.uint64), axis=0)
        if is_store != 0:
            tl.store(table_val_ptr + slot, val, mask=mask)

    # 2. build indices_out
    bs = output_coord_offset // (P * Q)
    p = (output_coord_offset % (P * Q)) // Q
    q = output_coord_offset % Q
    
    bs_offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * 3
    p_offs = bs_offs + 1
    q_offs = bs_offs + 2

    tl.store(indices_out_ptr + bs_offs, bs, bs_offs < 3 * num_act_in_real)
    tl.store(indices_out_ptr + p_offs, p, p_offs < 3 * num_act_in_real)
    tl.store(indices_out_ptr + q_offs, q, q_offs < 3 * num_act_in_real)


def build_conv_hash_table(
    indices_out: torch.Tensor, 
    indice_pairs_for_uniq,
    P, Q
):
    
    num_indices = indices_out.shape[0]
    assert len(indice_pairs_for_uniq) == num_indices, f'{len(indice_pairs_for_uniq)} != {num_indices}'
    table = LinearHashTableSplit(num_indices, rate = 2.0)

    if num_indices < 1024:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_indices, 32) * 32)
    else:
        BLOCK_SIZE = 1024

    grid = (triton.cdiv(num_indices, BLOCK_SIZE), )
    indice_pairs_for_uniq = indice_pairs_for_uniq.to(torch.uint32)
    build_conv_hash_table_kernel[grid](
        table._table_key, table._table_val, table._hash_size, table._empty_key_uint,
        indices_out,
        indice_pairs_for_uniq,
        num_indices,
        P, Q,
        BLOCK_SIZE,
    )
    return table


@triton.jit
def calc_conv_indices_stage1_kernel(      
    indices_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    indice_pairs_uniq_ptr,
    indice_num_per_loc_ptr,
    num_indices_in,
    indices_pair_size,
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

    r = pid_1 // S
    s = pid_1 % S

    bs_offs = (pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * 3
    p_offs = bs_offs + 1
    q_offs = bs_offs + 2

    bs   = tl.load(indices_ptr + bs_offs + 0, bs_offs < 3 * num_indices_in)
    h = tl.load(indices_ptr + p_offs, p_offs < 3 * num_indices_in)
    w = tl.load(indices_ptr + q_offs, q_offs < 3 * num_indices_in)
    mask_mod_h = (h + pad_h - r * dil_h) % str_h == 0
    mask_mod_w = (w + pad_w - s * dil_w) % str_w == 0
    mask_mod = mask_mod_h & mask_mod_w
    is_mod_broke = tl.sum(mask_mod.to(tl.int32), axis=0)
    if is_mod_broke != 0:
        p = (h + pad_h - r * dil_h) // str_h
        q = (w + pad_w - s * dil_w) // str_w        
        mask_p = p >= 0 and p < P
        mask_q = q >= 0 and q < Q
        mask_pq = mask_p & mask_q
        is_pq_broke = tl.sum(mask_pq.to(tl.int32), axis=0)
        if is_pq_broke != 0:
            p_out_idx = bs * P * Q + p * Q + q
            i_in = pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            gather_idx_offs = pid_1 * num_indices_in + i_in
            indice_pairs_uniq_offs = pid_1 * indices_pair_size + i_in
            filter_mask = mask_mod & mask_pq & (i_in < num_indices_in)
            ss = tl.sum(filter_mask.to(tl.int32), axis=0)
            tl.atomic_add(indice_num_per_loc_ptr + pid_1, ss)
            tl.store(gather_idx_ptr + gather_idx_offs, i_in, mask=filter_mask)
            tl.store(indice_pairs_uniq_ptr + indice_pairs_uniq_offs, p_out_idx, mask=filter_mask)
 

def generate_conv_inds_stage1(
    indices,
    gather_idx,
    scatter_idx,
    indice_pairs_uniq,
    indice_num_per_loc,
    H, W,
    P, Q,
    R, S,
    str_h, str_w,
    pad_h, pad_w,
    dil_h, dil_w,
):
    num_indices = indices.shape[0]
    indices_pair_size = indices.shape[0]
    
    num_indices_in = indices.shape[0]
    if num_indices_in < 1024:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_indices_in, 32) * 32)
    else:
        BLOCK_SIZE = 1024
    
    RS = R * S
    grid = (triton.cdiv(num_indices_in, BLOCK_SIZE), RS)
    calc_conv_indices_stage1_kernel[grid](
        indices,
        gather_idx,
        scatter_idx,
        indice_pairs_uniq,
        indice_num_per_loc,
        num_indices,
        indices_pair_size,
        H, W,
        P, Q,
        R, S,
        str_h, str_w,
        pad_h, pad_w,
        dil_h, dil_w,
        BLOCK_SIZE,
    )


@triton.jit
def linear_hash_table_lookup_offset_impl(
    table_key_ptr, table_val_ptr, 
    key, kv_size, hash_size, empty_key_uint, 
    BLOCK_SIZE: tl.constexpr
):
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
def calc_conv_indices_stage2_kernel(
    table_key_ptr, table_val_ptr, hash_size, empty_key_uint,
    indice_pairs_uniq_before_sort_ptr,
    scatter_idx_ptr,
    num_indices_in,
    indices_pair_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    offs = pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 
    scatter_idx_offs = pid_1 * indices_pair_size + offs
    indice_pairs_uniq_offs = pid_1 * indices_pair_size + offs
    
    offs_mask = offs < num_indices_in

    indice_pairs_uniq_before_sort = tl.load(indice_pairs_uniq_before_sort_ptr + indice_pairs_uniq_offs, mask=offs_mask)
    output_coord_offset = tl.where(offs_mask & (indice_pairs_uniq_before_sort != 0x7FFFFFFF), indice_pairs_uniq_before_sort, empty_key_uint)
    output_coord_offset = output_coord_offset.to(tl.uint32)
    table_slot, idx = linear_hash_table_lookup_offset_impl(
        table_key_ptr, table_val_ptr, output_coord_offset, BLOCK_SIZE, hash_size, empty_key_uint, BLOCK_SIZE
    )

    filter_mask = (table_slot != -1) & offs_mask
    tl.store(scatter_idx_ptr + scatter_idx_offs, idx, mask=filter_mask)


def generate_conv_inds_stage2(
    indices, 
    gather_idx,
    scatter_idx, 
    indice_pairs_uniq, 
    indice_pairs_uniq_before_sort, 
    out_inds,
    indice_num_per_loc, 
    num_out_act, 
    H, W,
    P, Q,
    R, S,
    str_h, str_w,
    pad_h, pad_w,
    dil_h, dil_w,
): 
    table = build_conv_hash_table(out_inds, indice_pairs_uniq, P, Q)
    num_indices_in = indices.shape[0]
    indices_pair_size = indices.shape[0]
    RS = R * S
    if num_indices_in < 1024:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_indices_in, 32) * 32)
    else:
        BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_indices_in, BLOCK_SIZE), RS, )
    calc_conv_indices_stage2_kernel[grid](
        table._table_key, table._table_val, table._hash_size, table._empty_key_uint,
        indice_pairs_uniq_before_sort,
        scatter_idx,
        num_indices_in,
        indices_pair_size,
        BLOCK_SIZE,
    )


def get_indice_pairs(
    indices,
    H, W,
    P, Q,
    R, S,
    str_h, str_w,
    pad_h, pad_w,
    dil_h, dil_w,
):
    RS = R * S
    num_points = indices.shape[0]
    gather_idx = torch.full((RS, num_points), -1, dtype=torch.int32, device=indices.device)
    scatter_idx = torch.full((RS, num_points), -1, dtype=torch.int32, device=indices.device)
    indice_num_per_loc = torch.zeros((RS, ), dtype=torch.int32, device=indices.device)

    indice_pairs_uniq = torch.full((gather_idx.numel() + 1, ), 0x7FFFFFFF, dtype=torch.int32, device=indices.device)
    
    generate_conv_inds_stage1(
        indices,
        gather_idx,
        scatter_idx,
        indice_pairs_uniq,
        indice_num_per_loc,
        H, W,
        P, Q,
        R, S,
        str_h, str_w,
        pad_h, pad_w,
        dil_h, dil_w,
    )

    uniq_res = indice_pairs_uniq.unique()
    num_act_out = uniq_res.shape[0] - 1
    uniq_res = uniq_res[:-1]
    if num_act_out == 0:
        raise Exception('No active output')
    
    out_inds = torch.empty((num_act_out, indices.shape[1]), dtype=indices.dtype, device=indices.device)
    
    generate_conv_inds_stage2(
        indices, 
        gather_idx, 
        scatter_idx, 
        uniq_res, 
        indice_pairs_uniq, 
        out_inds,
        indice_num_per_loc, 
        num_act_out, 
        H, W,
        P, Q,
        R, S,
        str_h, str_w,
        pad_h, pad_w,
        dil_h, dil_w,
    )
    return out_inds, gather_idx, scatter_idx, indice_num_per_loc


# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _triton_sparseconv2d(torch.autograd.Function):
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
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation
        num_points = len(features)
        
        K, R, S, C = weight.shape
        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        out_indices, gather_idx, scatter_idx, indice_num_per_loc = get_indice_pairs(
            indices,
            H, W,
            P, Q,
            R, S,
            str_h, str_w,
            pad_h, pad_w,
            dil_h, dil_w,
        )

        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)
        RS = R * S
        for rs in range(RS):
            if indice_num_per_loc[rs] != 0:
                r = rs // S
                s = rs % S
                w = weight[:, r, s, :]
                idx = gather_idx[rs][gather_idx[rs] != -1]
                p_in = features[idx]
                p_out = p_in @ w.t()
                idx = scatter_idx[rs][scatter_idx[rs] != -1]
                out_features[idx] += p_out
 
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
            for rs in range(RS):
                if indice_num_per_loc[rs] != 0:
                    r = rs // S
                    s = rs % S
                    w = weight[:, r, s, :]
                    idx = gather_idx[rs][gather_idx[rs] != -1]
                    p_dout = dout_features[idx]
                    p_din = p_dout @ w
                    idx = scatter_idx[rs][scatter_idx[rs] != -1]
                    dfeatures[idx] += p_din

        dweight = None
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)
            dout_gather_idx = fwd_scatter_idx
            input_gather_idx = fwd_gather_idx
            
            RS = R * S
            for rs in range(RS):
                if indice_num_per_loc[rs] != 0:
                    idx = dout_gather_idx[rs][dout_gather_idx[rs] != -1]
                    p_dout = dout_features[idx]

                    idx = input_gather_idx[rs][input_gather_idx[rs] != -1]
                    p_in = features[idx]

                    r = rs // S
                    s = rs % S
                    dweight[:, r, s, :] += p_dout.t() @ p_in

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, dweight, dbias, None, None, None, None


triton_sparseconv2d = _triton_sparseconv2d.apply
