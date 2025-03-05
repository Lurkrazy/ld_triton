
import torch
import triton
import triton.language as tl
from ld_triton.ops.spconv.triton_hash import LinearHashTableSplit


@triton.jit
def build_conv_hash_table_kernel(
    table_key_ptr, table_val_ptr, hash_size, empty_key_uint,
    indices_out_ptr, 
    indice_pairs_for_uniq_ptr,
    num_indices,
    PQ_0, PQ_1, PQ_2,
    BLOCK_SIZE: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    indice_pairs_for_uniq_offs = pid_0 * BLOCK_SIZE + offsets
    mask = indice_pairs_for_uniq_offs < num_indices
    output_coord_offset = tl.load(indice_pairs_for_uniq_ptr + indice_pairs_for_uniq_offs, mask=mask)
    bs = output_coord_offset // (PQ_0 * PQ_1 * PQ_2)
    pq_0 = (output_coord_offset % (PQ_0 * PQ_1 * PQ_2)) // (PQ_1 * PQ_2)
    pq_1 = (output_coord_offset % (PQ_1 * PQ_2)) // PQ_2
    pq_2 = output_coord_offset % PQ_2
    
    bs_offs = (pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * 4
    pq_0_offs = bs_offs + 1
    pq_1_offs = bs_offs + 2
    pq_2_offs = bs_offs + 3
    tl.store(indices_out_ptr + bs_offs, bs, bs_offs < 4 * num_indices)
    tl.store(indices_out_ptr + pq_0_offs, pq_0, pq_0_offs < 4 * num_indices)
    tl.store(indices_out_ptr + pq_1_offs, pq_1, pq_1_offs < 4 * num_indices)
    tl.store(indices_out_ptr + pq_2_offs, pq_2, pq_2_offs < 4 * num_indices)

    # hash
    key = output_coord_offset
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
    
    kv_mask = indice_pairs_for_uniq_offs < num_indices
    key = tl.where(kv_mask, key, empty_key_uint).to(tl.uint32)
    slot = tl.where(kv_mask, key.to(tl.uint32) % hash_size.to(tl.uint32), hash_size)
    cmp = tl.full((BLOCK_SIZE, ), empty_key_uint, dtype=tl.uint32)
    # if table_key_ptr + slot address is illegal, the prev = 0
    prev = tl.atomic_cas(table_key_ptr + slot, cmp, key)
    prev = tl.where(kv_mask, prev, cmp)
    mask = tl.where(prev == cmp or prev == key, True, False) & kv_mask
    
    val = indice_pairs_for_uniq_offs
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


def build_conv_hash_table(
    indices_out: torch.Tensor, 
    indice_pairs_for_uniq,
    PQ_0, PQ_1, PQ_2
):
    num_indices = indices_out.shape[0]
    table = LinearHashTableSplit(num_indices, rate = 2.0)

    if num_indices < 1024:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_indices, 32) * 32)
    else:
        BLOCK_SIZE = 1024

    grid = (triton.cdiv(num_indices, BLOCK_SIZE), )
    build_conv_hash_table_kernel[grid](
        table._table_key, table._table_val, table._hash_size, table._empty_key_uint,
        indices_out,
        indice_pairs_for_uniq,
        num_indices,
        PQ_0, PQ_1, PQ_2,
        BLOCK_SIZE,
    )
    return table


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
def calc_conv_indices_stage1_kernel(      
    indices_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    indice_pairs_uniq_ptr,
    indice_num_per_loc_ptr,
    num_indices_in,
    indices_pair_size,
    HW_0, HW_1, HW_2,
    PQ_0, PQ_1, PQ_2,
    RS_0, RS_1, RS_2,
    str_hw_0, str_hw_1, str_hw_2,
    pad_hw_0, pad_hw_1, pad_hw_2,
    dil_hw_0, dil_hw_1, dil_hw_2,
    BLOCK_SIZE: tl.constexpr
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1) 

    rs_0 = pid_1 // (RS_1 * RS_2)
    rs_1 = (pid_1 % (RS_1 * RS_2)) // RS_2
    rs_2 = pid_1 % RS_2

    center_offsets = (pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * 4
    bs   = tl.load(indices_ptr + center_offsets + 0, center_offsets + 0 < 4 * num_indices_in)
    hw_0 = tl.load(indices_ptr + center_offsets + 1, center_offsets + 1 < 4 * num_indices_in)
    hw_1 = tl.load(indices_ptr + center_offsets + 2, center_offsets + 2 < 4 * num_indices_in)
    hw_2 = tl.load(indices_ptr + center_offsets + 3, center_offsets + 3 < 4 * num_indices_in)
    mask_mod_0 = (hw_0 + pad_hw_0 - rs_0 * dil_hw_0) % str_hw_0 == 0
    mask_mod_1 = (hw_1 + pad_hw_1 - rs_1 * dil_hw_1) % str_hw_1 == 0
    mask_mod_2 = (hw_2 + pad_hw_2 - rs_2 * dil_hw_2) % str_hw_2 == 0
    mask_mod = mask_mod_0 & mask_mod_1 & mask_mod_2
    is_mod_broke = tl.sum(mask_mod.to(tl.int32), axis=0)
    if is_mod_broke != 0:
        pq_0 = (hw_0 + pad_hw_0 - rs_0 * dil_hw_0) // str_hw_0
        pq_1 = (hw_1 + pad_hw_1 - rs_1 * dil_hw_1) // str_hw_1
        pq_2 = (hw_2 + pad_hw_2 - rs_2 * dil_hw_2) // str_hw_2
        
        mask_pq_0 = pq_0 >= 0 and pq_0 < PQ_0
        mask_pq_1 = pq_1 >= 0 and pq_1 < PQ_1
        mask_pq_2 = pq_2 >= 0 and pq_2 < PQ_2
        mask_pq = mask_pq_0 & mask_pq_1 & mask_pq_2
        is_pq_broke = tl.sum(mask_pq.to(tl.int32), axis=0)
        if is_pq_broke != 0:
            p_out_idx = bs * PQ_0 * PQ_1 * PQ_2 + pq_0 * PQ_1 * PQ_2 + pq_1 * PQ_2 + pq_2
            i_in = pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            gather_idx_offs = pid_1 * num_indices_in + i_in
            indice_pairs_uniq_offs = pid_1 * indices_pair_size + i_in
            filter_mask = mask_mod & mask_pq & (i_in < num_indices_in)
            ss = tl.sum(filter_mask.to(tl.int32), axis=0)
            tl.atomic_add(indice_num_per_loc_ptr + pid_1, ss)
            tl.store(gather_idx_ptr + gather_idx_offs, i_in, mask=filter_mask)
            tl.store(indice_pairs_uniq_ptr + indice_pairs_uniq_offs, p_out_idx, mask=filter_mask)


def calc_conv_indices_stage1(     
    indices,
    gather_idx,
    scatter_idx,
    indice_pairs_uniq,
    indice_num_per_loc,
    num_indices,
    indices_pair_size,
    HW_0, HW_1, HW_2,
    PQ_0, PQ_1, PQ_2,
    RS_0, RS_1, RS_2,
    str_hw_0, str_hw_1, str_hw_2,
    pad_hw_0, pad_hw_1, pad_hw_2,
    dil_hw_0, dil_hw_1, dil_hw_2
):
    num_indices_in = indices.shape[0]
    if num_indices_in < 1024:
        BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_indices_in, 32) * 32)
    else:
        BLOCK_SIZE = 1024
    RS = RS_0 * RS_1 * RS_2
    grid = (triton.cdiv(num_indices_in, BLOCK_SIZE), RS)
    calc_conv_indices_stage1_kernel[grid](
        indices,
        gather_idx,
        scatter_idx,
        indice_pairs_uniq,
        indice_num_per_loc,
        num_indices,
        indices_pair_size,
        HW_0, HW_1, HW_2,
        PQ_0, PQ_1, PQ_2,
        RS_0, RS_1, RS_2,
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,
        BLOCK_SIZE,
    )
    

def generate_conv_inds_stage1(
    indices,
    gather_idx,
    scatter_idx,
    indice_pairs_uniq,
    indice_num_per_loc,
    HW_0, HW_1, HW_2,
    PQ_0, PQ_1, PQ_2,
    RS_0, RS_1, RS_2,
    str_hw_0, str_hw_1, str_hw_2,
    pad_hw_0, pad_hw_1, pad_hw_2,
    dil_hw_0, dil_hw_1, dil_hw_2
):
    num_indices = indices.shape[0]
    indices_pair_size = indices.shape[0]
    
    calc_conv_indices_stage1(
        indices,
        gather_idx,
        scatter_idx,
        indice_pairs_uniq,
        indice_num_per_loc,
        num_indices,
        indices_pair_size,
        HW_0, HW_1, HW_2,
        PQ_0, PQ_1, PQ_2,
        RS_0, RS_1, RS_2,
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,
    )


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


def calc_conv_indices_stage2(
    table: LinearHashTableSplit,
    indice_pairs_uniq_before_sort,
    scatter_idx,
    num_indices_in,
    indices_pair_size,
    RS_0, RS_1, RS_2,
):
    RS = RS_0 * RS_1 * RS_2
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


def generate_conv_inds_stage2(
    indices, 
    gather_idx,
    scatter_idx, 
    indice_pairs_uniq, 
    indice_pairs_uniq_before_sort, 
    out_inds,
    indice_num_per_loc, 
    num_out_act, 
    HW_0, HW_1, HW_2,
    PQ_0, PQ_1, PQ_2,
    RS_0, RS_1, RS_2,
    str_hw_0, str_hw_1, str_hw_2,
    pad_hw_0, pad_hw_1, pad_hw_2,
    dil_hw_0, dil_hw_1, dil_hw_2
): 
    table = build_conv_hash_table(out_inds, indice_pairs_uniq, PQ_0, PQ_1, PQ_2)

    num_indices_in = indices.shape[0]
    calc_conv_indices_stage2(
        table,
        indice_pairs_uniq_before_sort,
        scatter_idx,
        num_indices_in,
        num_indices_in,
        RS_0, RS_1, RS_2,
    )


def get_indice_pairs(
    indices,
    HW_0, HW_1, HW_2,
    PQ_0, PQ_1, PQ_2,
    RS_0, RS_1, RS_2,
    str_hw_0, str_hw_1, str_hw_2,
    pad_hw_0, pad_hw_1, pad_hw_2,
    dil_hw_0, dil_hw_1, dil_hw_2,
):
    RS = RS_0 * RS_1 * RS_2
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
        HW_0, HW_1, HW_2,
        PQ_0, PQ_1, PQ_2,
        RS_0, RS_1, RS_2,
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,
    )
    uniq_res = indice_pairs_uniq.unique()
    num_act_out = uniq_res.shape[0] - 1
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
        HW_0, HW_1, HW_2,
        PQ_0, PQ_1, PQ_2,
        RS_0, RS_1, RS_2,
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2
    )

    return out_inds, gather_idx, scatter_idx, indice_num_per_loc


# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _triton_sparseconv3d(torch.autograd.Function):
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
        out_indices, gather_idx, scatter_idx, indice_num_per_loc = get_indice_pairs(
            indices,
            HW_0, HW_1, HW_2,
            PQ_0, PQ_1, PQ_2,
            RS_0, RS_1, RS_2,
            str_0, str_1, str_2,
            pad_0, pad_1, pad_2,
            dil_0, dil_1, dil_2,
        )

        RS = RS_0 * RS_1 * RS_2
        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)

        for rs in range(RS):
            if indice_num_per_loc[rs] != 0:
                rs_0 = rs // (RS_1 * RS_2)
                rs_1 = (rs % (RS_1 * RS_2)) // RS_2
                rs_2 = rs % RS_2
                w = weight[:, rs_0, rs_1, rs_2, :]
                idx = gather_idx[rs][gather_idx[rs] != -1]
                p_in = features[idx]
                p_out = p_in @ w.t()
                idx = scatter_idx[rs][scatter_idx[rs] != -1]
                out_features[idx] += p_out
 
        if bias is not None:
            out_features += bias

        ctx.save_for_backward(gather_idx, scatter_idx, indice_num_per_loc, features, weight)
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
        fwd_gather_idx, fwd_scatter_idx, indice_num_per_loc, features, weight = ctx.saved_tensors
        RS_0 = ctx.RS_0
        RS_1 = ctx.RS_1
        RS_2 = ctx.RS_2

        dfeatures = None
        if features.requires_grad:
            dfeatures = torch.zeros_like(features)
            gather_idx = fwd_scatter_idx
            scatter_idx = fwd_gather_idx

            RS = RS_0 * RS_1 * RS_2
            for rs in range(RS):
                if indice_num_per_loc[rs] != 0:
                    rs_0 = rs // (RS_1 * RS_2)
                    rs_1 = (rs % (RS_1 * RS_2)) // RS_2
                    rs_2 = rs % RS_2

                    w = weight[:, rs_0, rs_1, rs_2, :]
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
            
            RS = RS_0 * RS_1 * RS_2
            for rs in range(RS):
                if indice_num_per_loc[rs] != 0:
                    rs_0 = rs // (RS_1 * RS_2)
                    rs_1 = (rs % (RS_1 * RS_2)) // RS_2
                    rs_2 = rs % RS_2
                    idx = dout_gather_idx[rs][dout_gather_idx[rs] != -1]
                    p_dout = dout_features[idx]
                    idx = input_gather_idx[rs][input_gather_idx[rs] != -1]
                    p_in = features[idx]
                    dweight[:, rs_0, rs_1, rs_2, :] += p_dout.t() @ p_in

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, None, dweight, dbias, None, None, None, None


triton_sparseconv3d = _triton_sparseconv3d.apply
