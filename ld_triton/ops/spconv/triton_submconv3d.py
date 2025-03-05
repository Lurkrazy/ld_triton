
import torch
import time
import triton
import triton.language as tl
from ld_triton.ops.spconv.triton_hash import LinearHashTableSplit
from ld_triton.ops.spconv.triton_hash import linear_hash_table_lookup_offset

TRITON_SUBMCONV3D_DEBUG = True

def build_subm_conv_hash_table(indices: torch.Tensor, spatial_shape):
    kv_size = indices.shape[0]
    indices = indices.to(torch.int32)
    key = indices[:, 0] * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] + \
            indices[:, 1] * spatial_shape[1] * spatial_shape[2] + \
            indices[:, 2] * spatial_shape[2] + \
            indices[:, 3]
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
def calc_subm_conv_indices_kernel(table_key_ptr, table_val_ptr, hash_size, empty_key_uint,
                                  indices_ptr,
                                  gather_idx_ptr, scatter_idx_ptr, indice_num_per_loc,
                                  num_indices_in, 
                                  RS,
                                  RS_0, RS_1, RS_2, 
                                  str_hw_0, str_hw_1, str_hw_2,
                                  pad_hw_0, pad_hw_1, pad_hw_2,
                                  dil_hw_0, dil_hw_1, dil_hw_2,
                                  PQ_0, PQ_1, PQ_2,
                                  BLOCK_SIZE: tl.constexpr):
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
        rs_0 = pid_1 // (RS_1 * RS_2)
        rs_1 = (pid_1 % (RS_1 * RS_2)) // RS_2
        rs_2 = pid_1 % RS_2
        center_rs_0 = RS_0 // 2
        center_rs_1 = RS_1 // 2
        center_rs_2 = RS_2 // 2
        center_offsets = (pid_0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * 4
        bs   = tl.load(indices_ptr + center_offsets + 0, center_offsets + 0 < 4 * num_indices_in)
        hw_0 = tl.load(indices_ptr + center_offsets + 1, center_offsets + 1 < 4 * num_indices_in)
        hw_1 = tl.load(indices_ptr + center_offsets + 2, center_offsets + 2 < 4 * num_indices_in)
        hw_2 = tl.load(indices_ptr + center_offsets + 3, center_offsets + 3 < 4 * num_indices_in)
        center_hw_0 = hw_0 + (center_rs_0 - rs_0) * dil_hw_0
        center_hw_1 = hw_1 + (center_rs_1 - rs_1) * dil_hw_1
        center_hw_2 = hw_2 + (center_rs_2 - rs_2) * dil_hw_2
        center_p_idx = bs * PQ_0 * PQ_1 * PQ_2 + center_hw_0 * PQ_1 * PQ_2 + center_hw_1 * PQ_2 + center_hw_2
        center_p_slot, _ = linear_hash_table_lookup_offset_impl(table_key_ptr, table_val_ptr, 
                                                                center_p_idx,
                                                                BLOCK_SIZE, hash_size, 
                                                                empty_key_uint, BLOCK_SIZE)
        mask_mod_0 = (hw_0 + pad_hw_0 - rs_0 * dil_hw_0) % str_hw_0 == 0
        mask_mod_1 = (hw_1 + pad_hw_1 - rs_1 * dil_hw_1) % str_hw_1 == 0
        mask_mod_2 = (hw_2 + pad_hw_2 - rs_2 * dil_hw_2) % str_hw_2 == 0
        mask_mod_3 = center_p_slot != -1
        mask_mod = mask_mod_0 & mask_mod_1 & mask_mod_2 & mask_mod_3
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


def calc_subm_conv_indices(
        table: LinearHashTableSplit, 
        indices, 
        gather_idx, scatter_idx, indice_num_per_loc,
        RS_0, RS_1, RS_2, 
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,
        HW_shape,
        PQ_0, PQ_1, PQ_2, 
    ):
    RS = RS_0 * RS_1 * RS_2
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
        RS_0, RS_1, RS_2, 
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,
        PQ_0, PQ_1, PQ_2,
        BLOCK_SIZE,
        )


def generate_subm_conv_inds(indices,
                            HW_0, HW_1, HW_2,
                            PQ_0, PQ_1, PQ_2,
                            RS_0, RS_1, RS_2,
                            str_hw_0, str_hw_1, str_hw_2,
                            pad_hw_0, pad_hw_1, pad_hw_2,
                            dil_hw_0, dil_hw_1, dil_hw_2,
                            ):
    RS = RS_0 * RS_1 * RS_2
    num_points = indices.shape[0]
    table = build_subm_conv_hash_table(indices, [HW_0, HW_1, HW_2])
    gather_idx = torch.full((RS, num_points), -1, dtype=torch.int32, device=indices.device)
    scatter_idx = torch.full((RS, num_points), -1, dtype=torch.int32, device=indices.device)
    indice_num_per_loc = torch.zeros((RS, ), dtype=torch.int32, device=indices.device)
    calc_subm_conv_indices(
        table,
        indices,
        gather_idx, scatter_idx, indice_num_per_loc,
        RS_0, RS_1, RS_2,
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,
        [HW_0, HW_1, HW_2],
        PQ_0, PQ_1, PQ_2,
    )

    return indices, gather_idx, scatter_idx, indice_num_per_loc


def get_indice_pairs(indices,
                    HW_0, HW_1, HW_2,
                    PQ_0, PQ_1, PQ_2,
                    RS_0, RS_1, RS_2,
                    str_hw_0, str_hw_1, str_hw_2,
                    pad_hw_0, pad_hw_1, pad_hw_2,
                    dil_hw_0, dil_hw_1, dil_hw_2,
):
    return generate_subm_conv_inds(indices,
                                    HW_0, HW_1, HW_2,
                                    PQ_0, PQ_1, PQ_2,
                                    RS_0, RS_1, RS_2,
                                    str_hw_0, str_hw_1, str_hw_2,
                                    pad_hw_0, pad_hw_1, pad_hw_2,
                                    dil_hw_0, dil_hw_1, dil_hw_2,
                                    )

# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/nn/SubmConv2D_cn.html#submconv2d
# [SECOND: Sparsely Embedded Convolutional Detection] https://www.mdpi.com/1424-8220/18/10/3337/pdf?version=1538798176
class _triton_submconv3d(torch.autograd.Function):
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
                dilation: int = (1, 1, 1)):
        _fwd_debug = False
        # why? Adapted from spconv
        str_hw_0, str_hw_1, str_hw_2 = 1, 1, 1
        K, RS_0, RS_1, RS_2, C = weight.shape

        assert RS_0 % 2 == 1, "subm only support odd ksize(RS_0={RS_0})"
        assert RS_1 % 2 == 1, "subm only support odd ksize(RS_1={RS_1})"
        assert RS_2 % 2 == 1, "subm only support odd ksize(RS_2={RS_2})"

        dil_hw_0, dil_hw_1, dil_hw_2 = dilation
        pad_hw_0 = (RS_0 // 2) * dil_hw_0
        pad_hw_1 = (RS_1 // 2) * dil_hw_1
        pad_hw_2 = (RS_2 // 2) * dil_hw_2

        PQ_0 = HW_0
        PQ_1 = HW_1
        PQ_2 = HW_2

        RS = RS_0 * RS_1 * RS_2
        if _fwd_debug:
            start_get_indice_pairs = torch.cuda.Event(enable_timing=True)
            end_get_indice_pairs = torch.cuda.Event(enable_timing=True)
            start_get_indice_pairs.record()

        out_indices, gather_idx, scatter_idx, indice_num_per_loc = get_indice_pairs(indices,
                        HW_0, HW_1, HW_2,
                        PQ_0, PQ_1, PQ_2,
                        RS_0, RS_1, RS_2,
                        str_hw_0, str_hw_1, str_hw_2,
                        pad_hw_0, pad_hw_1, pad_hw_2,
                        dil_hw_0, dil_hw_1, dil_hw_2,)
        if _fwd_debug:
            end_get_indice_pairs.record()
            torch.cuda.synchronize()
            get_indice_pairs_time = start_get_indice_pairs.elapsed_time(end_get_indice_pairs)
            print(f'get_indice_pairs_time: {get_indice_pairs_time:.3f}ms')


        if _fwd_debug:
            start_matmul = torch.cuda.Event(enable_timing=True)
            end_matmul = torch.cuda.Event(enable_timing=True)
            start_matmul.record()

        out_num_points = len(out_indices)
        out_features = torch.zeros(out_num_points, K, dtype=features.dtype, device=features.device)
       
        w = weight[:, RS_0 // 2, RS_1 // 2, RS_2 // 2, :]

        p_in = features[gather_idx[RS // 2]]
        p_out = p_in @ w.t()
        out_features[scatter_idx[RS // 2]] += p_out

        if not torch.all(indice_num_per_loc == 0):
            for rs in range(RS // 2):
                if indice_num_per_loc[rs] != 0:
                    rs_0 = rs // (RS_1 * RS_2)
                    rs_1 = (rs % (RS_1 * RS_2)) // RS_2
                    rs_2 = rs % RS_2
                    rotate_rs = RS - 1 - rs
                    rotate_rs_0 = rotate_rs // (RS_1 * RS_2)
                    rotate_rs_1 = (rotate_rs % (RS_1 * RS_2)) // RS_2
                    rotate_rs_2 = rotate_rs % RS_2

                    w = weight[:, rs_0, rs_1, rs_2, :]
                    idx = gather_idx[rs][gather_idx[rs] != -1]
                    p_in = features[idx]
                    p_out = p_in @ w.t()
                    idx = scatter_idx[rs][scatter_idx[rs] != -1]
                    out_features[idx] += p_out

                    w = weight[:, rotate_rs_0, rotate_rs_1, rotate_rs_2, :]
                    idx = gather_idx[rotate_rs][gather_idx[rotate_rs] != -1]
                    rotate_p_in = features[idx]
                    rotate_p_out = rotate_p_in @ w.t()
                    idx = scatter_idx[rotate_rs][scatter_idx[rotate_rs] != -1]
                    out_features[idx] += rotate_p_out


        if bias is not None:
            out_features += bias
        if _fwd_debug:
            end_matmul.record()
            torch.cuda.synchronize()
            matmul_time = start_matmul.elapsed_time(end_matmul)
            print(f'matmul_time: {matmul_time:.3f}ms')
            print(f'get_indice_pairs_time and matmul_time: {get_indice_pairs_time + matmul_time:.3f}ms')

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
            w = weight[:, RS_0 // 2, RS_1 // 2, RS_2 // 2, :]
            p_dout = dout_features[gather_idx[RS // 2]]
            p_din = p_dout @ w
            dfeatures[scatter_idx[RS // 2]] += p_din

            if not torch.all(indice_num_per_loc == 0):
                for rs in range(RS // 2):
                    if indice_num_per_loc[rs] != 0:
                        rs_0 = rs // (RS_1 * RS_2)
                        rs_1 = (rs % (RS_1 * RS_2)) // RS_2
                        rs_2 = rs % RS_2

                        rotate_rs = RS - 1 - rs
                        rotate_rs_0 = rotate_rs // (RS_1 * RS_2)
                        rotate_rs_1 = (rotate_rs % (RS_1 * RS_2)) // RS_2
                        rotate_rs_2 = rotate_rs % RS_2

                        w = weight[:, rs_0, rs_1, rs_2, :]
                        idx = gather_idx[rs][gather_idx[rs] != -1]
                        p_dout = dout_features[idx]
                        p_din = p_dout @ w
                        idx = scatter_idx[rs][scatter_idx[rs] != -1]
                        dfeatures[idx] += p_din

                        w = weight[:, rotate_rs_0, rotate_rs_1, rotate_rs_2, :]
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
            
            RS = RS_0 * RS_1 * RS_2
            p_dout = dout_features[dout_gather_idx[RS // 2]]
            p_in = features[input_gather_idx[RS // 2]]
            dweight[:, RS_0 // 2, RS_1 // 2, RS_2 // 2, :] += p_dout.t() @ p_in

            if not torch.all(indice_num_per_loc == 0):
                for rs in range(RS // 2):
                    if indice_num_per_loc[rs] != 0:
                        rs_0 = rs // (RS_1 * RS_2)
                        rs_1 = (rs % (RS_1 * RS_2)) // RS_2
                        rs_2 = rs % RS_2

                        rotate_rs = RS - 1 - rs
                        rotate_rs_0 = rotate_rs // (RS_1 * RS_2)
                        rotate_rs_1 = (rotate_rs % (RS_1 * RS_2)) // RS_2
                        rotate_rs_2 = rotate_rs % RS_2
                        
                        idx = dout_gather_idx[rs][dout_gather_idx[rs] != -1]
                        p_dout = dout_features[idx]
                        idx = input_gather_idx[rs][input_gather_idx[rs] != -1]
                        p_in = features[idx]
                        dweight[:, rs_0, rs_1, rs_2, :] += p_dout.t() @ p_in

                        idx = dout_gather_idx[rotate_rs][dout_gather_idx[rotate_rs] != -1]
                        p_dout = dout_features[idx]
                        idx = input_gather_idx[rotate_rs][input_gather_idx[rotate_rs] != -1]
                        p_in = features[idx]
                        dweight[:, rotate_rs_0, rotate_rs_1, rotate_rs_2, :] += p_dout.t() @ p_in

        dbias = None
        if ctx.bias_requires_grad:
            dbias = dout_features.sum(dim=0)
        
        return dfeatures, None, None, None, None, None, dweight, dbias, None, None, None, None


triton_submconv3d = _triton_submconv3d.apply