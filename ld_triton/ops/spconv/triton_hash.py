
import torch
import triton
import triton.language as tl


@triton.jit
def linear_hash_table_insert(table_key_ptr, table_val_ptr, key_ptr, val_ptr, num_act_in_real, hash_size, empty_key_int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    kv_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    key_ptrs = key_ptr +  (kv_start + offsets) % hash_size
    val_ptrs = val_ptr +  (kv_start + offsets) % hash_size
    kv_mask = (kv_start + offsets) < num_act_in_real
    key = tl.load(key_ptrs, mask=kv_mask, other=empty_key_int)
    val = tl.load(val_ptrs, mask=kv_mask, other=empty_key_int)

    # hash
    key ^= key >> 16
    key *= 0x85ebca6b
    key ^= key >> 13
    key *= 0xc2b2ae35
    key ^= key >> 16

    key = tl.where(kv_mask, key, empty_key_int).to(tl.uint32)
    slot = tl.where(kv_mask, key.to(tl.uint32) % hash_size.to(tl.uint32), hash_size)
    cmp = tl.full((BLOCK_SIZE, ), empty_key_int, dtype=tl.uint32)
    prev = tl.atomic_cas(table_key_ptr + slot, cmp, key)
    mask = tl.where(prev == cmp or prev == key, True, False) & kv_mask
    
    tl.store(table_val_ptr + slot, val, mask=mask)

    is_broke = tl.sum(mask.to(tl.int32), axis=0)
    total_mask = tl.where(cmp, False, False)
    while is_broke != 0:
        prev_mask = mask
        total_mask = total_mask | prev_mask

        slot = tl.where(kv_mask & ~prev_mask, (slot + 1) % hash_size.to(tl.uint32), hash_size)
        key = tl.where(prev_mask, empty_key_int, key).to(tl.uint32)
        prev = tl.atomic_cas(table_key_ptr + slot, cmp, key)
        mask = tl.where((prev == cmp or prev == key) and slot != hash_size, True, False) & kv_mask & ~total_mask
        tl.store(table_val_ptr + slot, val, mask=mask)
        is_broke = tl.sum(mask.to(tl.int32), axis=0)



@triton.jit
def linear_hash_table_lookup_offset(table_key_ptr, table_val_ptr, key_ptr, val_ptr, kv_size, hash_size, empty_key_int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    kv_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    key_ptrs = key_ptr +  (kv_start + offsets) % hash_size
    val_ptrs = val_ptr +  (kv_start + offsets) % hash_size
    kv_mask = (kv_start + offsets) < kv_size
    key = tl.load(key_ptrs, mask=kv_mask, other=empty_key_int)
    # hash
    key ^= key >> 16
    key *= 0x85ebca6b
    key ^= key >> 13
    key *= 0xc2b2ae35
    key ^= key >> 16
    
    key = tl.where(kv_mask, key, empty_key_int).to(tl.uint32)
    slot = tl.where(kv_mask, key.to(tl.uint32) % hash_size.to(tl.uint32), hash_size)
    key_target = tl.load(table_key_ptr + slot, mask=kv_mask, other=empty_key_int)
    mask = (key_target == key) & kv_mask
    val = tl.load(table_val_ptr + slot, mask=mask, other=empty_key_int)
    tl.store(val_ptrs, val, mask=mask)

    is_broke = tl.sum(mask.to(tl.uint32), axis=0)
    total_mask = tl.where(mask, False, False)

    while is_broke != 0:
        prev_mask = mask
        total_mask = total_mask | prev_mask
        slot = tl.where(kv_mask & ~prev_mask, (slot + 1) % hash_size.to(tl.uint32), hash_size)
        key_target = tl.load(table_key_ptr + slot, mask=kv_mask, other=empty_key_int)
        mask = (key_target == key) & kv_mask & ~total_mask
        val = tl.load(table_val_ptr + slot, mask=mask, other=empty_key_int)
        tl.store(val_ptrs, val, mask=mask)
        is_broke = tl.sum(mask.to(tl.uint32), axis=0)   


# Adapted from https://github.com/FindDefinition/cumm/blob/v0.4.10/include/tensorview/hash/linear.cu.h
class LinearHashTableSplit(object):
    def __init__(self, kv_size: int, rate: float = 2.0):
        hash_size = int(kv_size * rate)
        self._kv_size = kv_size
        self._rate = rate
        self._hash_size = hash_size
        self._empty_key_int = 0xFFFFFFFF
        self._table_key = torch.full((hash_size + 1, ), self._empty_key_int, dtype=torch.uint32, device='cuda')
        self._table_val = torch.full((hash_size + 1, ), self._empty_key_int, dtype=torch.uint32, device='cuda')
        self._dtype = torch.uint32
        self._device = 'cuda'


    def insert(self, key: torch.Tensor, value: torch.Tensor):
        assert key.dtype == self._dtype , f'key only support uint32'
        assert value.dtype == self._dtype , f'value only support uint32'
        assert len(key) == len(value), f'key and value must have the same length'
        assert len(key.shape) == 1, f'key must be 1D'
        assert len(value.shape) == 1, f'value must be 1D'
        self.insert_raw(key, value)

    def insert_raw(self, key: torch.Tensor, val: torch.Tensor):
        BLOCK_SIZE = 32
        grid = lambda META: (triton.cdiv(len(key), BLOCK_SIZE),)
        linear_hash_table_insert[grid](self._table_key, 
                                       self._table_val,
                                       key,
                                       val,
                                       len(key),
                                       self._hash_size,
                                       self._empty_key_int,
                                       BLOCK_SIZE)
 
    def lookup_offset(self, key: torch.Tensor):
        assert key.dtype == self._dtype , f'key only support uint32'
        assert len(key.shape) == 1, f'key must be 1D'
        return self.lookup_offset_raw(key)
    
    def lookup_offset_raw(self, key: torch.Tensor):
        assert key.dtype == self._dtype , f'key only support uint32'
        assert key.shape[0] == len(key), f'key must be 1D'
        BLOCK_SIZE = 32
        kv_size = len(key)
        val = torch.full((kv_size, ), self._empty_key_int, dtype=self._dtype , device=self._device )
        grid = lambda META: (triton.cdiv(kv_size, BLOCK_SIZE),)
        linear_hash_table_lookup_offset[grid](self._table_key,
                                        self._table_val,
                                        key,
                                        val,
                                        len(key),
                                        self._hash_size,
                                        self._empty_key_int,
                                        BLOCK_SIZE)
        return val
    
    def __str__(self):
        return f'LinearHashTableSplit(kv_size={self._kv_size}, rate={self._rate}, hash_size={self._hash_size}, empty_key_int={self._empty_key_int}, table_key={self._table_key}, table_val={self._table_val})'
    

def build_hash(indices: torch.Tensor, spatial_shape, rate: float = 2.0):
    BLOCK_SIZE = 32
    kv_size = indices.shape[0]
    indices = indices.to(torch.int32)
    key = indices[:, 0] * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] + \
            indices[:, 1] * spatial_shape[0] * spatial_shape[1] + \
            indices[:, 2] * spatial_shape[0] + \
            indices[:, 3]
    val = torch.arange(0, kv_size, dtype=torch.int32, device=indices.device)
    
    table = LinearHashTableSplit(kv_size, rate = 2.0)
    table.insert(key.to(torch.uint32), val.to(torch.uint32))
    return table


def lookup_offset(table, indices):
    indices = indices.to(torch.int32)
    key = indices[:, 0] * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] + \
            indices[:, 1] * spatial_shape[0] * spatial_shape[1] + \
            indices[:, 2] * spatial_shape[0] + \
            indices[:, 3]
    val = table.lookup_offset(key.to(torch.uint32))
    return val


if __name__ == '__main__':
    spatial_shape = [23, 23, 23]
    indices = torch.tensor([(0, 3, 7, 1), (0, 6, 1, 2), (0, 18, 9, 3), (0, 15, 7, 4), (0, 7, 8, 5), (0, 20, 7, 6), (0, 3, 13, 7)], 
                           device='cuda', 
                           dtype=torch.uint32)

    table = build_hash(indices, spatial_shape)
    print(table)
    val = lookup_offset(table, indices)
    print(val)

    N = 100000
    table = LinearHashTableSplit(N, rate = 2.0)
    key = torch.arange(0, N, dtype=torch.int32, device='cuda')
    val = torch.arange(0, N, dtype=torch.int32, device='cuda')
    key_idx = torch.randperm(N)
    key = key[key_idx]
    val_idx = torch.randperm(N)
    val = val[val_idx]
    table.insert(key.to(torch.uint32), val.to(torch.uint32))
    val = table.lookup_offset(key.to(torch.uint32))


    d_key = key.cpu().tolist()
    d_val = val.cpu().tolist()
    d = {}
    for i in range(N):
        d[d_key[i]] = d_val[i]
    d_val = [d[i] for i in d_key]
    
    assert torch.allclose(val, torch.tensor(d_val, dtype=torch.uint32, device='cuda'))




        
