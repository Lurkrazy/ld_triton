
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
    # prev = tl.atomic_cas(pointer, cmp, val), if pointer address is illegal, the prev = 0,
    # when the key = 0, hash(key) = 0, it will cause some bug
    # key = 0 or key = 0xFFFFFFFF the hash(k) = 0
    # we shift the key by 1, so the key = 0 will not cause the hash(k) = 0
    key += 1

    key = tl.where(kv_mask, key, empty_key_int).to(tl.uint32)
    slot = tl.where(kv_mask, key.to(tl.uint32) % hash_size.to(tl.uint32), hash_size)
    cmp = tl.full((BLOCK_SIZE, ), empty_key_int, dtype=tl.uint32)
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
        
        key = tl.where(total_mask, empty_key_int, key).to(tl.uint32)
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


@triton.jit
def linear_hash_table_lookup_offset(table_key_ptr, table_val_ptr, key_ptr, slot_ptr, val_ptr, kv_size, hash_size, empty_key_int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    kv_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    key_ptrs = key_ptr +  (kv_start + offsets) % hash_size
    val_ptrs = val_ptr +  (kv_start + offsets) % hash_size
    slot_ptrs = slot_ptr + (kv_start + offsets) % hash_size
    kv_mask = (kv_start + offsets) < kv_size
    key = tl.load(key_ptrs, mask=kv_mask, other=empty_key_int)
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

    key = tl.where(kv_mask, key, empty_key_int).to(tl.uint32)
    
    slot = tl.where(kv_mask, key.to(tl.uint32) % hash_size.to(tl.uint32), hash_size)
    key_target = tl.load(table_key_ptr + slot, mask=kv_mask, other=empty_key_int + 1)
    mask = (key_target == key) & kv_mask
    val = tl.load(table_val_ptr + slot, mask=mask, other=empty_key_int)
    tl.store(val_ptrs, val, mask=mask)
    tl.store(slot_ptrs, slot, mask=mask)

    is_broke = tl.sum(mask.to(tl.int32), axis=0)
    total_mask = tl.where(mask, False, False)

    while is_broke != 0:
        prev_mask = mask
        # record the slot is used
        total_mask = total_mask | prev_mask
        slot = tl.where(kv_mask & ~total_mask, (slot + 1) % hash_size.to(tl.uint32), hash_size)
        # for an example
        # table key: 1, 2, 3, 4, 5, 6, 7, 8
        # find  key: 9, 9, 3, 4, 5, 6, 7, 8
        #           2, 3, 4, 5, 6, 7, 8, 9
        # key_target other value is empty_key_int + 1
        key_target = tl.load(table_key_ptr + slot, mask=(kv_mask & ~total_mask), other=empty_key_int + 1)
        # find a slot, this slot has a key is the same as the current key, break
        mask_1 = (key_target == key) & kv_mask & ~total_mask
        # find a slot, this slot has a key is the same as empty_key_int, break
        mask_2 = (key_target == empty_key_int) & kv_mask & ~total_mask
        # find a slot, this slot has a key is diffrent of current key and empty_key_int, continue to find the next slot
        mask_3 = (key_target != key) & (key_target != empty_key_int) & (key_target != empty_key_int+ 1 ) & kv_mask & ~total_mask
        # find a slot, key is the same as the current key or empty_key_int, record the slot is used
        mask = mask_1 | mask_2

        # find a slot, key is the same as the current key, store the value
        is_store = tl.sum(mask_1.to(tl.int32), axis=0)
        mask_broke = mask_1 | mask_2 | mask_3
        is_broke = tl.sum(mask_broke.to(tl.int32), axis=0)
        if is_store != 0:
            val = tl.load(table_val_ptr + slot, mask=mask_1, other=empty_key_int)
            tl.store(val_ptrs, val, mask=mask_1)
            tl.store(slot_ptrs, slot, mask=mask_1)
        is_store_2 = tl.sum(mask_2.to(tl.int32), axis=0)
        if is_store_2 != 0:
            tl.store(slot_ptrs, -1, mask=mask_2)


# Adapted from https://github.com/FindDefinition/cumm/blob/v0.4.10/include/tensorview/hash/linear.cu.h
class LinearHashTableSplit(object):
    def __init__(self, kv_size: int, rate: float = 2.0):
        hash_size = int(kv_size * rate)
        self._kv_size = kv_size
        self._rate = rate
        self._hash_size = hash_size
        self._empty_key_int = 0xFFFFFFFF - 1
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
        # slot = torch.full((kv_size, ), -1, dtype=torch.int32 , device=self._device )
        slot = torch.empty((kv_size, ), dtype=torch.int32 , device=self._device )
        grid = lambda META: (triton.cdiv(kv_size, BLOCK_SIZE),)
        linear_hash_table_lookup_offset[grid](self._table_key,
                                        self._table_val,
                                        key,
                                        slot,
                                        val,
                                        len(key),
                                        self._hash_size,
                                        self._empty_key_int,
                                        BLOCK_SIZE)
        return slot, val
    
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
    key = torch.cat([key[0:6] - 1, key], dim=0)
    slot, val = table.lookup_offset(key.to(torch.uint32))
    print(f'slot: {slot}')
    print(f'val: {val}')
    return val


@triton.jit
def test_cas_kerenl(key_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    cmp = tl.full((BLOCK_SIZE, ), 3, dtype=tl.uint32)
    val = tl.full((BLOCK_SIZE, ), 100, dtype=tl.uint32)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < BLOCK_SIZE
    prev = tl.atomic_cas(key_ptr + offs, cmp, val)
    tl.device_print('cmp', cmp)
    tl.device_print('val', val)
    tl.device_print('prev', prev)

    available_prev = tl.where(mask, prev, 0xFFFF + 2)
    tl.device_print('available_prev', available_prev)


def test_cas():
    BLOCK_SIZE = 8
    N = 8
    key = torch.arange(1, N + 1, dtype=torch.int32, device='cuda')
    test_cas_kerenl[(1, )](key, BLOCK_SIZE)


# test_cas()

if __name__ == '__main__':
    spatial_shape = [23, 23, 23]
    # (0, 6, 1, 1), (0, 18, 9, 1), (0, 15, 7, 1), (0, 7, 8, 1), (0, 20, 7, 1), (0, 19, 6, 0), (0, 3, 13, 1), (0, 4, 14, 2), (0, 3, 7, 1), 
    indices = torch.tensor([(0, 3, 7, 1), (0, 6, 1, 2), (0, 18, 9, 3), (0, 15, 7, 4), (0, 7, 8, 5), (0, 20, 7, 6), (0, 3, 13, 7)], 
                           device='cuda', 
                           dtype=torch.uint32)

    indices = torch.tensor([(0, 6, 1, 1), (0, 18, 9, 1), (0, 15, 7, 1), (0, 7, 8, 1), (0, 20, 7, 1), (0, 19, 6, 0), (0, 3, 13, 1), (0, 4, 14, 2), (0, 3, 7, 1), ], 
                           device='cuda', 
                           dtype=torch.uint32)
    
    indices = torch.tensor([(0, 6, 1, 1), (0, 18, 9, 1), (0, 15, 7, 1), (0, 7, 8, 1), (0, 20, 7, 1), (0, 19, 6, 0), (0, 3, 13, 1), (0, 4, 14, 2), (0, 3, 7, 1), ], 
                           device='cuda', dtype=torch.int32)

    table = build_hash(indices, spatial_shape)
    print(table)
    val = lookup_offset(table, indices)
    print(val)

    N = 100000
    table = LinearHashTableSplit(N + 1, rate = 2.0)
    key = torch.arange(1 , N + 1, dtype=torch.int32, device='cuda')
    val = torch.arange(1 , N + 1, dtype=torch.int32, device='cuda')
    perm_idx = torch.randperm(N)
    perm_key = key[perm_idx].contiguous()
    perm_val = val[perm_idx].contiguous()
    table.insert(perm_key.to(torch.uint32), perm_val.to(torch.uint32))
    res_key = torch.arange(1 , N + 1, dtype=torch.int32, device='cuda')
    _, res_val = table.lookup_offset(res_key.to(torch.uint32))

    assert torch.allclose(val.to(torch.uint32), res_val)




        
