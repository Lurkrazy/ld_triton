
import torch
import triton
import triton.language as tl


@triton.jit
def linear_hash_table_insert(table_key_ptr, table_val_ptr, key_ptr, val_ptr, num_act_in_real, hash_size, empty_key_uint, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    kv_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    key_ptrs = key_ptr +  (kv_start + offsets) % hash_size
    val_ptrs = val_ptr +  (kv_start + offsets) % hash_size
    kv_mask = (kv_start + offsets) < num_act_in_real
    key = tl.load(key_ptrs, mask=kv_mask, other=empty_key_uint)
    val = tl.load(val_ptrs, mask=kv_mask, other=empty_key_uint)

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


@triton.jit
def linear_hash_table_lookup_offset(table_key_ptr, table_val_ptr, key_ptr, slot_ptr, val_ptr, kv_size, hash_size, empty_key_uint, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    kv_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    key_ptrs = key_ptr +  (kv_start + offsets) % hash_size
    val_ptrs = val_ptr +  (kv_start + offsets) % hash_size
    slot_ptrs = slot_ptr + (kv_start + offsets) % hash_size
    kv_mask = (kv_start + offsets) < kv_size
    key = tl.load(key_ptrs, mask=kv_mask, other=empty_key_uint)
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
    tl.store(val_ptrs, val, mask=mask_1)
    tl.store(slot_ptrs, slot, mask=mask_1)
    
    is_broke = tl.sum(broke_mask.to(tl.int32), axis=0)
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
        self._empty_key_uint = 0xFFFFFFFF - 1
        self._table_key = torch.full((hash_size + 1, ), self._empty_key_uint, dtype=torch.uint32, device='cuda')
        self._table_val = torch.full((hash_size + 1, ), self._empty_key_uint, dtype=torch.uint32, device='cuda')
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
        num_act_in_real = key.shape[0]
        if num_act_in_real < 1024:
            BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(num_act_in_real, 32) * 32)
        else:
            BLOCK_SIZE = 1024
        grid = lambda META: (triton.cdiv(len(key), BLOCK_SIZE),)
        linear_hash_table_insert[grid](self._table_key, 
                                       self._table_val,
                                       key,
                                       val,
                                       num_act_in_real,
                                       self._hash_size,
                                       self._empty_key_uint,
                                       BLOCK_SIZE)
 
    def lookup_offset(self, key: torch.Tensor):
        assert key.dtype == self._dtype , f'key only support uint32'
        assert len(key.shape) == 1, f'key must be 1D'
        return self.lookup_offset_raw(key)
    
    def lookup_offset_raw(self, key: torch.Tensor):
        assert key.dtype == self._dtype , f'key only support uint32'
        assert key.shape[0] == len(key), f'key must be 1D'
        kv_size = len(key)
        if kv_size < 1024:
            BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(kv_size, 32) * 32)
        else:
            BLOCK_SIZE = 1024
        
        val = torch.full((kv_size, ), self._empty_key_uint, dtype=self._dtype , device=self._device )
        slot = torch.full((kv_size, ), -1, dtype=torch.int32 , device=self._device )
        # slot = torch.empty((kv_size, ), dtype=torch.int32 , device=self._device )
        grid = lambda META: (triton.cdiv(kv_size, BLOCK_SIZE),)
        linear_hash_table_lookup_offset[grid](self._table_key,
                                        self._table_val,
                                        key,
                                        slot,
                                        val,
                                        len(key),
                                        self._hash_size,
                                        self._empty_key_uint,
                                        BLOCK_SIZE)
        return slot, val
    
    def __str__(self):
        return f'LinearHashTableSplit(kv_size={self._kv_size}, rate={self._rate}, hash_size={self._hash_size}, empty_key_uint={self._empty_key_uint}, table_key={self._table_key}, table_val={self._table_val})'
    

def build_hash(indices: torch.Tensor, spatial_shape, rate: float = 2.0):
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


def lookup_offset(table: LinearHashTableSplit, indices, spatial_shape):
    indices = indices.to(torch.int32)
    key = indices[:, 0] * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] + \
            indices[:, 1] * spatial_shape[1] * spatial_shape[2] + \
            indices[:, 2] * spatial_shape[2] + \
            indices[:, 3]

    slot, val = table.lookup_offset(key.to(torch.uint32))
    return slot, val


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
    spatial_shape = [23, 24, 25]
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
    slot, val = lookup_offset(table, indices, spatial_shape)
    print(val)

    indices = torch.tensor([[   0,   19,  903, 1065],
        [   0,   39,  640,  724],
        [   0,   25,  843,  845],
        [   0,   13,  661,  837],
        [   0,   22, 1205, 1124],
        [   0,   52,  733,  900],
        [   0,   38, 1014,  447],
        [   0,   34,  943,  552],
        [   0,   38,  677,  715],
        [   0,   36,  661,  870],
        [   0,   35,  715,  692],
        [   0,   19, 1032, 1166],
        [   0,   33, 1089,  918],
        [   0,   40,  688,  722],
        [   0,   26,  748,  683],
        [   0,   22,  764,  893],
        [   0,   44, 1093,  758],
        [   0,   37,  766,  566],
        [   0,   29,  688,  670],
        [   0,   30,  888,  792],
        [   0,   46,  870,  950],
        [   0,   39, 1008, 1217],
        [   0,   28,  671,  521],
        [   0,   51, 1083,  960],
        [   0,   42,  666, 1101],
        [   0,   33, 1001, 1224],
        [   0,   54,  983,  451],
        [   0,   30,  673,  219],
        [   0,   36, 1055,  968],
        [   0,   23,  631,  926],
        [   0,   32,  838,  849],
        [   0,   42,  713,  571],
        [   0,   21,  776,  780],
        [   0,   15,  670,  770],
        [   0,   20,  862,  822],
        [   0,   51,  688, 1109],
        [   0,   41,  684,  427],
        [   0,   35, 1129,  934],
        [   0,   50,  978,  365],
        [   0,   24,  722,  318],
        [   0,   45,  966, 1128],
        [   0,   23, 1037,  673],
        [   0,   25,  741,  579],
        [   0,   41,  645,  536],
        [   0,   24,  913,  788],
        [   0,   47, 1064,  744],
        [   0,   39,  731,  891],
        [   0,   48, 1085,  908],
        [   0,   40,  909,  606],
        [   0,   42,  642, 1041],
        [   0,   20,  854,  649],
        [   0,   56,  969,  295],
        [   0,   25,  923,  787],
        [   0,   41,  612,  738],
        [   0,   12,  694,  915],
        [   0,   37,  650, 1099],
        [   0,   39,  738,  590],
        [   0,   46,  731,  580],
        [   0,   57, 1100,  480],
        [   0,   55, 1075,  423],
        [   0,   20,  874,  848],
        [   0,   29,  814,  701],
        [   0,   56, 1170, 1163],
        [   0,   36, 1093, 1273],
        [   0,   25,  757,  709],
        [   0,   24,  722,  262],
        [   0,   37,  675, 1050],
        [   0,   48,  747, 1007],
        [   0,   23, 1060,  376],
        [   0,   63,  898,  987],
        [   0,   21,  914,  785],
        [   0,   17,  918,  869],
        [   0,   34,  971,  358],
        [   0,   19,  902,  936],
        [   0,   31,  879,  622],
        [   0,   52,  942,  489],
        [   0,   42,  712, 1107],
        [   0,   44,  654,  764],
        [   0,   41, 1139, 1126],
        [   0,   70, 1000,  832],
        [   0,   39,  721,  542],
        [   0,   47,  822,  557],
        [   0,   16,  706,  800],
        [   0,   52, 1101,  917],
        [   0,   40,  712, 1007],
        [   0,   48,  966,  569],
        [   0,   34,  836,  851],
        [   0,   36,  634,  733],
        [   0,   19, 1026, 1062],
        [   0,   19,  610,  921],
        [   0,   21,  618,  881],
        [   0,   50,  633,  807],
        [   0,   25, 1025,  975],
        [   0,   27,  834,  855],
        [   0,   27,  733,  716],
        [   0,   38, 1043, 1186],
        [   0,   19,  898,  798],
        [   0,   50,  764, 1010],
        [   0,   19,  966,  647],
        [   0,   34,  726,  812],
        [   0,   50, 1134, 1160],
        [   0,   19,  724,  620],
        [   0,   39,  734,  537],
        [   0,   34,  652,  868],
        [   0,   59,  790,   96],
        [   0,   19,  950,  981],
        [   0,   19,  895,  997],
        [   0,   41,  637,  593],
        [   0,   19,  901,  797],
        [   0,   32,  670, 1027],
        [   0,   20,  841,  482],
        [   0,   43,  711,  760],
        [   0,   17,  969,  800],
        [   0,   46, 1065,  590],
        [   0,   31,  936,  831],
        [   0,   37, 1047,  566],
        [   0,   22,  789,  911],
        [   0,   21,  745,  789],
        [   0,   33,  679,  672],
        [   0,   21, 1065,  581],
        [   0,   26, 1152,  900],
        [   0,   14, 1019, 1257],
        [   0,   42, 1086, 1280],
        [   0,   38,  697,  874],
        [   0,   46,  651,  782],
        [   0,   52, 1175, 1175],
        [   0,   45,  622,  636],
        [   0,   49,  719, 1052],
        [   0,   20,  831, 1204],
        [   0,   40, 1088,  725],
        [   0,   16,  761,  886],
        [   0,   37,  682,  840],
        [   0,   20,  816,  785],
        [   0,   17,  879,  805],
        [   0,   19,  945,  995],
        [   0,   17,  674,  848],
        [   0,   22,  631,  908],
        [   0,   45,  899,  956],
        [   0,   33, 1043,  778],
        [   0,   60, 1141, 1182],
        [   0,   28, 1024,  860],
        [   0,   35, 1067,  656],
        [   0,   19,  917,  878],
        [   0,   20, 1085,  887],
        [   0,   19,  880,  706],
        [   0,   20,  884,  776],
        [   0,   29,  914,  852],
        [   0,   24,  646,  859],
        [   0,   38, 1109,  938],
        [   0,   24,  620,  930],
        [   0,   20,  858,  863],
        [   0,   15, 1003, 1087],
        [   0,   16, 1016,  901],
        [   0,   44,  740,  752],
        [   0,   27,  654,  688],
        [   0,   11,  711,  804],
        [   0,   48,  605,  959],
        [   0,   45,  619,  733],
        [   0,   38,  668,  740],
        [   0,   19,  798,  838],
        [   0,   39,  699,  712],
        [   0,   46,  691,  543],
        [   0,   34,  661,  790],
        [   0,   25, 1043,  149],
        [   0,   29,  937,  824],
        [   0,   19,  913,  883],
        [   0,   23,  903,  599],
        [   0,   43, 1095,  733],
        [   0,   32, 1021,  446],
        [   0,   36,  625,  735],
        [   0,   22, 1052,  542],
        [   0,   34, 1055,  675],
        [   0,   50,  635,  896],
        [   0,   20,  860,  397],
        [   0,   18,  985,  705],
        [   0,   32,  802,  869],
        [   0,   59,  976,  772],
        [   0,   54, 1008,  551],
        [   0,   38,  643,  854],
        [   0,   43,  703,  832],
        [   0,   45, 1087,  843],
        [   0,   35,  714,  888],
        [   0,   47, 1082,  903],
        [   0,   23,  656,  786],
        [   0,   17,  902, 1009],
        [   0,   40,  612,  862],
        [   0,   54,  980,  363],
        [   0,   24,  670,  367],
        [   0,   22,  883,  775],
        [   0,   20,  888, 1325],
        [   0,   26, 1076,  690],
        [   0,   35,  679,  687],
        [   0,   28,  627,  795],
        [   0,   41,  786,  541],
        [   0,   19,  875, 1301],
        [   0,   20,  818,  726],
        [   0,   39, 1109,  937],
        [   0,   31,  680,  724],
        [   0,   28,  687,  637],
        [   0,   21,  751,  720],
        [   0,   45,  904,  731],
        [   0,   32,  678, 1044],
        [   0,   53, 1105,  911],
        [   0,   21,  830,  336],
        [   0,   31,  865,  844],
        [   0,   43,  905,  692],
        [   0,   51, 1089,  966],
        [   0,   19,  707,  736],
        [   0,   19,  765,  718],
        [   0,   47,  986,  551],
        [   0,   26,  714,  725],
        [   0,   28,  773,  925],
        [   0,   20,  814, 1167],
        [   0,   47, 1054,  590],
        [   0,   21,  811,  329],
        [   0,   32,  636,  745],
        [   0,   38, 1203, 1113],
        [   0,   19, 1043, 1443],
        [   0,   17,  989,  959],
        [   0,   48,  666,  880],
        [   0,   49,  628,  883],
        [   0,   33,  734,  854],
        [   0,   30,  833,  859],
        [   0,   25,  893,  769],
        [   0,   41, 1109,  929],
        [   0,   37, 1082,  984],
        [   0,   26,  915,  795],
        [   0,   14,  967, 1394],
        [   0,   49,  734, 1007],
        [   0,   19,  767,  738],
        [   0,   42,  958,  468],
        [   0,   23, 1102,  763],
        [   0,   26,  917,  964],
        [   0,   45, 1058,  487],
        [   0,   18,  905,  934],
        [   0,   44,  779,  754],
        [   0,   22,  703,  854],
        [   0,   51, 1077,  819],
        [   0,   19,  794,  909],
        [   0,   28,  643,  913],
        [   0,   38, 1078,  967],
        [   0,   27, 1060,  654],
        [   0,   52,  630, 1080],
        [   0,   15,  974, 1110],
        [   0,   44,  753,  807],
        [   0,   21, 1046,  579],
        [   0,   24,  685, 1035],
        [   0,   27,  638,  796],
        [   0,   28, 1038,  718],
        [   0,   27, 1084,  966],
        [   0,   57,  982, 1177],
        [   0,   43,  617, 1017],
        [   0,   19,  929,  961],
        [   0,   19,  824,  727],
        [   0,   35,  935,  545],
        [   0,   31,  656,  346],
        [   0,   28, 1068,  812],
        [   0,   37, 1079,  802],
        [   0,   20,  826,  813],
        [   0,   48,  754,  315],
        [   0,   28,  774,  694],
        [   0,   20,  855, 1118],
        [   0,   20, 1061,  763],
        [   0,   43,  917,  554],
        [   0,   23,  765,  895],
        [   0,   19, 1028,  523],
        [   0,   47,  729,  886],
        [   0,   29,  722,  882],
        [   0,   19,  851,  834],
        [   0,   20, 1061,  884],
        [   0,   19, 1032,  982],
        [   0,   52,  736,  530],
        [   0,   39,  647,  857],
        [   0,   38, 1036,  690],
        [   0,   33,  756,  685],
        [   0,   44,  689,  788],
        [   0,   30, 1040,  653],
        [   0,   32, 1089,  967],
        [   0,   31,  954, 1055],
        [   0,   45, 1001, 1164],
        [   0,   36, 1118,  723],
        [   0,   33,  632,  672],
        [   0,   25, 1018,  627],
        [   0,   43,  665, 1050],
        [   0,   36, 1075,  462],
        [   0,   21, 1117, 1283],
        [   0,   18, 1005, 1055],
        [   0,   29,  722,  837],
        [   0,   51,  711, 1006],
        [   0,   48, 1305, 1135],
        [   0,   48,  904, 1001],
        [   0,   21,  968, 1177],
        [   0,   21,  793,  876],
        [   0,   34,  664,  699],
        [   0,   23,  684,  795],
        [   0,   35, 1057,  739],
        [   0,   18,  908,  651],
        [   0,   35, 1045,  589],
        [   0,   39, 1101, 1006],
        [   0,   32,  643,  691],
        [   0,   51, 1065,  837],
        [   0,   24,  883,  717],
        [   0,   22, 1076,  397],
        [   0,   49, 1015,  682],
        [   0,   42,  638,  392],
        [   0,   30,  945,  735],
        [   0,   49, 1059,  703],
        [   0,   42,  745,  808],
        [   0,   45, 1046,  850],
        [   0,   31,  874,  547],
        [   0,   46,  662,  898],
        [   0,   29, 1087,  287],
        [   0,   41,  670,  852],
        [   0,   45, 1044, 1116],
        [   0,   22,  746, 1263],
        [   0,   38,  698,  826],
        [   0,   25, 1009,  837],
        [   0,   24,  667,  673],
        [   0,   12,  663,  751],
        [   0,   19,  806,  803],
        [   0,   20,  782,  724],
        [   0,   36, 1001,  557],
        [   0,   52,  932,  459],
        [   0,   28,  769,  969],
        [   0,   24,  981, 1459],
        [   0,   33,  693,  293],
        [   0,   44, 1141,  454],
        [   0,   19,  807,  572],
        [   0,   43, 1133, 1043],
        [   0,   20,  819,  844],
        [   0,   26,  645,  749],
        [   0,   35,  608,  851],
        [   0,   10,  687,  889],
        [   0,   35,  614,  766],
        [   0,   21,  774,  847],
        [   0,   49, 1048,  807],
        [   0,   35, 1125,  964],
        [   0,   16,  715,  857],
        [   0,   40,  658,  843],
        [   0,   36,  653,  882],
        [   0,   45,  912,  957],
        [   0,   32, 1040,  694],
        [   0,   44,  583,  847],
        [   0,   27, 1167, 1183],
        [   0,   48,  773,  571],
        [   0,   14,  995, 1254],
        [   0,   33,  672,  886],
        [   0,   19,  873,  909],
        [   0,   26,  730, 1191],
        [   0,   35,  751,  660],
        [   0,   18,  963,  846],
        [   0,   20,  808,  642]], device='cuda', dtype=torch.int32)
    spatial_shape = [80, 1600, 1600]
    table = build_hash(indices, spatial_shape)
    print(table)
    lookup_indices = indices[148: 149, :]
    lookup_indices = indices[:, :]
    slot, val = lookup_offset(table, lookup_indices, spatial_shape)
    print(f'lookup_indices: {lookup_indices}')
    print(f'slot: {slot}')
    print(f'val: {val}')


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




        
