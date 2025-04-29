
import torch
import triton
import triton.language as tl


@triton.jit
def bandwidth_kerenl(input_ptr, output_ptr, num_elements, NUM_SMS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(num_elements, BLOCK_SIZE)
    tiles_per_SM = num_tiles // NUM_SMS
    if pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    
    tile_id = pid

    for _ in range(0, tiles_per_SM):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_elements
        input = tl.load(input_ptr + offs, mask=mask, other=0.0, cache_modifier='.cv')
        input *= 2
        tl.store(output_ptr + offs, input, mask=mask, cache_modifier='.wt')
        tile_id += NUM_SMS
    

def bandwidth(input, output, stream=None):
    num_elements = input.numel()
    print(f"num_elements: {num_elements}")
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE = 1024
    grid = (NUM_SMS,)
    
    with torch.cuda.stream(stream):
        bandwidth_kerenl[grid](
            input,
            output,
            num_elements,
            NUM_SMS,
            BLOCK_SIZE,
        )


if __name__ == "__main__":
    
    hardware_tflops = 0
    hardware_gmemorys = torch.cuda.get_device_properties('cuda').total_memory / 1024 / 1024 / 1024
    hardware_L2_cache_size = torch.cuda.get_device_properties('cuda').L2_cache_size

    num_elements = 1024 * 1024 * 1024
    dtype = torch.float16
    input = torch.randn((num_elements), device='cuda', dtype=dtype)
    output = torch.empty((num_elements), device='cuda', dtype=dtype)
    stream = torch.cuda.Stream()
    bandwidth(input, output, stream)
    # assert torch.allclose(input, output)
    ms = triton.testing.do_bench(lambda: bandwidth(input, output, stream))
    torch.cuda.synchronize()
    gb = (2 * num_elements * dtype.itemsize) / 1024 / 1024 / 1024
    bandwidth_per_second = gb / (ms * 1e-3)
    print(f"ms: {ms}, gb: {gb}, bandwidth_per_second: {bandwidth_per_second:.3f} GB/s")


