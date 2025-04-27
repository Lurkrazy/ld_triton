
def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0


def _step_to_micro_batch(stages, stage_id, step_id):
    if _is_even(step_id) and _is_even(stage_id):
        micro_batch_id = _even_step_forward_id(stage_id, step_id)
        is_forward = True

    elif _is_odd(step_id) and _is_odd(stage_id):
        micro_batch_id = _odd_step_forward_id(stage_id, step_id)
        is_forward = True

    elif _is_even(step_id) and _is_odd(stage_id):
        micro_batch_id = _even_step_backward_id(stages, stage_id, step_id)
        is_forward = False

    elif _is_odd(step_id) and _is_even(stage_id):
        micro_batch_id = _odd_step_backward_id(stages, stage_id, step_id)
        is_forward = False

    else:
        assert False

    return micro_batch_id, is_forward

def _even_step_forward_id(stage_id, step_id):
    base = step_id // 2
    micro_batch_id = int(base - stage_id // 2)
    return micro_batch_id

def _odd_step_forward_id(stage_id, step_id):
    base = (step_id - 1) // 2
    micro_batch_id = int(base - stage_id // 2)
    return micro_batch_id

def _even_step_backward_id(stages, stage_id, step_id):
    base = step_id // 2
    micro_batch_id = int(base - stages + (stage_id + 1) // 2)
    return micro_batch_id

def _odd_step_backward_id(stages, stage_id, step_id):
    base = ((step_id - 1) // 2) - stages + 1
    micro_batch_id = int(base + stage_id // 2)
    return micro_batch_id

def _valid_micro_batch(micro_batches, micro_batch_id):
    return 0 <= micro_batch_id < micro_batches

def graphic_pipe_dream(stages: int, micro_batches: int):
    total_steps = 2 * (micro_batches + stages - 1)
    for step_id in range(total_steps):
        print(f"| {step_id:2} ", end="")
    print("|")
    for stage_id in range(stages):
        for step_id in range(total_steps):
            micro_batch_id, is_forward = _step_to_micro_batch(stages, stage_id, step_id)
            if is_forward and _valid_micro_batch(micro_batches, micro_batch_id):
                print(f"| {micro_batch_id}f ", end="")
            elif _valid_micro_batch(micro_batches, micro_batch_id):
                print(f"| {micro_batch_id}b ", end="")
            else:
                print(f"| *  ", end="")
            
            # for micro_batch_id in range(num_micro_batches):
            #     if micro_batch_id == 0:
            #         print(f'| {micro_batch_id} ', end="")
            #     else:
            #         print(f"| * ", end="")
        print("|")
    
    total_compute = total_steps * stages
    valid_compute = 2 * micro_batches * stages
    print(f"total compute: {total_compute}")
    print(f"valid compute: {valid_compute}")
    print(f"bubble: {total_compute - valid_compute}")
    print(f"bubble ratio: {(total_compute - valid_compute) / total_compute:.2%}")

def graphic_gpipe(stages: int, micro_batches: int):
    total_steps = 2 * (micro_batches + stages - 1)
    for step_id in range(total_steps):
        print(f"| {step_id:3} ", end="")
    print("|")
    for stage_id in range(stages):
        for step_id in range(total_steps):
            if step_id < micro_batches + stages - 1:
                micro_batch_id = step_id - stage_id
                if micro_batch_id < 0 or micro_batch_id >= micro_batches:
                    print(f"|  *  ", end="")
                else:
                    print(f"| {micro_batch_id:2}f ", end="")
            else:
                micro_batch_id = total_steps - (step_id + stage_id + 1)
                if micro_batch_id < 0 or micro_batch_id >= micro_batches:
                    print(f"|  *  ", end="")
                else:
                    print(f"| {micro_batch_id:2}b ", end="")
        print("|")
    total_compute = total_steps * stages
    valid_compute = 2 * micro_batches * stages
    print(f"total compute: {total_compute}")
    print(f"valid compute: {valid_compute}")
    print(f"bubble: {total_compute - valid_compute}")
    print(f"bubble ratio: {(total_compute - valid_compute) / total_compute:.2%}")

def graphic_continuous_pipe(stages: int, micro_batches: int):
    total_steps = 2 * (micro_batches + stages - 1)
    for step_id in range(total_steps):
        print(f"| {step_id:3} ", end="")
    print("|")
    for stage_id in range(stages):
        for step_id in range(total_steps):
            micro_batch_id, is_forward = _step_to_micro_batch(stages, stage_id, step_id)
            if is_forward and 0 <= micro_batch_id:
                print(f"| {micro_batch_id:2}F ", end="")
            elif 0 <= micro_batch_id:
                print(f"| {micro_batch_id:2}B ", end="")
            else:
                print(f"|  *  ", end="")
            
            # for micro_batch_id in range(num_micro_batches):
            #     if micro_batch_id == 0:
            #         print(f'| {micro_batch_id} ', end="")
            #     else:
            #         print(f"| * ", end="")
        print("|")
    
    total_compute = total_steps * stages
    valid_compute = (total_steps - 2 * (stages-1))* stages
    print(f"total compute: {total_compute}")
    print(f"valid compute: {valid_compute}")
    print(f"bubble: {total_compute - valid_compute}")
    print(f"bubble ratio: {(total_compute - valid_compute) / total_compute:.2%}")


if __name__ == "__main__":
    stages = 4
    micro_batches = 4
    graphic_gpipe(stages, micro_batches)
    graphic_pipe_dream(stages, micro_batches)
    
    graphic_continuous_pipe(stages, micro_batches)
    # print("")
    # print("")

    # graphic_pipe_dream(4, 3)
    # print("")
    # print("")

    # graphic_pipe_dream(4, 4)
    # print("")
    # print("")