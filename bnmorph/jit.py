from torch.utils.cpp_extension import load
add_one_machine = load(
    'add_one_machine', ['add_one_machine.cpp', 'add_one_machine_kernel.cu'], verbose=True)
help(add_one_machine)
