import numpy as np
import toml
import os


def export_quant_table(quantizers: dict, quant_dir: str, format: str = 'toml'):

    table = {}

    def save_tensor(name: str, tensor):
        np.save(os.path.join(quant_dir, name), tensor.numpy())
        return '{}.npy'.format(name)

    for key, value in quantizers.items():
        quantizer = value[0]

        dump = dict()

        sym = quantizer.sym
        if not sym:
            dump['zero'] = save_tensor(name=key + '.zero', tensor=value[2])
        dump['scale'] = save_tensor(name=key + '.scale', tensor=value[1])
        dump['wbits'] = value[4]
        dump['groupsize'] = value[5]
        if value[5] > 0:
            dump['group_ids'] = save_tensor(name=key + '.group_ids', tensor=value[3])

        dump['sym'] = sym
        dump['perchannel'] = quantizer.perchannel

        table[key] = dump

    if not os.path.exists(quant_dir):
        os.mkdir(quant_dir)

    with open(os.path.join(quant_dir, 'quant.toml'), 'w') as f:
        toml.dump(table, f)
