import torch
import random
import pytest
from colossalai.kernel import transpose_pad, transpose_depad

seq_lens =  torch.tensor([24,127,31,65,24,127,31,65], dtype=torch.int64).cuda()
batch_size = 8
max_padding_size = 128
head_size = 64
head_num = 12
hidden_size = head_num * head_size


def seq_init(x):
    # for i in range(batch_size):
    #     len = seq_lens[i]
    #     for j in range (max_padding_size):
    #         for k in range(hidden_size):
    #             if(j<len):
    #                 x[i,j,k] =  float(random.randint(0,9))

    for i in range(batch_size):
        len = seq_lens[i]
        for j in range (max_padding_size):
            for k in range(hidden_size):
                if j < len:
                    tmp = x[i:i+1, j:j+1, :]
                    tmp.copy_(torch.randn(1,1,hidden_size).float())
    return x      


def manual_depad(hidden_states):
        # torch.Size([4, 512, 2048])
        new_hidden_states = torch.zeros((1, batch_size*max_padding_size, hidden_size), dtype=torch.float).cuda()

        valid_num = 0
        for i in range(batch_size):
            tmp = new_hidden_states[:, valid_num : valid_num + seq_lens[i], :]
            tmp.copy_(hidden_states[i:i+1, 0:seq_lens[i], :])
            valid_num += seq_lens[i]
        new_hidden_states = new_hidden_states[:, 0:valid_num, :]
        return new_hidden_states

def reshape(x):
    new_x_shape = x.size()[:-1] + (head_num, head_size)
    x = x.view(*new_x_shape)
    return x

# cpp_extension 
def compare(ta, tb):
    tta = torch.flatten(ta)
    ttb = torch.flatten(tb)

    len = torch.numel(tta)
    for i in range(len):
        if((tta[i]-ttb[i]) > 0.001):
            print(i)
            print(tta[i])
            print(ttb[i])
            return False
    return True

def test_kernel():
    # original 
    hidden_states = torch.zeros(batch_size, max_padding_size, head_num*head_size).cuda().float()
    hidden_states = seq_init(hidden_states)
    input_pad = reshape(hidden_states)
    res_original_pad = input_pad.permute(0,2,1,3)

    # transpose_pad
    hidden_states_depad = manual_depad(hidden_states)
    input_depad = reshape(hidden_states_depad)
    res_transpose_pad = transpose_pad(input_depad, batch_size, max_padding_size, seq_lens, head_num, head_size)
    assert compare(res_transpose_pad, res_original_pad) == True, "transpose_pad fault."

    # transpose_depad
    sum_seq = torch.sum(seq_lens)
    res_transpose_depad = transpose_depad(res_original_pad, batch_size, sum_seq, max_padding_size, seq_lens, head_num, head_size)
    assert compare(input_depad, res_transpose_depad) == True, "transpose_depad fault."





if __name__ == '__main__':
    test_kernel()