import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.registry import LOSSES
from torch.nn.modules.linear import Linear

@LOSSES.register_module
class NT_Xentloss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, label):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape 
        device = z1.device 
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

        negatives = similarity_matrix[~diag].view(2*N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        labels = torch.zeros(2*N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)


if __name__=='__main__':
    criterion = NT_Xentloss()
    net = Linear(256,512)
    output = [net(torch.randn(512,256)), net(torch.randn(512,256))]
    label = [torch.randn(512)]
    loss = criterion(*output, *label)
    print(loss)

    