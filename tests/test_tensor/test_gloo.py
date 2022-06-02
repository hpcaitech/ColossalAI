from colossalai.communication.collective import all_gather
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = torch.nn.Embedding(20, 4)
        self.proj = torch.nn.Linear(4, 8)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.proj(x)
        return x

def test_gloo():
    model = Net()
    x = torch.randint(2, (20,))
    out = model(x)

if __name__ == '__main__':
    test_gloo()