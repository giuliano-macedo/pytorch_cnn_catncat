import torch.nn as nn
NO_CV=8
class V2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cvs=nn.ModuleList()
        for _ in range(NO_CV):
            cv=nn.Conv2d(
                in_channels=1, 
                out_channels=1,
                kernel_size=5, 
                stride=1, 
                padding=0
            )
            activation=nn.LeakyReLU()
            pool=self.pool1 = nn.MaxPool2d(kernel_size=1)
            self.cvs.extend((cv,activation,pool))


        self.fc = nn.Linear(1 * 32 * 32, 2)
    
    def forward(self, x):
        out = x
        for f in self.cvs:
            out=f(out)
        # print(out.size());exit()
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out