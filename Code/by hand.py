import torch
from torch import nn,optim
from torch.utils.data import DataLoader,TensorDataset
from accelerate import Accelerator
from tqdm.auto import tqdm
class simpleModel(nn.Module):
  def __init__(self):
    super(simpleModel,self).__init__()
    self.fc = nn.Linear(20,1)
    
  def forward(self, x):
    return self.fc(x)

data = torch.randn(100,20)
labels = torch.randn(100,1)
dataset = TensorDataset(data,labels)
dataloader = DataLoader(dataset,batch_size=10)
num_epochs=10

model=simpleModel()
optimizer = optim.AdamW(model.parameters(),lr=0.01)
cr=nn.MSELoss()
accelerator=Accelerator()
model,optimizer,dataloader=accelerator.prepare(model,optimizer,dataloader) 
# 初始化进度条
total_steps=num_epochs*len(dataloader)
progress_bar = tqdm(range(total_steps))  


for epoch in range(num_epochs):
  model.train()
  for batch in dataloader:
    inputs,labels = batch
    optimizer.zero_grad()
    
    # batch = {k: v.to(device) for k, v in batch.items()}
    # outputs = model(**batch)  
    out=model(inputs)
    los=cr(out,labels)
    accelerator.backward(los)
    optimizer.step()
    # lr.step()
    # progress_bar.updata(1)

    

