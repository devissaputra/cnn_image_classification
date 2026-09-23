from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt, torch, torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay
torch.manual_seed(42); torch.set_num_threads(1); X,y=load_digits(return_X_y=True); X=(X.astype('float32')/16.).reshape(-1,1,8,8); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); dl=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr,dtype=torch.long)),64,shuffle=True)
class CNN(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)); self.fc=nn.Linear(128,10)
 def forward(self,x): return self.fc(self.net(x).flatten(1))
m=CNN(); opt=torch.optim.Adam(m.parameters(),lr=.003); lossf=nn.CrossEntropyLoss(); losses=[]
for ep in range(14):
 ls=[]
 for xb,yb in dl: opt.zero_grad(); loss=lossf(m(xb),yb); loss.backward(); opt.step(); ls.append(loss.item())
 losses.append(float(np.mean(ls)))
with torch.no_grad(): pred=m(torch.tensor(Xte)).argmax(1).numpy()
out={'accuracy':float(accuracy_score(yte,pred)),'macro_f1':float(f1_score(yte,pred,average='macro')),'epochs':len(losses)}; Path('results').mkdir(exist_ok=True); Path('results/metrics.json').write_text(json.dumps(out,indent=2))
plt.figure(figsize=(7,5)); plt.plot(range(1,len(losses)+1),losses,marker='o'); plt.xlabel('Epoch'); plt.ylabel('Training loss'); plt.title('CNN learning curve'); plt.tight_layout(); plt.savefig('assets/03_data_or_model.png',dpi=150); plt.close()
fig,ax=plt.subplots(figsize=(7,6)); ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax,colorbar=False); ax.set_title('CNN held-out confusion matrix'); fig.tight_layout(); fig.savefig('assets/04_evaluation_or_results.png',dpi=150); plt.close(fig); print(json.dumps(out,indent=2))