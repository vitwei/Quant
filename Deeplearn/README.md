# modelbase
bestks0.025
'''
class MLPrg(nn.Module):
    def __init__(self):
        super(MLPrg,self).__init__()
        self.hidden1=nn.Linear(in_features=100,out_features=512,bias=True)
        self.drop=nn.Dropout(0.2)
        self.hidden2=nn.Linear(in_features=512,out_features=256)
        self.predict=nn.Linear(256,1)
        self.bn1=nn.BatchNorm1d(512)
        self.bn2=nn.BatchNorm1d(256)
    def forward(self,x):
        x=F.tanh(self.hidden1(x))
        x=self.drop(x)
        x=F.tanh(self.hidden2(x))
        x=self.drop(x)
        output=self.predict(x)
        return output[:,0]
if __name__ == '__main__':
    rg = torch.load('J:/quant_trade/modelbase/bestks0.025.pt').cpu()
#rg = torch.load('your_model.pt', map_location=torch.device('cpu'))

'''


