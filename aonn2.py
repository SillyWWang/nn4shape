import torch
import numpy as np
import time

import matplotlib.pyplot as plt
from scipy.stats import qmc # for quasi random sampling
from pyDOE import lhs

import data as data
from icecream import ic
import pandas as pd

import bfgs

np.random.seed(1234)
torch.manual_seed(1234)
torch.set_default_dtype(torch.float64)          
                  

# Neural network
class Net(torch.nn.Module):
    def __init__(self, layers, device):
        super(Net, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        self.device = device

        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)
    
    def forward(self, x):
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            temp = torch.zeros(x.shape[0],self.layers[i+1]-self.layers[i]).to(self.device)
            x = h + torch.cat((x,temp),1)
        return self.fc[-1](x)

class InteriroSet():
    def __init__(self, Geometry2D, size, device, quasi=False):
        self.device = device

        lb = Geometry2D.bounds[:, 0]
        ub = Geometry2D.bounds[:, 1]
        if quasi==False:
            tmp_v_cen = lb + (ub - lb) * lhs(2, 3*size)
        else:
            sampler = qmc.Sobol(d=2)
            tmp_v_cen = lb + (ub - lb) * sampler.random(n=3*size)
    
        tmp_ind = data.filter_2d(tmp_v_cen, Geometry2D.edges)
        tmp_v_cen = tmp_v_cen[tmp_ind,:]

        self.x = tmp_v_cen[:size, :]
        self.size = self.x.shape[0]
        self.f = f(self.x)
        self.weight_grad = torch.ones(self.size, 1)

        self.x.requires_grad = True        
        self.x = self.x.to(self.device)
        self.weight_grad = self.weight_grad.to(self.device)
        self.f = self.f.to(self.device)
        

        
class BoundarySet():
    def __init__(self, Geometry2D, device):
        self.x = Geometry2D.vertices
        self.size = self.x.shape[0]
        self.device = device
        tau = torch.roll(self.x,1,0) - torch.roll(self.x,-1,0)
        n = torch.hstack([tau[:,1:2], -tau[:,0:1]])
        self.n = n/(n**2).sum(1,keepdims=True)**0.5
        self.f = f(self.x)
        self.weight_grad = torch.ones(self.size,1)

        self.x.requires_grad = True
        self.x = self.x.to(self.device)
        self.n = self.n.to(self.device)
        self.weight_grad = self.weight_grad.to(self.device)
        self.f = self.f.to(self.device)

def loss_func_y(NetY, InSet, BdSet):
    InSet.y = NetY(InSet.x)
    BdSet.y = NetY(BdSet.x)

    InSet.yx, = torch.autograd.grad(InSet.y, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.yx0 = InSet.yx[:,0:1]
    InSet.yx1 = InSet.yx[:,1:2]
    InSet.yx0x, = torch.autograd.grad(InSet.yx0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.yx0x0 = InSet.yx0x[:,0:1]
    InSet.yx1x, = torch.autograd.grad(InSet.yx1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.yx1x1 = InSet.yx1x[:,1:2]

    InSet.res_y = -(InSet.yx0x0+InSet.yx1x1) - InSet.f
    
    BdSet.res_y = BdSet.y

    # loss and J
    InSet.J = InSet.y.mean()*np.pi
    # for shape derivative
    BdSet.yx, = torch.autograd.grad(BdSet.y, BdSet.x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=BdSet.weight_grad)
    BdSet.yx_n = (BdSet.yx*BdSet.n).sum(1,keepdims=True).detach()
    
    InSet.loss_y = (InSet.res_y**2).mean()
    BdSet.loss_y = (BdSet.res_y**2).mean()
    return (InSet.loss_y + 100 * BdSet.loss_y)**0.5

def loss_func_p(NetP, InSet, BdSet):
    InSet.p = NetP(InSet.x)
    BdSet.p = NetP(BdSet.x)

    InSet.px, = torch.autograd.grad(InSet.p, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.px0 = InSet.px[:,0:1]
    InSet.px1 = InSet.px[:,1:2]
    InSet.px0x, = torch.autograd.grad(InSet.px0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.px0x0 = InSet.px0x[:,0:1]
    InSet.px1x, = torch.autograd.grad(InSet.px1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.px1x1 = InSet.px1x[:,1:2]
    
    InSet.res_p = -(InSet.px0x0+InSet.px1x1) - 1
    
    BdSet.res_p = BdSet.p
 
    # for shape derivative
    BdSet.px, = torch.autograd.grad(BdSet.p, BdSet.x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=BdSet.weight_grad)
    BdSet.px_n = (BdSet.px*BdSet.n).sum(1,keepdims=True).detach()
    
    InSet.loss_p = (InSet.res_p**2).mean()
    BdSet.loss_p = (BdSet.res_p**2).mean()
    return (InSet.loss_p + 100 * BdSet.loss_p)**0.5


# H1 ritz representation  
def loss_func_W(NetW, InSet, BdSet):
    InSet.W = NetW(InSet.x)
    BdSet.W = NetW(BdSet.x)
    InSet.w0 = InSet.W[:,0:1]
    InSet.w1 = InSet.W[:,1:2]
    BdSet.w0 = BdSet.W[:,0:1]
    BdSet.w1 = BdSet.W[:,1:2]
    
    #w0
    InSet.w0x, = torch.autograd.grad(InSet.w0, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.w0x0 = InSet.w0x[:,0:1]
    InSet.w0x1 = InSet.w0x[:,1:2]
    InSet.w0x0x, = torch.autograd.grad(InSet.w0x0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.w0x0x0 = InSet.w0x0x[:,0:1]
    InSet.w0x1x, = torch.autograd.grad(InSet.w0x1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.w0x1x1 = InSet.w0x1x[:,1:2]

    InSet.res_w0 = -(InSet.w0x0x0+InSet.w0x1x1) + InSet.w0

    BdSet.w0x, = torch.autograd.grad(BdSet.w0, BdSet.x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=BdSet.weight_grad)
    BdSet.w0x_n = (BdSet.w0x*BdSet.n).sum(1,keepdims=True)
    BdSet.res_w0 = BdSet.w0x_n + BdSet.yx_n*BdSet.px_n*BdSet.n[:,0:1]
    
    #w1
    InSet.w1x, = torch.autograd.grad(InSet.w1, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.w1x0 = InSet.w1x[:,0:1]
    InSet.w1x1 = InSet.w1x[:,1:2]
    InSet.w1x0x, = torch.autograd.grad(InSet.w1x0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.w1x0x0 = InSet.w1x0x[:,0:1]
    InSet.w1x1x, = torch.autograd.grad(InSet.w1x1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.w1x1x1 = InSet.w1x1x[:,1:2]

    InSet.res_w1 = -(InSet.w1x0x0+InSet.w1x1x1) + InSet.w1

    BdSet.w1x, = torch.autograd.grad(BdSet.w1, BdSet.x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=BdSet.weight_grad)
    BdSet.w1x = (BdSet.w1x*BdSet.n).sum(1,keepdims=True)
    BdSet.res_w1 = BdSet.w1x + BdSet.yx_n*BdSet.px_n*BdSet.n[:,1:2]
    
    InSet.loss_w = (InSet.res_w0**2).mean() + (InSet.res_w1**2).mean()
    BdSet.loss_w = (BdSet.res_w0**2).mean() + (BdSet.res_w1**2).mean()
    
    loss = InSet.loss_w + 100 * BdSet.loss_w
    return loss**0.5


# Train neural network
def train_y(NetY, InSet, BdSet, OptimY, epochs): 
    print('Train Neural Network', flush=True)
    # Record the optimal parameters
    loss = loss_func_y(NetY, InSet, BdSet).data
    print('epoch: %d, loss: %.3e, time: %.2f'
          %(0, loss, 0.00), flush=True)

    # Training cycle
    for it in range(int(epochs)):
        start_time = time.time()

        # Forward and backward propogation
        def closure():
            OptimY.zero_grad()
            loss = loss_func_y(NetY, InSet, BdSet)
            loss.backward()
            return loss
        OptimY.step(closure)
        loss = loss_func_y(NetY, InSet, BdSet).data
        # Print
        elapsed = time.time() - start_time
        
        print('epoch: %d, loss: %.3e, time: %.2f'
              %((it+1)*100, loss, elapsed), flush=True)

# Train neural network
def train_p(NetY, InSet, BdSet, OptimY, epochs): 
    print('Train Neural Network', flush=True)
    # Record the optimal parameters
    loss = loss_func_p(NetY, InSet, BdSet).data
    print('epoch: %d, loss: %.3e, time: %.2f'
          %(0, loss, 0.00), flush=True)

    # Training cycle
    for it in range(int(epochs)):
        start_time = time.time()

        # Forward and backward propogation
        def closure():
            OptimY.zero_grad()
            loss = loss_func_p(NetY, InSet, BdSet)
            loss.backward()
            return loss
        OptimY.step(closure)
        loss = loss_func_p(NetY, InSet, BdSet).data
        # Print
        elapsed = time.time() - start_time
        
        print('epoch: %d, loss: %.3e, time: %.2f'
              %((it+1)*100, loss, elapsed), flush=True)
        
# Train neural network
def train_W(NetY, InSet, BdSet, OptimY, epochs): 
    print('Train Neural Network', flush=True)
    # Record the optimal parameters
    loss = loss_func_W(NetY, InSet, BdSet).data
    print('epoch: %d, loss: %.3e, time: %.2f'
          %(0, loss, 0.00), flush=True)

    # Training cycle
    for it in range(int(epochs)):
        start_time = time.time()

        # Forward and backward propogation
        def closure():
            OptimY.zero_grad()
            loss = loss_func_W(NetY, InSet, BdSet)
            loss.backward()
            return loss
        OptimY.step(closure)
        loss = loss_func_W(NetY, InSet, BdSet).data
        # Print
        elapsed = time.time() - start_time
        
        print('epoch: %d, loss: %.3e, time: %.2f'
              %((it+1)*100, loss, elapsed), flush=True)


bd_cashocs=np.load('vertices_cashocs.npy')
def data_plot(inner, boundary,i):
    
    # plot samples
    fig, ax = plt.subplots()
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_aspect('equal', 'box')
    inner = inner.detach().numpy()
    boundary = boundary.detach().numpy()
    ax.scatter(inner[:,0],inner[:,1],s=1,label='inner data')
    ax.scatter(boundary[:,0],boundary[:,1],s=1,label='boundary data')
    plt.axis('off')
    
    ax.plot(bd_cashocs[:,0],bd_cashocs[:,1],'b--',label='reference')
    plt.legend(loc='upper left', prop={'size': 12})
    # plt.title('step: %d' %(int(i)))
    fig.tight_layout()
    plt.savefig('figure_{}.png'.format(i))
    
def f(x):
    x1 = x[:,0:1]; x2 = x[:,1:2]
    return 2.5*(x1+0.4-x2**2)**2 + x1**2 + x2**2 - 1


nx_in = 1000
nx_bd = 500
epochs = 5
c = 1.0
device = 'cuda:0'
quasi = True

net_y = Net([2,10,10,1], device).to(device)
net_p = Net([2,10,10,1], device).to(device)
net_W = Net([2,10,10,2], device).to(device)

optim_y = bfgs.BFGS(net_y.parameters(),
                          lr=1, max_iter=100,
                          tolerance_grad=1e-16, tolerance_change=1e-16,
                          line_search_fn='strong_wolfe')
optim_p = bfgs.BFGS(net_p.parameters(),
                          lr=1, max_iter=100,
                          tolerance_grad=1e-16, tolerance_change=1e-16,
                          line_search_fn='strong_wolfe')
optim_W = bfgs.BFGS(net_W.parameters(),
                          lr=1, max_iter=100,
                          tolerance_grad=1e-16, tolerance_change=1e-16,
                          line_search_fn='strong_wolfe')

t = np.linspace(0,2*np.pi,nx_bd+1)[:-1].reshape(-1,1)

# node = np.hstack((np.cos(t),0.5*np.sin(t)))

node = np.zeros((nx_bd, 2))

for i in range(125):
    node[i, 0] = 1
    node[i, 1] = 0 + i*(1/125)

for i in range(125):
    node[125+i, 0] = 1 - i*(1/125)
    node[125+i, 1] = 1

for i in range(125):
    node[250+i, 0] = 0
    node[250+i, 1] = 1 - i*(1/125)

for i in range(125):
    node[375+i, 0] = 0 + i*(1/125)
    node[375+i, 1] = 0


edge = np.linspace(1,nx_bd,nx_bd).reshape(-1,1) + np.array([0,1])
edge[-1,:] = [nx_bd,1]
geometry = data.Geometry2D(node, edge, torch.float64, csv_file=False)

df = pd.DataFrame(node, columns = ['x0','x1'])
df.to_csv(path_or_buf='./vertices/rectangle_0.csv',index=False)
df = pd.DataFrame(edge, columns = ['ind0','ind1'])
df.to_csv(path_or_buf='./vertices/rectangle_ind.csv',index=False)

in_set = InteriroSet(geometry, nx_in, device, quasi)
bd_set = BoundarySet(geometry, device)

data_plot(in_set.x.data.cpu(), bd_set.x.data.cpu(),0)


for i in range(50):
    print('step: %d' %(int(i+1)), flush=True)
    train_y(net_y, in_set, bd_set, optim_y, epochs)
    train_p(net_p, in_set, bd_set, optim_p, epochs)
    train_W(net_W, in_set, bd_set, optim_W, epochs)

    optim_y = bfgs.BFGS(net_y.parameters(),
                          lr=1, max_iter=100,
                          tolerance_grad=1e-16, tolerance_change=1e-16,
                          line_search_fn='strong_wolfe')
    optim_p = bfgs.BFGS(net_p.parameters(),
                            lr=1, max_iter=100,
                            tolerance_grad=1e-16, tolerance_change=1e-16,
                            line_search_fn='strong_wolfe')
    optim_W = bfgs.BFGS(net_W.parameters(),
                            lr=1, max_iter=100,
                            tolerance_grad=1e-16, tolerance_change=1e-16,
                            line_search_fn='strong_wolfe')
    
    #updata boundary and in_set, bd_set
    delta_bd = net_W(bd_set.x)
    node_new = (geometry.vertices.data + c * delta_bd.data.cpu()).numpy()

    
    geometry = data.Geometry2D(node_new, edge, torch.float64, csv_file=False)
    in_set = InteriroSet(geometry, nx_in, device, quasi)
    bd_set = BoundarySet(geometry, device)

    if i in [0, 9, 19, 29, 39, 49]:
        df = pd.DataFrame(node_new, columns = ['x0','x1'])
        df.to_csv(path_or_buf='./vertices/rectangle_{}.csv'.format(i+1),index=False)
        data_plot(in_set.x.data.cpu(), bd_set.x.data.cpu(),i+1)