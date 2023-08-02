### FREQ_PINN ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os
import copy
import pandas as pd
from torchsummary import summary
import re
matplotlib.use('Agg')
from IPython.display import update_display, display

"""
Solution:
u(x,y) = Sin((x-2)**2+y**2) + Sin(x**2+(y-3)**2) 
for x in [-2,2] and y in [-2,2]

with BC
u(x,2)
u(x,-2)
u(2,y)
u(-2,y)

"""


###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################


### Model ###

class Augmented_Proba_i(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_freq, nb_hidden_layers, init_factor):
        super().__init__()
        self.d_freq = d_freq
        self.nb_hidden_layers = nb_hidden_layers
        self.enc = nn.Linear(input_dim, d_model)
        self.layers = nn.Sequential(nn.ModuleList(nn.Linear(d_model,d_model, bias=True) for i in range(nb_hidden_layers-1)))
        self.dec = nn.Linear(d_model, output_dim)
        self.init_factor = init_factor

    def forward(self, x):                                       # x of the form [N, L, d_model]
        y = torch.sin(self.enc(x))
        for i in range(self.nb_hidden_layers-1):
            y = torch.sin(self.layers[0][i](y))
        y = self.init_factor*F.softmax(self.dec(y), dim=-1)     # [N, L, output_dim]
        return y                                                # [N, L, output_dim=d_freq*d_model]


class NN_i(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_freq, nb_hidden_layers):
        super().__init__()

        self.nb_hidden_layers = nb_hidden_layers
        self.enc = nn.Linear(input_dim, d_model, bias=False)
        self.layers = nn.Sequential(nn.ModuleList(nn.Linear(d_model*d_freq,d_model*d_freq, bias=True) for i in range(nb_hidden_layers-1))) 
        self.dec = nn.Linear(d_model*d_freq, output_dim)
        self.weights_freq = torch.zeros((1,1,d_model,d_freq), dtype=torch.float32)     # [1, 1, 1, d_freq]
        self.weights_freq = nn.Parameter(self.weights_freq)
        self.bias_freq = torch.zeros((1,1,d_model,d_freq), dtype=torch.float32)       # [1, 1, 1, d_freq]
        self.bias_freq = nn.Parameter(self.bias_freq)
        self.d_model = d_model

    def forward(self, x):                                       # x of the form [N, L, input_dim]
        y = self.enc(x)
        y = y.unsqueeze(-1)
        y = y*(self.weights_freq) + torch.pi*self.bias_freq     # [N, L, d_model, d_freq]
        y = y.permute((0,1,3,2))                                # [N, L, d_freq, d_model]
        y = y.reshape((y.shape[0], y.shape[1], -1))             # [N, L, d_freq*d_model]
        return y, self.weights_freq.expand((y.shape[0], y.shape[1], self.d_model, -1)).permute((0,1,3,2)).reshape((y.shape[0], y.shape[1], -1))     # [N, L, d_model*d_freq]


class Proba_Model(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, d_freq, nb_hidden_layers, nb_NN, lower_bound=0, upper_bound=5):
        super().__init__()

        self.nb_NN = nb_NN
        self.NN_i_list = nn.ModuleList(NN_i(input_dim=input_dim, output_dim=output_dim, d_model=d_model, d_freq=d_freq, nb_hidden_layers=nb_hidden_layers) for i in range(nb_NN))
        self.Proba_i_list = nn.ModuleList(Augmented_Proba_i(input_dim=input_dim, output_dim=d_freq*d_model, d_model=d_model, d_freq=d_freq, nb_hidden_layers=nb_hidden_layers, init_factor=5) for i in range(nb_NN))        
        self.Bias_i_list = nn.ModuleList(Augmented_Proba_i(input_dim=input_dim, output_dim=d_freq*d_model, d_model=d_model, d_freq=d_freq, nb_hidden_layers=nb_hidden_layers, init_factor=torch.pi/2) for i in range(output_dim))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.d_freq = d_freq
        self.d_model = d_model

    def forward(self, x):                                                                                       # x of the form [N, L, input_dim]
        NN_i_tensor = torch.cat([network(x)[0].unsqueeze(-1) for network in self.NN_i_list], dim=-1)            # [N, L, output_dim, nb_NN]
        #w_i_tensor = torch.cat([network(x)[1].unsqueeze(-1) for network in self.NN_i_list], dim=-1)             # [N, L, output_dim, nb_NN]
        Proba_i_tensor =  torch.cat([network(x).unsqueeze(-1) for network in self.Proba_i_list], dim=-1)        # [N, L, output_dim, nb_NN]
        Bias_i_tensor =  torch.cat([network(x).unsqueeze(-1) for network in self.Bias_i_list], dim=-1)          # [L, d_freq_t*d_freq*d_model, nb_NN=output_dim]
        prod_NN_i_Proba_i_tensor = Proba_i_tensor*torch.sin(NN_i_tensor+Bias_i_tensor)                          # [N, L, output_dim, nb_NN]
        sol = torch.sum(torch.sum(prod_NN_i_Proba_i_tensor, dim=-1), dim=-1).unsqueeze(-1)                      # [N, L, output_dim]
        return sol, NN_i_tensor, Proba_i_tensor, prod_NN_i_Proba_i_tensor                                       # [N, L, output_dim], [N, L, output_dim, nb_NN], [N, L, output_dim, nb_NN], [N, L, output_dim, nb_NN]

    def init_param(self, init_factor):
        with torch.no_grad():
            for i in range(self.nb_NN):#
                theta = torch.linspace(0, 2*torch.pi, self.d_model)             # d_model
                self.NN_i_list[i].enc.weight.data[:,0] = torch.cos(theta)       # d_model
                self.NN_i_list[i].enc.weight.data[:,1] = torch.sin(theta)       # d_model
                #self.NN_i_list[i].weights_freq[:,:,:,:] = torch.linspace(init_factor-1, init_factor+1, self.d_freq, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                self.NN_i_list[i].weights_freq[:,:,:,:] = torch.linspace(2, init_factor, self.d_freq, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)


        

###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

### Train Model ###

# bc loss
class Sol_Eq(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x_eq):
        x = x_eq[:,:,0].unsqueeze(-1)
        y = x_eq[:,:,1].unsqueeze(-1)
        return torch.sin((x-2)**2+y**2) + torch.sin(x**2+(y-3)**2)  # [N, L, 1]

# func on RHS of Poisson equation for a 2D input: x_eq [N, L, 2]: 
class Func(nn.Module):
    def __init__(self,sign):
        super().__init__()
        self.sign = sign
    def forward(self,x_eq):
        x = x_eq[:,:,0].unsqueeze(-1)
        y = x_eq[:,:,1].unsqueeze(-1)
        return (2*(x-2)+self.sign*2*y)*torch.cos((x-2)**2+y**2) + (2*(x)+self.sign*2*(y-3))*torch.cos((x-2)**2+(y-3)**2)  # [N, L, 1]

class Func_2_x(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x_eq):
        x = x_eq[:,:,0].unsqueeze(-1)
        y = x_eq[:,:,1].unsqueeze(-1)
        return 2*torch.cos((x-2)**2+y**2) - 4*(x-2)**2*torch.sin((x-2)**2+y**2) + 2*torch.cos((x)**2+(y-3)**2) - 4*(x)**2*torch.sin((x)**2+(y-3)**2)  # [N, L, 1]

class Func_2_y(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x_eq):
        x = x_eq[:,:,0].unsqueeze(-1)
        y = x_eq[:,:,1].unsqueeze(-1)
        return 2*torch.cos((x-2)**2+y**2) - 4*y**2*torch.sin((x-2)**2+y**2) + 2*torch.cos((x)**2+(y-3)**2) - 4*(y-3)**2*torch.sin((x)**2+(y-3)**2) # [N, L, 1]
    
# residual loss with sin(x**2)
def res_loss_x(criterion, u_eq, x_eq, Sol_Eq):
    u_x = torch.autograd.grad(outputs=u_eq, inputs=x_eq, grad_outputs=torch.ones_like(u_eq), retain_graph=True, create_graph=True)[0] # [N, L, 1]
    return criterion(u_x, 2*x_eq*torch.cos(x_eq**2)), criterion(u_eq[-1], Sol_Eq(x_eq[-1])) # scalars: loss_eq, loss_bc


# bc loss
def bd_loss(criterion, u_context, u_tgt):   # u_context, u_tgt as [N, S, 1]
    return criterion(u_context,u_tgt)       # scalar

# residual loss with Poisson equation
def res_loss(criterion, u_eq, x_eq, x_eq_x, x_eq_y, func_pos, func_neg, func_2_x, func_2_y):
    u_x_y = torch.autograd.grad(outputs=u_eq, inputs=x_eq, grad_outputs=torch.ones_like(u_eq), retain_graph=True, create_graph=True)[0] # [N, L, 2]
    u_x = u_x_y[:,:,0].unsqueeze(-1) # [N, L, 1]
    u_y = u_x_y[:,:,1].unsqueeze(-1) # [N, L, 1]
    u_xx = torch.autograd.grad(outputs=u_x, inputs=x_eq_x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]  # [N, L, 1]
    u_yy = torch.autograd.grad(outputs=u_y, inputs=x_eq_y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]  # [N, L, 1]
    return  criterion(u_x + u_y, func_pos(x_eq)) + criterion(u_x - u_y, func_neg(x_eq)) + criterion(u_xx, func_2_x(x_eq)) + criterion(u_yy, func_2_y(x_eq)) # scalar 

# train model
class Train_Model(nn.Module):
    def __init__(self, supervized, model, criterion, optimizer, eps, num_epoch, folder_sol, folder_NN_i, folder_Proba_i, folder_prod, folder_loss, Sol_Eq, domain_u_bound, plot_every, nb_NN, mini_batch_size):
        super().__init__()

        self.supervized = supervized
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.eps = eps
        self.num_epoch = num_epoch
        self.Sol_Eq = Sol_Eq
        self.domain_u_bound = domain_u_bound
        self.folder_sol = folder_sol
        self.folder_NN_i = folder_NN_i
        self.folder_Proba_i = folder_Proba_i
        self.folder_prod_NN_i_Proba_i = folder_prod
        self.folder_loss = folder_loss
        self.plot_every = plot_every
        self.nb_NN = nb_NN
        self.mini_batch_size = mini_batch_size

    def forward(self, x_eq, x_eq_x, x_eq_y, x_bc, u_tgt_bc, u_tgt_eq, Loss_bc = 100, Loss_eq=100, Loss_tot=100):

        epoch = 1
        display('',display_id='loss')
        N = x_eq.shape[0]

        L_tot = []
        L_eq = []
        L_bc = []
        list_epoch = []

        # split the input tesnors into minibatches of size mini_batch_size (or of lower size for the last tensor)
        x_bc_split = torch.split(x_bc,self.mini_batch_size,dim=0)       # tuple of tensors
        x_eq_x = torch.split(x_eq_x,self.mini_batch_size,dim=0)         # tuple of tensors
        x_eq_y = torch.split(x_eq_y,self.mini_batch_size,dim=0)         # tuple of tensors
        u_tgt_bc = torch.split(u_tgt_bc, self.mini_batch_size, dim=0)   # tuple of tensors
        u_tgt_eq = torch.split(u_tgt_eq, self.mini_batch_size, dim=0)   # tuple of tensors

        while epoch <= self.num_epoch :
           
            for b in range(len(x_bc)):
                
                with torch.no_grad():
                    x_bc_batch = x_bc_split[b]
                    x_eq_x_batch = x_eq_x[b]
                    x_eq_y_batch = x_eq_y[b]
                    u_tgt_bc_batch = u_tgt_bc[b]
                    u_tgt_eq_batch = u_tgt_eq[b]
                
                # print(x_eq_x_batch.shape) 

                x_bc_batch.requires_grad = True
                x_eq_x_batch.requires_grad = True
                x_eq_y_batch.requires_grad = True

                x_eq_batch = torch.cat((x_eq_x_batch, x_eq_y_batch), dim=-1) #[N,L,2]

                # initialize all param.grad to zero
                self.optimizer.zero_grad()

                # compute the forward pass
                u_bc, _, _, _ = self.model(x_bc_batch)   # [N, S, 1]
                u_eq, _, _, _ = self.model(x_eq_batch)   # [N, L, 1]

      
                # compute boundary loss
                Loss_bc = bd_loss(self.criterion, u_bc, u_tgt_bc_batch)

                # compute loss for collocation points
                if self.supervized == True:
                    Loss_eq = bd_loss(self.criterion, u_eq, u_tgt_eq_batch)
                else:
                    Loss_eq = res_loss(criterion=self.criterion, u_eq=u_eq, x_eq=x_eq_batch, x_eq_x=x_eq_x_batch, x_eq_y=x_eq_y_batch, func_pos=Func(sign=1.), func_neg=Func(sign=-1.), func_2_x=Func_2_x(), func_2_y=Func_2_y())
                
                Loss_tot = Loss_bc + Loss_eq 
                Loss_tot.backward()

                # update all the weights of the model using self.optimizer
                self.optimizer.step()


            L_bc.append(Loss_bc.item())
            L_eq.append(Loss_eq.item())
            L_tot.append(Loss_tot.item())
            list_epoch.append(epoch)
                    
            # print losses
            with torch.no_grad():

                if (epoch) % 1 == 0:
                    update_display("it. " + str(epoch) + " Loss_bc " + str(Loss_bc.item()) + " Loss_eq " + str(Loss_eq.item()) + " Loss_tot " + str(Loss_tot.item()), display_id='loss')
                    
                if (epoch) % self.plot_every == 0 or epoch == 1:

                    print('plotting intermediate results...')
                    
                    # transform the data points for plots
                    X_train, Y_train, tgt_train = train_results(x_eq=x_eq, x_bc=x_bc, Sol_Eq=self.Sol_Eq)

                    # compute the forward pass for plots
                    u_bc, NN_i_tensor_bc, Proba_i_tensor_bc, prod_NN_i_Proba_i_tensor_bc = self.model(x_bc)   # [N, S, output_dim], [N, S, output_dim, nb_Proba_i], [N, S, output_dim, nb_Proba_i], [N, S, output_dim, nb_Proba_i]
                    u_eq, NN_i_tensor_eq, Proba_i_tensor_eq, prod_NN_i_Proba_i_tensor_eq = self.model(x_eq)   # [N, L, output_dim], [N, L, output_dim, nb_Proba_i], [N, L, output_dim, nb_Proba_i], [N, L, output_dim, nb_Proba_i]

                    # plot ground truth, solution and abs. error on the same subplot
                    plot_train_results(u_eq=u_eq, u_bc=u_bc, X_train=X_train, Y_train=Y_train, tgt_train=tgt_train, epoch=epoch, domain_u_bound=self.domain_u_bound, dir_path=self.folder_sol +'imgs')

            epoch = epoch + 1
            
        print('generating_and_saving_training_results...')
        plot_loss(list_epoch=list_epoch, L_bc=L_bc, L_eq=L_eq, L_tot=L_tot, dir_path=self.folder_loss +'loss.png')
        save_loss_csv(list_epoch=list_epoch, L_bc=L_bc, L_eq=L_eq, L_tot=L_tot, dir_path=self.folder_loss+'loss.csv')
        save_gif(dir_img=self.folder_sol)

        return epoch, Loss_tot


# at end of simulation:
# print test solution (subplot 1x2) :    1) u_eq vs u_sol   2) abs(u_eq-u_sol)

def train_results(x_eq, x_bc, Sol_Eq):

    tgt_train = Sol_Eq(x_eq).reshape(-1).tolist()
    tgt_bc_train = Sol_Eq(x_bc).reshape(-1).tolist()
    tgt_train.extend(tgt_bc_train)

    X_train = x_eq.reshape(-1,2)[:,0].tolist()              # [1000]
    Y_train = x_eq.reshape(-1,2)[:,1].tolist()              # [1000]
    
    X_bc_train = x_bc.reshape(-1,2)[:,0].tolist()           # [1000]
    Y_bc_train = x_bc.reshape(-1,2)[:,1].tolist()           # [1000]
    
    X_train.extend(X_bc_train)                              # [2000]
    Y_train.extend(Y_bc_train)                              # [2000]

    return X_train, Y_train, tgt_train


def plot_train_results(u_eq, u_bc, X_train, Y_train, tgt_train, epoch, domain_u_bound, dir_path):

    Z_train = u_eq.reshape(-1).tolist()         # [1000]
    Z_bc_train = u_bc.reshape(-1).tolist()      # [1000]
    Z_train.extend(Z_bc_train)                  # [2000]

    fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(20,5))

    im_train = ax[0].tricontourf(X_train, Y_train, tgt_train)
    plt.colorbar(im_train, ax=ax[0])

    im = ax[1].tricontourf(X_train, Y_train, Z_train)
    plt.colorbar(im, ax=ax[1])

    im_tgt = ax[2].tricontourf(X_train, Y_train, torch.abs(torch.tensor(Z_train)-torch.tensor(tgt_train)).tolist(), cmap='binary')
    plt.colorbar(im_tgt, ax=ax[2])

    ax[0].set_title("U_target")
    ax[0].set_xlabel("[x]") 
    ax[0].set_ylabel("[y]")
    ax[0].set_xlim(-domain_u_bound-1,domain_u_bound+1)
    ax[0].set_ylim(-domain_u_bound-1,domain_u_bound+1)

    ax[1].set_title("U_estimated")
    ax[1].set_xlabel("[x]") 
    ax[1].set_ylabel("[y]")

    ax[2].set_title("Abs. Error")
    ax[2].set_xlabel("[x]") 
    ax[2].set_ylabel("[y]")
    plt.savefig(dir_path + '_' + str(epoch) + '.png')
    plt.close()



def plot_NN_i_results(NN_eq, NN_bc, X_train, Y_train, epoch, nb_NN, domain_u_bound, dir_path):
    
    fig, ax = plt.subplots(1, nb_NN, sharex='all', sharey='all', figsize=(6*nb_NN,5))

    Z_train = NN_eq[:,:,:,0].reshape(-1).tolist()
    Z_bc_train = NN_bc[:,:,:,0].reshape(-1).tolist()
    Z_train.extend(Z_bc_train)

    im = ax.tricontourf(X_train, Y_train, Z_train)
    plt.colorbar(im, ax=ax)
    ax.set_title("indiv_NN_" + str(1))
    ax.set_xlabel("[x]") 
    ax.set_ylabel("[y]")
    ax.set_xlim(-domain_u_bound,domain_u_bound)
    ax.set_ylim(-domain_u_bound,domain_u_bound)

    plt.savefig(dir_path + '_' + str(epoch) + '.png')
    plt.close()


def plot_Proba_i_results(Proba_eq, Proba_bc, X_train, Y_train, epoch, nb_NN, domain_u_bound, dir_path):
    
    fig, ax = plt.subplots(1, nb_NN, sharex='all', sharey='all', figsize=(6*nb_NN,5))
    levels = np.linspace(0,1,51)
    for i in range(Proba_eq.shape[-1]):

        Z_train = Proba_eq[:,:,:,i].reshape(-1).tolist()
        Z_bc_train = Proba_bc[:,:,:,i].reshape(-1).tolist()
        Z_train.extend(Z_bc_train)

        im = ax[i].tricontourf(X_train, Y_train, Z_train, levels=levels, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title("indiv_Proba_" + str(i))
        ax[i].set_xlabel("[x]") 
        ax[i].set_ylabel("[y]")
        ax[i].set_xlim(-domain_u_bound,domain_u_bound)
        ax[i].set_ylim(-domain_u_bound,domain_u_bound)
    


    plt.savefig(dir_path + '_' + str(epoch) + '.png')
    plt.close()



def plot_prod_NN_i_Proba_i_results(prod_eq, prod_bc, u_eq, u_bc, X_train, Y_train, epoch, nb_NN, domain_u_bound, dir_path):
    
    fig, ax = plt.subplots(1, nb_NN+1, sharex='all', sharey='all', figsize=(7*nb_NN,5))

    for i in range(prod_eq.shape[-1]):

        Z_train = prod_eq[:,:,:,i].reshape(-1).tolist()
        Z_bc_train = prod_bc[:,:,:,i].reshape(-1).tolist()
        Z_train.extend(Z_bc_train)

        im = ax[i].tricontourf(X_train, Y_train, Z_train)
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title("prod_NN_" + str(i) + "_NN_" + str(i))
        ax[i].set_xlabel("[x]") 
        ax[i].set_ylabel("[y]")
        ax[i].set_xlim(-domain_u_bound,domain_u_bound)
        ax[i].set_ylim(-domain_u_bound,domain_u_bound)
    
    Z_train = u_eq.reshape(-1).tolist()         # [1000]
    Z_bc_train = u_bc.reshape(-1).tolist()      # [1000]
    Z_train.extend(Z_bc_train)                  # [2000]

    im = ax[nb_NN+1-1].tricontourf(X_train, Y_train, Z_train)
    plt.colorbar(im, ax=ax[nb_NN])

    for i in range(nb_NN):
        ax[nb_NN].set_title("U_estimated")
        ax[nb_NN].set_xlabel("[x]") 
        ax[nb_NN].set_ylabel("[y]")
        ax[nb_NN].set_xlim(-domain_u_bound,domain_u_bound)
        ax[nb_NN].set_ylim(-domain_u_bound,domain_u_bound)

    plt.savefig(dir_path + '_' + str(epoch) + '.png')
    plt.close()


def plot_loss(L_bc, L_eq, L_tot, dir_path, list_epoch):
    fig, ax = plt.subplots(1,1,figsize=(10,5))

    ax.plot(list_epoch, L_bc, 'g', label='loss_bc')
    ax.plot(list_epoch, L_eq, 'b', label='loss_eq')
    ax.plot(list_epoch, L_tot, 'k', label='loss_tot')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Loss")
    ax.set_xlabel("[epoch]") 
    ax.legend(loc='lower left')
    ax.grid(visible=True, which='major', color='grey', linestyle='-')
    ax.grid(visible=True, which='minor', color='lightgrey', linestyle='--')

    plt.savefig(dir_path)
    print('saved_loss.png')
    plt.close()


def save_loss_csv(list_epoch, L_bc, L_eq, L_tot, dir_path):
    data = {'list_epoch': list_epoch,'L_bc' : L_bc, 'L_eq' : L_eq, 'L_tot': L_tot}
    df = pd.DataFrame(data=data)
    df.to_csv(dir_path)
    print('saved_loss.csv')


def save_gif(dir_img):
    directory = os.fsencode(dir_img)
    with imageio.get_writer(dir_img+ '/results.gif', mode='i', fps=3) as writer:
        for file in sorted(os.listdir(directory), key=len):
            filename = os.fsdecode(file)
            if filename.endswith('.png'):
                image = imageio.imread(dir_img+'/'+filename)
                writer.append_data(image)
    print('saved_gif_' + str(dir_img))


def create_directory(dir):
    # If folder doesn't exist, then create it.
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print("created folder : " + dir)
    else:
        print("folder : " + dir + ' already exists' )


def create_directories(folder_dir, folder_name):
    folder_dir = folder_dir + folder_name + '/'
    folder_sol = folder_dir + 'sol'+ '/'
    folder_NN_i = folder_dir + 'NN_i' + '/'
    folder_Proba_i = folder_dir + 'Proba_i' + '/'
    folder_prod_NN_i_Proba_i = folder_dir + 'prod_NN_i_Proba_i' + '/'
    folder_loss = folder_dir + 'loss' + '/'
    folder_model = folder_dir + 'model' + '/'

    create_directory(folder_dir)
    #create_directory(folder_NN_i)
    #create_directory(folder_Proba_i)
    #create_directory(folder_prod_NN_i_Proba_i)
    create_directory(folder_sol)
    create_directory(folder_loss)
    create_directory(folder_model)

    return folder_sol, folder_NN_i, folder_Proba_i, folder_prod_NN_i_Proba_i, folder_loss, folder_model


def load_csv_to_list(dir_path, name_file):
    df = pd.read_csv(dir_path + name_file, index_col=0)
    list_epoch = df['list_epoch'].values.tolist()
    L_bc = df['L_bc'].values.tolist()
    L_eq = df['L_eq'].values.tolist()
    L_tot = df['L_tot'].values.tolist()
    return list_epoch, L_bc, L_eq, L_tot


# Preparing 2D domain samples
def data_prep_structured_grid(lower_bound, upper_bound, nb_pts, N, S, L, Sol_Eq):
    # collocation points
    x = torch.linspace(lower_bound, upper_bound, nb_pts) # [nb_pts]
    y = torch.linspace(lower_bound, upper_bound, nb_pts) # [nb_pts]

    # bc points
    x_bc_left_edge = torch.stack(torch.meshgrid(x[0],y)).reshape(2,-1).T    # [nb_pts, 2]
    x_bc_right_edge = torch.stack(torch.meshgrid(x[-1],y)).reshape(2,-1).T  # [nb_pts, 2]
    x_bc_bottomn_edge = torch.stack(torch.meshgrid(x,y[0])).reshape(2,-1).T # [nb_pts, 2]
    x_bc_top_edge = torch.stack(torch.meshgrid(x,y[-1])).reshape(2,-1).T    # [nb_pts, 2]
    x_bc = torch.cat((x_bc_left_edge, x_bc_right_edge, x_bc_bottomn_edge, x_bc_top_edge)) # [4*nb_pts, 2]

    # randomly choose S bc_pts and tgt_bc out of x_bc using idx without replacement for each batch N
    # creating N batches
    X_BC = torch.empty((N,S+4,2))     # [N,S,2]
    for n in range(N):
        idx = np.random.choice(a=x_bc.shape[0], size=S, replace=False) # N*[S,2]
        X_BC[n] = torch.cat((x_bc[idx,:], torch.tensor([[upper_bound,upper_bound], [upper_bound,lower_bound], [lower_bound,upper_bound], [lower_bound,lower_bound]])), dim=0) 
    TGT_BC = Sol_Eq(X_BC)

    # Latin Hypercube Sampling for collocation points & tgt_eq values for each batch
    X_EQ = torch.empty((N,L,2))     # [N,L,2]
    for n in range(N):
        X_EQ[n] = torch.tensor(qmc.scale(sample=qmc.LatinHypercube(d=2).random(L),l_bounds=[lower_bound,lower_bound], u_bounds=[upper_bound,upper_bound])) # N*[L,2]
    TGT_EQ = Sol_Eq(X_EQ)

    return X_EQ, TGT_EQ, X_BC, TGT_BC

# Preparing 2D test samples
def structured_test_grid(lower_bound, upper_bound, nb_pts):
    x = torch.linspace(lower_bound, upper_bound, nb_pts)            # [nb_pts]
    y = torch.linspace(lower_bound, upper_bound, nb_pts)            # [nb_pts]
    x_test_eq = torch.stack(torch.meshgrid(x,y)).reshape(2,-1).T    # [nb_pts*nb_pts, 2]

    # bc points
    x_bc_left_edge = torch.stack(torch.meshgrid(x[0],y)).reshape(2,-1).T            # [nb_pts, 2]
    x_bc_right_edge = torch.stack(torch.meshgrid(x[-1],y)).reshape(2,-1).T          # [nb_pts, 2]
    x_bc_bottomn_edge = torch.stack(torch.meshgrid(x[1:-1],y[0])).reshape(2,-1).T   # [nb_pts, 2]
    x_bc_top_edge = torch.stack(torch.meshgrid(x[1:-1],y[-1])).reshape(2,-1).T      # [nb_pts, 2]

    x_test_bc = torch.cat((x_bc_left_edge, x_bc_right_edge, x_bc_bottomn_edge, x_bc_top_edge)) # [4*nb_pts, 2]
    return x_test_eq.unsqueeze(0), x_test_bc.unsqueeze(0)

def key(value):
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))

def plot_loss_models(dir_path):
    plt.figure(figsize=(16,9))
    for folder_name in sorted(os.listdir(dir_path), key=key, reverse=True):
        if os.path.isdir(os.path.join(dir_path, folder_name)):
            dir_loss = dir_path + folder_name + '/' + 'loss' + '/'
            list_epoch, _, _, L_tot  = load_csv_to_list(dir_path=dir_loss, name_file='loss.csv')
            plt.plot(list_epoch, L_tot, label=folder_name)
            plt.xscale('log')
            plt.yscale('log')
            plt.title("Loss_Phys")
            plt.xlabel("[epoch]") 
            plt.legend(loc='lower left')
    plt.grid(visible=True, which='major', color='grey', linestyle='-')
    plt.grid(visible=True, which='minor', color='lightgrey', linestyle='--')
    plt.savefig(dir_path+ os.path.basename(dir_path[:-1])+ '_loss.png')
    print('saved_loss_plot')
    plt.close()


def requires_grad_data(x_eq, x_bc):

    x_eq_x = x_eq[:,:,0].unsqueeze(-1) #[N,L,1]
    x_eq_y = x_eq[:,:,1].unsqueeze(-1) #[N,L,1]

    x_eq = torch.cat((x_eq_x, x_eq_y), dim=-1) #[N,L,2]

    return x_eq, x_eq_x, x_eq_y, x_bc


class Simulation(nn.Module):
    def __init__(self, supervized=False, input_dim=2, output_dim=1, d_model=20, d_freq=20, bsz=1, mini_batch_size=1, nb_hidden_layers=1, num_colloc_sample=6000, num_bc_sample=200, num_test_sample=1000, domain_u_bound=5, tot_num_pts=5000, res_test_pts=50, num_epoch=1000, eps=1e-4, lr_proba=1e-3, lr=1e-3, plot_every=100, folder_dir = 'C:/Users/Q602596/Documents/Projects/PINN/Results/Standard_PINN/', folder_name = 'standard_test_11', nb_NN=10, init_factor=5):
        super().__init__()

        # Supervised/Unsupervized
        self.supervized = supervized

        # Model Parameters
        self.input_dim, self.output_dim, self.d_model, self.d_freq, self.nb_hidden_layers = input_dim, output_dim, d_model, d_freq, nb_hidden_layers # input dimension # ouput dimension # embedding dimension [E]
        self.nb_NN = nb_NN
        
        # Initialisation
        self.init_factor = init_factor

        # Equation Parameters
        self.Sol_Eq = Sol_Eq()

        # Sampling Parameters
        self.bsz, self.num_colloc_sample,  self.num_bc_sample, self.domain_u_bound, self.num_test_sample, self.tot_num_pts, self.res_test_pts = bsz, num_colloc_sample, num_bc_sample, domain_u_bound, num_test_sample, tot_num_pts, res_test_pts
        self.domain_l_bound = -self.domain_u_bound

        # Training Parameters
        self.num_epoch, self.eps, self.lr_proba, self.lr, self.mini_batch_size = num_epoch, eps, lr_proba, lr, mini_batch_size

        # Plot parameters
        self.plot_every = plot_every

        # Directories for saving
        self.folder_dir, self.folder_name = folder_dir, folder_name

         ### Create Directories
        print('creating_directories...')
        self.folder_sol, self.folder_NN_i, self.folder_Proba_i, self.folder_prod_NN_i_Proba_i, self.folder_loss, self.folder_model = create_directories(folder_dir=self.folder_dir, folder_name=self.folder_name)

        #### Instanciate Transformer Class & Training Modules
        print('instantiating_model...')
        self.model = Proba_Model(input_dim=self.input_dim, d_model=self.d_model, d_freq=self.d_freq, output_dim=self.output_dim, nb_hidden_layers=self.nb_hidden_layers, nb_NN=self.nb_NN)
        self.model.init_param(init_factor=self.init_factor)
        self.criterion = nn.MSELoss()
        self.group_enc = nn.ParameterList([])
        self.group_low_freq_amp = nn.ParameterList([])
        self.group_Proba = nn.ParameterList([])
        self.group_NN = nn.ParameterList([])
        for name, param in self.model.named_parameters():
            if name.startswith('Proba_i_list'):
                if name.endswith('dec'):
                    print(param.shape)
                    self.group_low_freq_amp.append(param[0:40,:])
                else:
                    self.group_Proba.append(param)
            elif name.startswith('enc'):
                self.group_enc.append(param)
            else: 
                self.group_NN.append(param)
        self.optimizer = torch.optim.Adam([
                {'params': self.group_NN, 'name' : 'group_NN'},
                {'params': self.group_enc, 'lr': self.lr_proba, 'name' : 'group_enc'},
                {'params': self.group_low_freq_amp, 'lr': 1e-2, 'name' : 'group_enc'},
                {'params': self.group_Proba, 'lr': self.lr_proba, 'name' : 'group_Proba'}], lr=lr, betas=(0.99, 0.99), eps=1e-8) #
        self.train_model = Train_Model(supervized=self.supervized, model=self.model, criterion=self.criterion, optimizer=self.optimizer, eps=self.eps, num_epoch=self.num_epoch, folder_sol=self.folder_sol, folder_NN_i=self.folder_NN_i, folder_Proba_i=self.folder_Proba_i, folder_prod=self.folder_prod_NN_i_Proba_i, folder_loss=self.folder_loss, Sol_Eq=self.Sol_Eq, domain_u_bound=self.domain_u_bound, plot_every=self.plot_every, nb_NN=self.nb_NN, mini_batch_size=self.mini_batch_size)
        
        ### Generate Data Samples : training pts and collocation pts with corresponding target values
        print('generating_training_data...')
        self.x_eq, self.u_tgt_eq, self.x_bc, self.u_tgt_bc = data_prep_structured_grid(lower_bound=self.domain_l_bound, upper_bound=self.domain_u_bound, nb_pts=self.tot_num_pts, N=self.bsz, S=self.num_bc_sample, L=self.num_colloc_sample, Sol_Eq=self.Sol_Eq)
        self.x_eq, self.x_eq_x, self.x_eq_y, self.x_bc = requires_grad_data(x_eq=self.x_eq, x_bc=self.x_bc)
        self.x_eq_test, self.x_bc_test = structured_test_grid(lower_bound=self.domain_l_bound, upper_bound=self.domain_u_bound, nb_pts=self.res_test_pts) # [1, nb_pts*nb_pts, 2], [1, 4*nb_pts, 2]



    def train(self):

        ### Train Model & Save Results
        print('training_model...')
        epoch, loss = self.train_model(x_eq=self.x_eq, x_eq_x=self.x_eq_x, x_eq_y=self.x_eq_y, x_bc=self.x_bc, u_tgt_bc=self.u_tgt_bc, u_tgt_eq=self.u_tgt_eq) # [N, S, 1], [N, L, 1]

        ### Save Model
        print('saving_model...')
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    }, self.folder_model + self.folder_name +'.pth')
        print('... simulation terminated!')
        print(self.model.parameters())


    def eval(self):
        
        print('evaluating_model...')

        # Load Model
        checkpoint = torch.load(self.folder_model + self.folder_name +'.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        with torch.no_grad():

            # Test Samples

            u_eq_test, _, _, _ = self.model(self.x_eq_test) # [1, L, 1]
            # u_bc_test = self.model(x_bc_test) # [1, S, 1]

            tgt_eq_test = self.Sol_Eq(self.x_eq_test) # [1, L, 1]
            # tgt_bc_test = sol_eq(x_bc_test) # [1, S, 1]

            X = torch.split(self.x_eq_test.squeeze(0).T,self.res_test_pts)[0][0].reshape(self.res_test_pts,self.res_test_pts)       # [10,10]
            Y = torch.split(self.x_eq_test.squeeze(0).T,self.res_test_pts)[0][1].reshape(self.res_test_pts,self.res_test_pts)       # [10,10]
            Z = torch.split(u_eq_test.squeeze(0).T,self.res_test_pts)[0][0].reshape(self.res_test_pts,self.res_test_pts)            # [10,10]
            Z_tgt = torch.split(tgt_eq_test.squeeze(0).T,self.res_test_pts)[0][0].reshape(self.res_test_pts,self.res_test_pts)      # [10,10]
            Z_abs_error = torch.abs(Z-Z_tgt)

            fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(20,5))
            im_tgt = ax[0].contourf(X.detach().numpy(), Y.detach().numpy(), Z_tgt.detach().numpy())
            plt.colorbar(im_tgt, ax=ax[0])
            im = ax[1].contourf(X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy())
            plt.colorbar(im, ax=ax[1])
            im_error = ax[2].contourf(X.detach().numpy(), Y.detach().numpy(), Z_abs_error.detach().numpy(), cmap='binary')
            plt.colorbar(im_error, ax=ax[2])

            ax[0].set_title("Ground Truth")
            ax[0].set_xlabel("[x]") 
            ax[0].set_ylabel("[y]")
            ax[0].set_xlim(-self.domain_u_bound-1,self.domain_u_bound+1)
            ax[0].set_ylim(-self.domain_u_bound-1,self.domain_u_bound+1)

            ax[1].set_title("U_test_estimated")
            ax[1].set_xlabel("[x]") 
            ax[1].set_ylabel("[y]")

            ax[2].set_title("Abs. Error")
            ax[2].set_xlabel("[x]") 
            ax[2].set_ylabel("[y]")
            plt.savefig(self.folder_sol + 'imgs' + '_fig_test.png')
            plt.close()

            print('saved_plots')

            print('... saved_results!')

            plt.close()




####################################################################################################################################################

folder_dir = 'folder_dir/'
param_name = 'folder_name'

list_param = [1]

for param in list_param:
    simu = Simulation(supervized=True, nb_NN=1, domain_u_bound=4, d_model=20, d_freq=20, nb_hidden_layers=1, num_epoch=100, init_factor=10, eps=1e-4, lr_proba=1e-1, lr=1e-1, plot_every=100, folder_dir=folder_dir+param_name+'/', folder_name=param_name+'_'+str(param), num_colloc_sample=6000, num_bc_sample=200)
    simu.train()
    simu.eval()
    #print(summary(simu.model, (1,2)))

plot_loss_models(dir_path=folder_dir+param_name+'/')


