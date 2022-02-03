#!/usr/bin/env python
# coding: utf-8

# In[20]:


from scipy import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set random seed
torch.manual_seed(0)
# Set data type to double
torch.set_default_dtype(torch.double)


# Domain Information
Mx = 4
My = 4

dx = 2**(-8)

Lx = 1.0
Ly = 1.0

Dx_overlap = 2**(-4)
Dy_overlap = 2**(-4)

x_sw_o = np.maximum(np.linspace(0,Lx-Lx/Mx,Mx)-Dx_overlap,0)
y_sw_o = np.maximum(np.linspace(0,Ly-Ly/My,My)-Dy_overlap,0)
x_ne_o = np.minimum(np.linspace(Lx/Mx,Lx,Mx)+Dx_overlap,Lx)
y_ne_o = np.minimum(np.linspace(Ly/My,Ly,My)+Dy_overlap,Ly)

Nx_patch_o = np.int_((x_ne_o - x_sw_o)/dx) + 1
Ny_patch_o = np.int_((y_ne_o - y_sw_o)/dx) + 1

# Hyperparamter in loss function
mu = 0.001

# Training datasets
Nt = 10000
Dx_buffer = 2**(-4)

for k in range(2,My):
    for j in range(2,Mx):


        # Load data & initial weights & test data
        data = io.loadmat("data_semilinear/data_Mx{:d}_My{:d}_({:d},{:d})_Ntrain{:d}_dxb{:.3e}.mat".format(Mx, My, j, k, Nt, Dx_buffer))
        data_init = io.loadmat("data_semilinear/init_Mx{:d}_My{:d}_({:d},{:d}).mat".format(Mx, My, j, k))

        train_inputs = torch.tensor(data['phi'],dtype = torch.double)
        train_labels = torch.tensor(data['phi_int'],dtype = torch.double)

        test_inputs = torch.tensor(data['phi'],dtype = torch.double)
        test_labels = torch.tensor(data['phi_int'],dtype = torch.double)

        N_neuron = int(data_init['N_neuron'])
        dim_inputs = train_inputs.size(1)
        dim_labels = train_labels.size(1)


        # Define the mask to avoid differentiating at the end points
        mask = torch.ones(1,dim_labels-1)
        
        mask[0, Nx_patch_o[j-1]-1] = 0.0
        mask[0, 2*Nx_patch_o[j-1]-1] = 0.0
        mask[0, 2*Nx_patch_o[j-1] + Ny_patch_o[k-1] - 1] = 0.0

        # Differentiate the outputs
        train_labels_d = (train_labels[:,1:] - train_labels[:,:-1]) / dx * mask
        test_labels_d = (test_labels[:,1:] - test_labels[:,:-1]) / dx * mask

        
        # Print data info
        print(f"Current patch: ({j}, {k})")
        print(f"phi Size: {train_inputs.shape}, Type:{train_inputs.dtype}")
        print(f"phi_int Size: {train_labels.shape}, Type:{train_labels.dtype}")
        print(f"Input Dimension: {dim_inputs}, Label Dimension: {dim_labels}")
        print(f"Number of Neurons: {N_neuron}")


        # Define network architecture (patch-dependent) and set initial weights/bias
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(dim_inputs, N_neuron)
                self.fc2 = nn.Linear(N_neuron, dim_labels)

                self.fc1.weight.data = torch.tensor(data_init['V_init'],dtype = torch.double).t()
                self.fc1.bias.data = torch.zeros(1,N_neuron)
                self.fc2.weight.data = torch.tensor(data_init['U_init'],dtype = torch.double)
                self.fc2.bias.data = torch.zeros(1,dim_labels)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        net = Net()


        # Loss function
        criterion = nn.MSELoss(reduction='mean')
        
        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr = 0.001)
        # optimizer = optim.SGD(net.parameters(), lr = 0.1)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

        # Print mu in loss function
        print(f'Weight in loss function: mu = {mu}')

        # Use DataSet class to handle batch size
        N_train = train_inputs.size(0)
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels, train_labels_d)
        batch_size = N_train // 20
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0,
                                                  pin_memory=False)

        print(f'N_train = {N_train}, Batch Size = {batch_size}')

        n_epoch = 5000
        n_epoch_output = 200

        for epoch in range(n_epoch):
            train_loss = 0.0
            # net.train()
            for batch_inputs, batch_labels, batch_labels_d in train_dataloader:
                # Send batch to the device
                # inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
                # Clear the gradients
                optimizer.zero_grad()
                # Forward pass
                batch_outputs = net(batch_inputs)
                # Differentiate outputs
                batch_outputs_d = (batch_outputs[:,1:] - batch_outputs[:,:-1]) / dx * mask
                # Compute loss
                loss = criterion(batch_outputs, batch_labels) + mu * criterion(batch_outputs_d,batch_labels_d)
                # Backward pass
                loss.backward()
                # Optimization step
                optimizer.step()
                # Running training loss
                train_loss += loss.item()


            # Print training statistics
            train_loss = train_loss/len(train_dataloader)
            if epoch % n_epoch_output == max(n_epoch_output - 1,0):
                print('Epoch: %d \t Training Loss: %.16f \t Learning Rate: %.10f' % 
                        (epoch + 1, train_loss, optimizer.param_groups[0]['lr']), flush = True)
                
            # Update scheduler after epoch
            scheduler.step()

        print(f"Finished Training ({j}, {k})\n")

        # Save network parameters
        param_outputs = {}
        for name, param in net.named_parameters():
            param_outputs.update({str(name).replace(".","_"): np.array(param.data)})
        
        io.savemat("data_semilinear/NN_param_Mx{:d}_My{:d}_({:d},{:d})_Ntrain{:d}_dxb{:.3e}.mat".format(Mx, My, j, k, Nt, Dx_buffer), param_outputs)

