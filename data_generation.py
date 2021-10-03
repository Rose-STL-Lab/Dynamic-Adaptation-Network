import torch
import numpy as np
import torch.nn.functional as F
import os
from netCDF4 import Dataset
from phi.flow import *  
import pylab
import os
import progressbar
import warnings
warnings.filterwarnings("ignore")

# Function for calculating the vorticity
def vorticity(u,v):
    return field_grad(v, 0) - field_grad(u, 1)

def field_grad(f, dim):
    # dim = 1: derivative to x direction, dim = 2: derivative to y direction
    dx = 1
    dim += 1
    N = len(f.shape)
    out = torch.zeros(f.shape)#.to(device)
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    # 2nd order interior
    slice1[-dim] = slice(1, -1)
    slice2[-dim] = slice(None, -2)
    slice3[-dim] = slice(1, -1)
    slice4[-dim] = slice(2, None)
    out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2*dx)
    
    # 2nd order edges
    slice1[-dim] = 0
    slice2[-dim] = 0
    slice3[-dim] = 1
    slice4[-dim] = 2
    a = -1.5 / dx
    b = 2. / dx
    c = -0.5 / dx
    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    slice1[-dim] = -1
    slice2[-dim] = -3
    slice3[-dim] = -2
    slice4[-dim] = -1
    a = 0.5 / dx
    b = -2. / dx
    c = 1.5/ dx

    out[tuple(slice1)] = a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
    return out



rawdata_direc = "raw_data/"
os.mkdir(rawdata_direc)

preprocessed_direc = "sliced_data/"
os.mkdir(preprocessed_direc)

Resolution
res = [64, 64]
for i in range(1,26):
    buoyancy_factor = i
    world = World()
    fluid = world.add(Fluid(Domain(res, boundaries = CLOSED), buoyancy_factor = i, batch_size=1), physics=IncompressibleFlow())
    
    # Position and Radius of Inlet Velocity
    inflow = Inflow(Sphere(center=[[16,res[0]//2]], radius = 4), rate = 1)
    world.add(inflow)
    Ws = []
    
    # 500-step Simulation
    for i in range(500):
        world.step(dt = 1)
        if i == 1:
            world.remove(inflow)
        Ws.append(np.concatenate([fluid.velocity.unstack()[0].data[:,:res[0],:res[1],:], fluid.velocity.unstack()[1].data[:,:res[0],:res[1],:]], axis = -1))
    Ws = np.concatenate(Ws)
    torch.save(torch.from_numpy(Ws[1:]).float(), rawdata_direc + "data" + str(int(buoyancy_factor))+".pt")
    
    
# Calculate the mean of velocity and the standard deviation of the velocity magnitude
dataset = []
for file in sorted(os.listdir(rawdata_direc)):
    data = torch.load(rawdata_direc + file)
    dataset.append(data)
    
dataset = torch.cat(dataset, dim = 0)
avg_vec = torch.mean(dataset, dim = (0,1,2), keepdim = True)
std_vec = torch.std(torch.sqrt(dataset[...,0]**2 + dataset[...,1]**2))

# Use sliding window to generate samples for training.
vor = []
for i in range(1,26):
    os.mkdir(preprocessed_direc + "data" + str(i))
    data = torch.load(rawdata_direc + "data" + str(i) + ".pt")
    data = (data - avg_vec)/std_vec
    # Calculate the mean vorticity of the training set of each task
    vor.append(torch.mean(torch.abs(vorticity(data[:350,:,:,0], data[:350,:,:,1]))).numpy().item())
    for j in range(0, len(data) - 50):
        torch.save(data[j:j+50].transpose(-1,-2).transpose(-2,-3).float(), preprocessed_direc + "data" + str(i) + "/" + "sample_" + str(j) + ".pt")
        
# Save Task Parameter
np.save("task_parameter_vorticity_turbulence.npy", vor)
