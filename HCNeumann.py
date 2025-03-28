"""
    29th January 2025

    Code for the preprint "HARD-CONSTRAINING NEUMANN BOUNDARY CONDITIONS IN PHYSICS-INFORMED NEURAL NETWORKS VIA FOURIER FEATURE EMBEDDINGS"

    Code to test the use of hard-constraining Neumann boundary conditions for a simple heat equation
    Similar heat equation example: 
        https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html

    Code is written in an as-simple-as-possible way
"""
import deepxde as dde
import numpy as np
from scipy.io import loadmat
from deepxde.backend import torch
import os
script_dir = os.path.dirname(__file__)


#Plotting stuff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
import time #For time measurement

# for reproducability
dde.config.set_random_seed(0)


"""  
    Classes for initial conditions and (explicitly known) reference solution
"""

#####
#   Abstract parent class for all initial conditions
#####
from abc import ABC, abstractmethod
class InitCond(ABC):
    @abstractmethod
    def eval(self, x): #Evaluate initial condition at tensor-type input x
        pass

    
#####
#   Initial condition for cosine with some frequency and an optional factor
#   g(x) = factor*cos(Pi*freq*x)
#####
class InitCos(InitCond):
    def __init__(self, freq, factor=1.):
        self.factor = factor
        self.freq = freq

    def eval(self, x):
        if isinstance(x, np.ndarray): #DeepXDE uses numpy arrays to compute loss function
            if x.ndim > 1: #Make function compatible for multiple inputs (either only x-vector or (x,t)-input of DeepXDE)
                xx = x[:,0:1]  
            else:
                xx = x
            return self.factor*np.cos(np.pi*self.freq*xx)
        else:
            if x.dim() > 1: 
                xx = x[:,0:1]  
            else:
                xx = x
            return self.factor*torch.cos(torch.pi*self.freq*xx)
        

#####
#   Initial condition for sum of multiple cosines with frequencies and optional factors
#####
class InitMultCos(InitCond):
    def __init__(self, freqs, factors=None):
        self.freqs = freqs

        if factors is None:
            self.factors = [1.]*len(self.freqs)
        else:
            self.factors = factors

        #Transform the input lists to tensors and arrays for efficient evaluation
        device_for_tensors = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.freqs_torch = torch.tensor(self.freqs, dtype=torch.bool).to(device_for_tensors)
        self.factors_torch = torch.tensor(self.factors, dtype=torch.float32).to(device_for_tensors)

        self.freqs_arr = np.array(self.freqs)
        self.factors_arr = np.array(self.factors)

        #Create list of the individual cosine parts
        self.cos_list = []
        for ifr in range(len(self.freqs)):
            self.cos_list.append(InitCos(self.freqs[ifr],self.factors[ifr]))


    def eval(self, x):
        if isinstance(x, np.ndarray): 
            if x.ndim > 1: 
                xx = x[:,0:1]  
            else:
                xx = x
            return np.reshape(np.sum(self.factors_arr*np.cos(np.pi*xx*self.freqs_arr), axis=1), (-1,1))
        else:
            raise ValueError("Illegal datatype in initial condition")

    

#####
#   Initial condition for step function taking some value (default: 1) on [0,h] and 0 otherwise
#####
class InitStep(InitCond):
    def __init__(self, h, factor=1):
        self.h = h
        self.factor = factor
        


    def eval(self, x):
        if isinstance(x, np.ndarray): #DeepXDE uses numpy arrays to compute loss function 
            if x.ndim > 1: #Make function compatible for mutliple inputs (either only x-vector or (x,t)-input of DeepXDE)
                xx = x[:,0:1]  
            else:
                xx = x
            return np.where(xx < self.h, 0.*xx+self.factor, 0.*xx)
        else:
            if x.dim() > 1:
                xx = x[:,0:1]  
            else:
                xx = x
            return torch.where(xx < self.h, 0.*xx+self.factor, 0.*xx)

#####
#   Initial condition for polynomial
#####
class InitPolynom(InitCond):
    def __init__(self, coeffs):
        self.coeffs = coeffs #Contains the coefficients of the polynomial in monom base representation
        self.exponents = list(range(len(coeffs))) #List containing the exponents used for the monoms

        #Transform the lists to tensors for efficient calculation
        device_for_tensors = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coeffs_torch = torch.tensor(self.coeffs, dtype=torch.float32).to(device_for_tensors)
        self.exponents_torch = torch.tensor(self.exponents, dtype=torch.int).to(device_for_tensors)

        self.coeffs_arr = np.array(self.coeffs)
        self.exponents_arr = np.array(self.exponents)

    def eval(self, x):
        if isinstance(x, np.ndarray): #DeepXDE uses numpy arrays to compute loss function -> transform
            if x.ndim > 1: #Make function compatible for mutliple inputs (either only x-vector or (x,t)-input of DeepXDE)
                xx = x[:,0:1]  
            else:
                xx = x
            
            return np.reshape(np.sum(self.coeffs_arr*(xx**self.exponents_arr), axis=1), (-1,1))
        else: #x-array is tensorflow tensor
            raise ValueError("Illegal datatype in initial condition")
        
      

#####
#   Class for explicit reference solution of 1D diffusion problem:
#
#   \partial_t u = D \partial_x^2 u
#       on (x,t) \in [0,1] \times [0,1],
#   \partial_x u(0,t) = 0 = \partial_x u(1,t),
#   u(x,0) = g(x),
# 
#   where g=g(x) is a given initial condition.
#   The explicit solution is represented via a Fourier series. 
#####
class RefSol:
    def __init__(self, D, initialcond, NrFourier=200):
        self.D = D #Diffusion coefficient in PDE
        self.initialcond = initialcond #Initial condition; a class as above
        self.NrFourier = NrFourier #Number of terms in Fourier series used to evaluate reference solution (in general settings...)

        #Pre-compute Fourier coefficients in settings where one has to 
        if isinstance(self.initialcond, InitStep):
            self.fourier_coeff = [2.*self.initialcond.h] #Initialize the 0th coefficient
            self.fourier_freqs = [0.]
            for ifr in range(1,NrFourier):
                self.fourier_coeff.append((2./(np.pi*ifr))*np.sin(np.pi*ifr*self.initialcond.h))
                self.fourier_freqs.append(np.pi*ifr)
        elif isinstance(self.initialcond, InitPolynom):
            self.fourier_freqs = [0.]
            fourier_coeff = [0.]
            for ifr in range(1,NrFourier):
                fourier_coeff.append(0.)
                self.fourier_freqs.append(np.pi*ifr)

            fourier_coeff = np.array(fourier_coeff)

            for iexp in range(len(self.initialcond.coeffs)):
                curr_exp = self.initialcond.exponents[iexp]
                curr_coeff = self.initialcond.coeffs[iexp]

                #Fourier coefficients for monoms have been calculated with WolframAlpha, e.g., https://www.wolframalpha.com/input?i=2*int_0%5E1+x%5E5+cos%28Pi*n*x%29+dx
                if curr_exp == 0:
                    curr_fourier_coeff = [2.]
                    for ifr in range(1,NrFourier):
                        curr_fourier_coeff.append(0.)
                elif curr_exp == 1:
                    curr_fourier_coeff = [2./2.]
                    for ifr in range(1,NrFourier):
                        curr_fourier_coeff.append((2.*(-1)**ifr-2.)/((np.pi**2)*(ifr**2)))
                elif curr_exp == 2:
                    curr_fourier_coeff = [2./3.]
                    for ifr in range(1,NrFourier):
                        curr_fourier_coeff.append((4.*(-1)**ifr)/((np.pi**2)*(ifr**2)))
                elif curr_exp == 3:
                    curr_fourier_coeff = [2./4.]
                    for ifr in range(1,NrFourier):
                        curr_fourier_coeff.append((6.*((np.pi**2)*(ifr**2)-2)*(-1)**ifr+12.)/((np.pi**4)*(ifr**4)))
                elif curr_exp == 4:
                    curr_fourier_coeff = [2./5.]
                    for ifr in range(1,NrFourier):
                        curr_fourier_coeff.append((8.*((np.pi**2)*(ifr**2)-6)*(-1)**ifr)/((np.pi**4)*(ifr**4)))
                elif curr_exp == 5:
                    curr_fourier_coeff = [2./6.]
                    for ifr in range(1,NrFourier):
                        curr_fourier_coeff.append((10.*((np.pi**4)*(ifr**4)-12.*(np.pi**2)*(ifr**2)+24.)*(-1)**ifr-240.)/((np.pi**6)*(ifr**6)))
                else:
                    raise ValueError("Illegal exponent in polynomial initial condition!")
                
                curr_fourier_coeff = np.array(curr_fourier_coeff)
                fourier_coeff = fourier_coeff + curr_coeff*curr_fourier_coeff
            
            self.fourier_coeff = fourier_coeff.tolist()

    ## Evaluate reference solution
    ## Is only called once at the start of the training -> does not need to be implemented super efficiently
    def eval(self, x):
        if isinstance(x, np.ndarray): #DeepXDE uses numpy arrays to compute loss function -> transform
            x = torch.from_numpy(x)   
        xx = x[:,0:1]  
        t = x[:,1:2]
        if isinstance(self.initialcond, InitCos):
            return self.initialcond.eval(x)*torch.exp(-self.D*torch.pi*torch.pi*self.initialcond.freq*self.initialcond.freq*t)
        elif isinstance(self.initialcond, InitMultCos):
            returner = 0.
            for ifr in range(len(self.initialcond.freqs)):
                returner += self.initialcond.cos_list[ifr].eval(x)*torch.exp(-self.D*torch.pi*torch.pi*self.initialcond.freqs[ifr]*self.initialcond.freqs[ifr]*t)
            return returner    
        elif isinstance(self.initialcond, InitStep) or isinstance(self.initialcond, InitPolynom):
            returner = self.fourier_coeff[0]/2.
            for ifr in range(1,self.NrFourier):
                returner += self.fourier_coeff[ifr]*torch.cos(self.fourier_freqs[ifr]*xx)*torch.exp(-self.D*self.fourier_freqs[ifr]*self.fourier_freqs[ifr]*t)
            if isinstance(self.initialcond, InitPolynom):
                return returner
            else:
                return torch.where(t < 1.e-8, self.initialcond.eval(x), returner) #At t=0, just return the step function instead of the Fourier approximation due to Gibb's phenomenonn in the step function case
        else:
            raise ValueError("Illegal type of initial condition!")


#####
#   Function to create reference data
#####
def gen_reference(IC, D, Nr_x, Nr_t, NrFourier=200):
    ref_sol = RefSol(D, IC, NrFourier=NrFourier)
    X = np.linspace(0, 1., Nr_x)
    T = np.linspace(0, 1., Nr_t)
    X_GRID, T_GRID = np.meshgrid(X, T)
    grid_points = np.vstack([X_GRID.ravel(), T_GRID.ravel()]).T

    u_eval = ref_sol.eval(grid_points).reshape((Nr_t, Nr_x, 1))
    return X_GRID, T_GRID, grid_points, u_eval


#####
#   Callback to write distance to reference data into file
#####
class Callback_Ref(dde.callbacks.Callback):
    def __init__(self, iter_period, ref_grid, ref_val, Nr_x_ref, Nr_t_ref, filename):
        super().__init__()
        self.iter_period = iter_period

        self.ref_grid = ref_grid #Grid on which reference solution has been evaluated
        self.ref_val = ref_val #Values of reference solution on the grid
        self.Nr_x_ref = Nr_x_ref
        self.Nr_t_ref = Nr_t_ref
        self.filename = filename

        self.best_trainloss = np.inf
        self.relerror4bestloss = 0.

        self.time_start = None
        self.time_lastout = None

        #Clear file
        if not self.filename is None:
            open(self.filename, 'w').close()

    def on_batch_end(self):
        curr_iter = self.model.train_state.step
        curr_epoch = self.model.train_state.epoch
        if self.time_start is None:
            self.time_start = time.perf_counter()
            self.time_lastout = time.perf_counter()

        if curr_iter % self.iter_period == 0:
            time_curr = time.perf_counter()
            time_sincelastout = time_curr - self.time_lastout
            time_sincestart = time_curr - self.time_start

            pred = self.model.predict(self.ref_grid).reshape((self.Nr_x_ref, self.Nr_t_ref, 1))
            # compute relative error of prediction to reference solution
            y_diff = torch.abs(torch.from_numpy(pred)-self.ref_val)
            y_diff_scaled = y_diff/np.linalg.norm(self.ref_val)
            rel_error = torch.norm(y_diff_scaled).numpy() 

            curr_trainloss = np.sum(self.model.train_state.loss_train) #Best loss (restricted to the loss values seen in this callback!)
            if curr_trainloss < self.best_trainloss:
                self.best_trainloss = curr_trainloss
                self.relerror4bestloss = rel_error
                print(f'Epoch {curr_epoch}. {time_sincelastout:.1f} s since last output. Relative error to reference solution = {rel_error}. New best loss.')
            else:
                print(f'Epoch {curr_epoch}. {time_sincelastout:.1f} s since last output. Relative error to reference solution = {rel_error}.')

            if not self.filename is None:
                with open(self.filename, 'a') as file:
                    file.write(f"{curr_iter}\t{rel_error}\t{self.relerror4bestloss}\t{time_sincestart}\n")
            
            self.time_lastout = time_curr


#####
#   Callback to write training time into file
#   Creates a file with following outputs:
#   |1) Iteration number |2) Time since start |3) Time since last output/line |4) Time per 1000 Iterations (averaged since start)
#####
class Callback_Traintime(dde.callbacks.Callback):
    def __init__(self, iter_period, filename):
        super().__init__()
        self.iter_period = iter_period #Print time after each iter_period elapsed periods
        self.filename = filename

        self.time_start = None
        self.time_lastout = None

        #Clear file
        if not self.filename is None:
            open(self.filename, 'w').close()

    def on_batch_end(self):
        curr_iter = self.model.train_state.step

        if self.time_start is None:
            self.time_start = time.perf_counter()
            self.time_lastout = time.perf_counter()

        if curr_iter % self.iter_period == 0:
            time_curr = time.perf_counter()
            time_sincelastout = time_curr - self.time_lastout
            time_sincestart = time_curr - self.time_start

            time_sincestart_per1000iters = time_sincestart*1000./curr_iter

            if not self.filename is None:
                with open(self.filename, 'a') as file:
                    file.write(f"{curr_iter}\t{time_sincestart}\t{time_sincelastout}\t{time_sincestart_per1000iters}\n")
            
            self.time_lastout = time_curr


from deepxde.nn.pytorch import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config

"""  
    Class for Fourier feature embedding
"""
class FourierEmbedding:
    def __init__(self, Nr_freqs, sigma, only_ints=False, only_cos=False, alternating_cossin=False, scale01=False, nec_freqs=None):
        self.Nr_freqs = Nr_freqs #Number of frequencies used for embedding (i.e., size of embedding vector)
        self.sigma = sigma #Standard deviation used to sample random frequencies
        self.only_ints = only_ints #Determines whether one only wants to sample integers or general floats
        self.only_cos = only_cos #Determines whether one only wants to use cos
        self.alternating_cossin = alternating_cossin #Determines whether one wants to always use alternating cos-sin application, where each (cos,sin)-pair is applied to the same frequency
        self.scale01 = scale01 #Determines whether one wants to apply the slight affine linear transformation to ensure that the output lies in [0,1]
        self.nec_freqs = nec_freqs #List of necessary frequencies

        self.Neumanns_hc = (self.only_ints and self.only_cos) #Parameter that tells whether we are hard-constraining Neumann boundary conditions via the new approach

        self.freqs = []
        if self.nec_freqs is not None:
            if alternating_cossin:
                nec_freqs_dublicated = [item for item in self.nec_freqs for _ in range(2)]
                self.nec_freqs = nec_freqs_dublicated
            self.freqs.extend(self.nec_freqs)

        while len(self.freqs) < self.Nr_freqs:
            rand_freq = np.random.normal(0., self.sigma)
            if self.only_cos:
                rand_freq = np.abs(rand_freq)
            if self.only_ints:
                rand_freq = int(np.rint(rand_freq))
            use_freq = True
            if self.only_cos:
                if rand_freq in self.freqs:
                    use_freq = False
                if rand_freq < 1.e-8:
                    use_freq = False
            
            if use_freq:
                self.freqs.append(rand_freq)
                if self.alternating_cossin and len(self.freqs) < self.Nr_freqs: #In alternating case, each frequency is used twice: once for cos, once for sin
                    self.freqs.append(rand_freq)

        self.Nr_freqs = int(len(self.freqs))
        
        self.useCos = []
        if self.only_cos:
            self.useCos = [True]*self.Nr_freqs
        elif self.alternating_cossin:
            self.useCos = [i % 2 == 0 for i in range(self.Nr_freqs)]
        else:
            self.useCos = np.random.choice([True,False], len(self.freqs))

        device_for_tensors = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.useCos_torch = torch.tensor(self.useCos, dtype=torch.bool).to(device_for_tensors)
        self.freqs_torch = torch.tensor(self.freqs, dtype=torch.float32).to(device_for_tensors)

    def print_description(self):
        print("Fourier feature embedding")
        print(f"Embedding Size: {self.Nr_freqs}")
        print(f"Standard Deviation used for sampling: sigma={self.sigma}")
        print(f"Frequencies: {self.freqs}")
        print(f"Cosines applied: {self.useCos}")


class MLP(NN):
    """
        MLP with option to hard-constrain in existing way, i.e., as described in Sc. 5.1.2 of the paper https://doi.org/10.1016/j.cma.2021.114333 
            Class is based on DeepXDE's (pytorch) MLP class https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/pytorch/fnn.py
    """
    def __init__(
        self, layer_sizes, activation, kernel_initializer, regularization=None, HC_Neumann0=False
    ):
        super().__init__()
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError("Total number of activation functions do not match with sum of hidden layers and output layer!")
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)))
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)

        self.HC_Neumann0 = HC_Neumann0 #Determine whether one wants to hard-constraint vanishing Neumann boundary data at x=0 and x=1

        if self._input_transform is not None and self.HC_Neumann0:
            print("WARNING! You are using both input transformation together with the output transformation to ensure vanishing Neumann boundary data. This has not been tested... Good luck!")
        
        if self._output_transform is not None and self.HC_Neumann0:
            print("WARNING! You are using an output transformation together with the output transformation to ensure vanishing Neumann boundary data. This might lead to unexpected results... Good luck!")

        

    def forward(self, inputs):
        def usual_MLP(x):
            if self._input_transform is not None:
                x = self._input_transform(x)
            for j, linear in enumerate(self.linears[:-1]):
                x = (
                    self.activation[j](linear(x))
                    if isinstance(self.activation, list)
                    else self.activation(linear(x))
                )
            x = self.linears[-1](x)
            return x
        x = usual_MLP(inputs)

        if self.HC_Neumann0:
            xx = inputs[:,0:1]
            tt = inputs[:,1:2]

            x0_tt = torch.cat([0.*xx, tt], 1) #Represents (0,t)
            x1_tt = torch.cat([0.*xx+1., tt], 1) #Represents (1,t)

            uNN_x0 = usual_MLP(x0_tt) #Represents u(0,t), where u is original net
            uNN_x1 = usual_MLP(x1_tt) #Represents u(1,t)

            uNN_x0_prime = torch.autograd.grad(outputs=uNN_x0, inputs=x0_tt, grad_outputs=torch.ones_like(xx), create_graph=True)[0][:, 0:1] #Represents u'(0,t) 
            uNN_x1_prime = torch.autograd.grad(outputs=uNN_x1, inputs=x1_tt, grad_outputs=torch.ones_like(xx), create_graph=True)[0][:, 0:1] #Represents u'(1,t) 

            additional_addend = - uNN_x0_prime*xx*(xx-1.)*(xx-1.) - uNN_x1_prime*xx*xx*(xx-1.) + 0.*xx
            x = x + additional_addend

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x



"""  
    Function that wraps the training of the model and plotting its predictions
    Default parameters are taken from those from the heat example in the Fourier feature paper, cf. https://doi.org/10.1016/j.cma.2021.113938
        Code with parameters: cf. https://github.com/PredictiveIntelligenceLab/MultiscalePINNs/blob/main/heat1D/heat1D.py
"""
def train_plot(output_folder, IC, D=1.,hc_Neumann_lit=False, eliminate_points=True,
               train_its=1_000_000, lr = 1.e-4, Nr_hidden_layers = 3, Nr_neurons_per_layers = 100, 
               fourier_embedding=None,
               Nr_train_domain=20000, Nr_train_boundary = 1000, Nr_train_init = 500, Nr_test = 10000,
               loss_weights=None,
               Nr_x_ref=500, Nr_t_ref=500,
               display_every = 100, Nr_out_time = 16, Nr_out_space = 101, Nr_out_timesteps=5,
               debug_HC=False):
    
    # Setup output folder
    output_folder_name = output_folder + '/'
    results_dir = os.path.join(script_dir, output_folder_name) # Subfolder for all outputs
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    ##  Define geometry
    geom = dde.geometry.Interval(0.,1.) #Fixed space domain: Interval [0,1.]
    timedomain = dde.geometry.TimeDomain(0, 1.) #Fixed time interval [0,1.]
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    ## Define the PDE -> Simple 1D heat equation with diffusion coefficient D
    def pde(x,y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - D*dy_xx
        
    ## Initial and boundary conditions:
    bc = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary) # Vanishing Neumann boundary conditions
    ic = dde.icbc.IC(geomtime, IC.eval, lambda _, on_initial: on_initial) #Initial condition given by input IC

    ## Set up loss functions (including names for plotting) depending on hard-constraint technique
    loss_cols = ['PDE', 'IC'] #List for names of loss terms
    conditions = [ic] #List of conditions (boundary + initial)

    Neumann_hardconstrained = False #Variable that determines whether one is hard-constraining the Neumann boundary data
    if hc_Neumann_lit:
        Neumann_hardconstrained = True
    if fourier_embedding is not None:
        if fourier_embedding.Neumanns_hc:
            Neumann_hardconstrained = True
    
    if not Neumann_hardconstrained:
        loss_cols.append('BC')
        conditions.append(bc)

    if debug_HC: #Debug the correct hard-constraining by just including all loss terms -> check whether they are actually zero
        conditions = [bc, ic]
        loss_cols = ['PDE', 'BC', 'IC']
        eliminate_points = False


    ## Eliminate points, e.g., set boundary points to 0 if bc is imposed via hard-constraint
        # Note that the PDE residual is always computed at all points (domain, boundary, initial) in DeepXDE
    if eliminate_points:
        if Neumann_hardconstrained:
            Nr_train_boundary = 0

    
    def feature_trafo_FourierFeature_x(x): #Input Trafo: Spatial Fourier Embedding with prescribed frequencies
        xx = x[:,0:1]

        #Requires FourierEmbedding object to exist
        if isinstance(xx, np.ndarray):
            if fourier_embedding.scale01:
                xx_extend = np.where(fourier_embedding.useCos, .5+.5*np.cos(np.pi*xx*fourier_embedding.freqs), .5+.5*np.sin(np.pi*xx*fourier_embedding.freqs))
            else:
                xx_extend = np.where(fourier_embedding.useCos, np.cos(np.pi*xx*fourier_embedding.freqs), np.sin(np.pi*xx*fourier_embedding.freqs))
        else:
            if fourier_embedding.scale01:
                xx_extend = torch.where(fourier_embedding.useCos_torch, .5+.5*torch.cos(torch.pi*xx*fourier_embedding.freqs_torch), .5+.5*torch.sin(torch.pi*xx*fourier_embedding.freqs_torch))
            else:
                xx_extend = torch.where(fourier_embedding.useCos_torch, torch.cos(torch.pi*xx*fourier_embedding.freqs_torch), torch.sin(torch.pi*xx*fourier_embedding.freqs_torch))

        return torch.cat([xx_extend, x[:,1:2]], 1)

        
    ## Define neural network and apply transformations
    input_dim = 2 #Dimension of total input vector
    if fourier_embedding is not None:
        input_dim = 1 + fourier_embedding.Nr_freqs #Temporal input and spatial embedding

    net = MLP([input_dim] + [Nr_neurons_per_layers] * Nr_hidden_layers + [1], "tanh", "Glorot normal", HC_Neumann0=hc_Neumann_lit)
    if fourier_embedding is not None:
        print("The following Fourier embedding is applied:")
        fourier_embedding.print_description()
        net.apply_feature_transform(feature_trafo_FourierFeature_x) 

    ##Setup reference solution
    X_ref, T_ref, grid_ref, y_true = gen_reference(IC, D, Nr_x=Nr_x_ref, Nr_t=Nr_t_ref)
    filename_err_rel_history = output_folder + '/err_rel_hist.dat'
    callback_ref = Callback_Ref(display_every, grid_ref, y_true, Nr_x_ref, Nr_t_ref, filename_err_rel_history)

    ##Setup Time measurement
    filename_time = output_folder + '/time_hist.dat'
    callback_time = Callback_Traintime(display_every, filename_time)



    ## Define PDE problem and train model 
    data = dde.data.TimePDE(geomtime, pde, conditions, num_domain=Nr_train_domain, num_boundary=Nr_train_boundary, num_initial=Nr_train_init, num_test=Nr_test)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss_weights=loss_weights) 
    losshistory, train_state = model.train(iterations=train_its, callbacks=[callback_ref,callback_time], display_every=display_every)

    print("Done with training :) Now save & plot.\n")

    dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=results_dir)
    
    ## Function to plot loss curve(s)
    def plot_losses(hist, loss_cols):
        loss_train = np.sum(hist.loss_train, axis=1)
        loss_test = np.sum(hist.loss_test, axis=1)

        # create separate plots for train and test
        fig_train = plt.figure()
        ax_train = fig_train.add_subplot(111)
        fig_test= plt.figure()
        ax_test = fig_test.add_subplot(111)
        ax_train.set_title('Train Losses')
        ax_test.set_title('Test Losses')

        ax_train.plot(hist.steps, loss_train, label="Total Train loss")
        ax_test.plot(hist.steps, loss_test, label="Total Test loss")

        for i, label in enumerate(loss_cols):

            train_loss = [item[i] for item in hist.loss_train]
            test_loss = [item[i] for item in hist.loss_test]
            ax_train.semilogy(hist.steps, train_loss, '.', label=label)
            ax_test.semilogy(hist.steps, test_loss, '.', label=label)

        for i in range(len(hist.metrics_test[0])):
            ax_train.plot(
                    hist.steps,
                    np.array(hist.metrics_test)[:, i],
                    label="Test metric",
            )
        for ax in [ax_train, ax_test]:
            ax.set_xlabel("# Steps")
            ax.grid()
            ax.legend(loc = "upper right")

        fig_train.savefig(os.path.join(results_dir, 'losses_train.png'))
        fig_test.savefig(os.path.join(results_dir, 'losses_test.png'))
        plt.close(fig_train)
        plt.close(fig_test)

    ## Function to plot history of relative error to reference solution
    def plot_relerr_history():
        steps = []
        relerrs = []
        relerrs4bestloss = []
        traintimes = []
        with open(filename_err_rel_history, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                steps.append(int(columns[0]))
                relerrs.append(float(columns[1]))
                relerrs4bestloss.append(float(columns[2]))
                traintimes.append(float(columns[3]))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Relative Error to Reference Solution')

        ax.plot(steps, relerrs, label='Rel. error')
        ax.plot(steps, relerrs4bestloss, label='Rel. error 4 best loss')

        ax.set_xlabel("# Steps")
        ax.set_ylabel("Rel. error")
        ax.set_yscale('log')
        ax.grid()
        ax.legend(loc = "upper right")
        ax.set_xlim(left=0, right=None)

        fig.savefig(os.path.join(results_dir, 'err_rel_history.png'))
        plt.close(fig)

    ## Function to plot history of relative error to reference solution and losses in same plot (side by side)
    def plot_relerr_history_and_losses(hist):
        fig, ax = plt.subplots(ncols=2,figsize=(21, 7))

        loss_train = np.sum(hist.loss_train, axis=1)
        loss_test = np.sum(hist.loss_test, axis=1)

        ax[0].plot(hist.steps, loss_train, label="Total Train loss")
        ax[0].plot(hist.steps, loss_test, label="Total Test loss")

        ax[0].set_title('Train and Test Losses')
        ax[0].set_xlabel("# Steps")
        ax[0].set_ylabel("Losses")
        ax[0].set_yscale('log')
        ax[0].grid()
        ax[0].legend(loc = "upper right")
        ax[0].set_xlim(left=0, right=None)


        steps = []
        relerrs = []
        relerrs4bestloss = []
        with open(filename_err_rel_history, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                steps.append(int(columns[0]))
                relerrs.append(float(columns[1]))
                relerrs4bestloss.append(float(columns[2]))

        ax[1].plot(steps, relerrs, label='Rel. error')
        ax[1].plot(steps, relerrs4bestloss, label='Rel. error 4 best loss')

        ax[1].set_title('Relative Error to Reference Solution')
        ax[1].set_xlabel("# Steps")
        ax[1].set_ylabel("Rel. error")
        ax[1].set_yscale('log')
        ax[1].grid()
        ax[1].legend(loc = "upper right")
        ax[1].set_xlim(left=0, right=None)

        fig.savefig(os.path.join(results_dir, 'err_rel_and_losses.png'))

    ## Function to create 3D scatter plot of PINN prediction 
    def plot_and_save_3d_scatter(Xsc, Tsc, Zsc, plot_filename, show_plot=False):
        # Plot the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Flatten X, Y, Z for scatter plot
        X_flat = Xsc.ravel()
        T_flat = Tsc.ravel()
        Z_flat = Zsc.ravel()

        # Create scatter plot where color represents the Z values
        scatter = ax.scatter(X_flat, T_flat, Z_flat, c=Z_flat, cmap='viridis')

        # Add labels
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('3D Scatter Plot of u(x,t)')

        ax.set_xlim(0, 1.)
        ax.set_ylim(0, 1.)

        ax.view_init(25, 60)

        # Add color bar
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

        # Save the 3D scatter plot
        plt.savefig(results_dir + plot_filename)
        if show_plot:
            plt.show()

    ## Function to create heatmap plot, both of solution and of (relative) error
    def plot_and_save_heatmap(Xsc, Tsc, Zsc, plot_filename, plot_title, show_plot=False, min_to_zero=False):
        fig, ax =plt.subplots(figsize=(10, 8))
        
        # Flatten X, Y, Z for plot
        X_flat = Xsc.ravel()
        T_flat = Tsc.ravel()
        Z_flat = Zsc.reshape(Xsc.shape)

        if not min_to_zero:
            heatmap = ax.pcolormesh(Xsc, Tsc, Zsc.reshape(Xsc.shape), cmap="RdBu")
        else:
            heatmap = ax.pcolormesh(Xsc, Tsc, Zsc.reshape(Xsc.shape), cmap="RdBu", vmin=0)
        ax.set_title(plot_title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label("u(x,t)")

        # Save plot
        plt.savefig(results_dir + plot_filename)
        if show_plot:
            plt.show()

    ## Function to evaluate trained PINN on grid
    def evaluate_on_grid(model, Np_t, Np_x, t_min=0., t_max=1., x_min=0., x_max=1.):
        t_grid = np.linspace(t_min, t_max, Np_t)
        x_grid = np.linspace(x_min, x_max, Np_x)

        X_GRID, T_GRID = np.meshgrid(x_grid, t_grid)
        grid_points = np.vstack([X_GRID.ravel(), T_GRID.ravel()]).T

        C_EVAL = model.predict(grid_points).reshape((Np_t, Np_x, 1))
        
        return X_GRID, T_GRID, C_EVAL   
    
    ## Function to plot solution at different time steps
    def plot_timesteps(model, Np_x, ts, filename):
        X = np.linspace(0, 1., Np_x)
        Np_t = ts.size
        X_GRID, T_GRID = np.meshgrid(X, ts)
        grid_points = np.vstack([X_GRID.ravel(), T_GRID.ravel()]).T

        C_EVAL = model.predict(grid_points).reshape((Np_t, Np_x, 1))

        
        output_folder_timesteps = output_folder + '/timesteps/' 
        results_timesteps_dir = os.path.join(script_dir, output_folder_timesteps) # SubSubfolder
        if not os.path.isdir(results_timesteps_dir):
            os.makedirs(results_timesteps_dir)

        for ip in range(ts.size):
            plt.figure(figsize=(10, 6))
            plt.plot(X, C_EVAL[ip,:,0])

            plt.xlabel('x')
            plt.ylabel('u(x,t)')
            plt.ylim(-1.1,1.1)
            plt.title(f'Solution u at Time t={ts[ip]}')
            plt.legend()
            plt.grid(True)
            plt.savefig(results_timesteps_dir + filename + '_' + str(ip).zfill(4) + '.png')
            plt.close('all')


    ## Apply above functions
    plot_losses(losshistory, loss_cols)
    plot_relerr_history()
    plot_relerr_history_and_losses(losshistory)

    X, T, C = evaluate_on_grid(model, Np_t=Nr_out_time, Np_x=Nr_out_space)
    plot_and_save_3d_scatter(X, T, C[:,:,0], 'scatter_plot')
    plot_timesteps(model, Nr_out_space, np.linspace(0,1., Nr_out_timesteps+1), 'sol')

    ## Compare with reference solution
    X_ref, T_ref, grid_ref, y_true = gen_reference(IC, D, Nr_x=Nr_x_ref, Nr_t=Nr_t_ref)
    y_pred = model.predict(grid_ref).reshape((Nr_x_ref, Nr_t_ref, 1))

    plot_and_save_3d_scatter(X_ref, T_ref, y_pred, 'scatter_sol')
    plot_and_save_3d_scatter(X_ref, T_ref, y_true, 'scatter_ref')

    plot_and_save_heatmap(X_ref, T_ref, y_pred[:,:,0], 'heat_sol', 'PINN solution')
    plot_and_save_heatmap(X_ref, T_ref, y_true[:,:,0], 'heat_ref', 'Reference solution')

    y_diff = torch.abs(torch.from_numpy(y_pred)-y_true)
    y_diff_scaled = y_diff/np.linalg.norm(y_true)
    plot_and_save_heatmap(X_ref, T_ref, y_diff[:,:,0], 'heat_err', 'Absolute Difference of PINN and reference solution', min_to_zero=True)
    plot_and_save_heatmap(X_ref, T_ref, y_diff_scaled[:,:,0], 'heat_err_rel', 'Relative Difference of PINN and reference solution', min_to_zero=True)

    relative_error = dde.metrics.l2_relative_error(y_true, y_pred)
    print("L2 relative error:", relative_error)
    relative_error = torch.norm(y_diff_scaled).numpy()
    print("L2 relative error other calculation:", relative_error)
    with open(output_folder + '/err_rel_final.txt', 'w') as file:
        file.write(str(relative_error))

    print("Saved outputs in folder " + output_folder + ".\n\n\n")


"""
    Functions for comparing and plotting results of different methods/runs
"""

# Function to calculate moving average
def comp_moving_average(data, window_size):
    # Use a smaller window size for the initial moving average
    if window_size > len(data):
        window_size = len(data)
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return moving_avg

def plot_compare_errors(folderlabellist, fname_results, overalltitle=None, skipfactor=1, usebestloss=False, usetime=False, maxit=None, moving_average=False, moving_average_window_size=20):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for foldername, labelname in folderlabellist:
        fname = script_dir + '/' + foldername + '/err_rel_hist.dat'
        fname = foldername + '/err_rel_hist.dat'
        
        fname_time = script_dir + '/' + foldername + '/time_hist.dat'  
        fname_time = foldername + '/time_hist.dat' 
    
        steps = []
        train_times = []
        relerrs = []
        counterhelper = 0
        with open(fname, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                considerline = True
                if counterhelper % skipfactor != 0:
                    considerline = False
                if maxit is not None:
                    if int(columns[0]) > maxit:
                        considerline = False
                if considerline:
                    steps.append(int(columns[0]))
                    if not usebestloss:
                        relerrs.append(float(columns[1]))
                    else:
                        relerrs.append(float(columns[2]))
                    if usetime:
                        train_times.append(float(columns[3]))
                counterhelper += 1

        time_per1000it = 0.
        with open(fname_time, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                time_per1000it = float(columns[3])

        if moving_average:
            relerrs = comp_moving_average(relerrs, moving_average_window_size)
            moving_average_window_size_it = steps[0]*moving_average_window_size

        if usetime:
            steps = train_times

        fulllabel = labelname
        fulllabel += f" ({time_per1000it:.2f}s/1000it)"
        ax.plot(steps, relerrs, label=fulllabel)


    titlestring = 'Rel. errors '
    if usebestloss:
        titlestring = titlestring + '4 bestloss '
    titlestring = titlestring + '2 reference solution'
    if moving_average:
        titlestring = titlestring + f' (moving average, window {moving_average_window_size_it} it)'
    if overalltitle is not None:
        titlestring = titlestring + ' - ' + overalltitle
    ax.set_title(titlestring)

    if not usetime:
        ax.set_xlabel("# Iterations")
    else:
        ax.set_xlabel("Training time [s]")
    ax.set_ylabel("Rel. error")
    ax.set_yscale('log')
    ax.grid()
    ax.legend(loc = "upper right")
    ax.set_xlim(left=0, right=None)

    #fig.savefig(script_dir + '/' + fname_results + '.png')
    fig.savefig(fname_results + '.png')
    plt.close(fig)

def compare_accurcies_table(folderlabellister, fname_results, maxit=None, baseind=0, baseind_time=0, usebestloss=True, fix_time=True):
    """"
        Creates table of data of different methods in file
    """
    folderlabellist = folderlabellister.copy() #Just to be sure...

    # Get total time of base time simulation
    foldername, labelname = folderlabellist[baseind_time]
    fname = script_dir + '/' + foldername + '/err_rel_hist.dat'
    fname_time = script_dir + '/' + foldername + '/time_hist.dat'  
    fname = foldername + '/err_rel_hist.dat'
    fname_time = foldername + '/time_hist.dat'  
    steps = []
    train_times = []
    relerrs = []
    time_per1000it = 0.

    with open(fname, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            considerline = True
            if maxit is not None:
                if int(columns[0]) > maxit:
                    considerline = False
            if considerline:
                steps.append(int(columns[0]))
                if not usebestloss:
                    relerrs.append(float(columns[1]))
                else:
                    relerrs.append(float(columns[2]))

                train_times.append(float(columns[3]))

    base_time = train_times[-1]

    
    # Get accuracy of base simulation
    foldername, labelname = folderlabellist[baseind]
    fname = script_dir + '/' + foldername + '/err_rel_hist.dat'
    fname_time = script_dir + '/' + foldername + '/time_hist.dat'  
    fname = foldername + '/err_rel_hist.dat'
    fname_time = foldername + '/time_hist.dat'  
    steps = []
    train_times = []
    relerrs = []
    time_per1000it = 0.

    with open(fname, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            considerline = True
            if maxit is not None:
                if int(columns[0]) > maxit:
                    considerline = False
            if fix_time and float(columns[3]) > base_time:
                considerline = False
            if considerline:
                steps.append(int(columns[0]))
                if not usebestloss:
                    relerrs.append(float(columns[1]))
                else:
                    relerrs.append(float(columns[2]))

                train_times.append(float(columns[3]))

    with open(fname_time, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            time_per1000it = float(columns[3])

    # base_time = train_times[-1]
    base_acc = relerrs[-1]
    its = steps[-1]

    out_str = "# 1) Index | 2) Name | 3) Acc | 4) Rel. improvement [%] | 5) Training Time | 6) Time per 1000 it | 7) Iterations\n"
    out_str = out_str + "# First line contains base configuration\n#\n"

    out_str_base = f"{baseind}\t{labelname}\t{base_acc}\t{0.}\t{base_time}\t{time_per1000it}\t{its}\n"

    #Remove base configuration from list
    # folderlabellist.pop(baseind)

    #Iterate through remaining simulations
    for curr_index, follabname in enumerate(folderlabellist):
        foldername, labelname = follabname
        fname = script_dir + '/' + foldername + '/err_rel_hist.dat'
        fname_time = script_dir + '/' + foldername + '/time_hist.dat'  
        fname = foldername + '/err_rel_hist.dat'
        fname_time = foldername + '/time_hist.dat'  
        steps = []
        train_times = []
        relerrs = []
        time_per1000it = 0.


        with open(fname, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                considerline = True
                if maxit is not None:
                    if int(columns[0]) > maxit:
                        considerline = False
                if fix_time and float(columns[3]) > base_time:
                    considerline = False

                if considerline:
                    steps.append(int(columns[0]))
                    if not usebestloss:
                        relerrs.append(float(columns[1]))
                    else:
                        relerrs.append(float(columns[2]))

                    train_times.append(float(columns[3]))

        with open(fname_time, 'r') as file:
            for line in file:
                columns = line.strip().split('\t')
                time_per1000it = float(columns[3])

        curr_time = train_times[-1]
        curr_acc = relerrs[-1]
        curr_its = steps[-1]

        curr_rel_imp = 100.*(base_acc-curr_acc)/base_acc

        if curr_index != baseind:
            out_str = out_str + f"{curr_index}\t{labelname}\t{curr_acc}\t{curr_rel_imp}\t{curr_time}\t{time_per1000it}\t{curr_its}\n"
        else:
            out_str = out_str + out_str_base

    # with open(script_dir + '/' + fname_results + '.txt', "w") as text_file:
    with open(fname_results + '.txt', "w") as text_file:
        text_file.write(out_str)

# Creates "bar plot" of relative improvements of methods, based on tables 
def bars_compares(folderlabellist, fname_results, datafile="_a_table_fixedtime.txt", maxindex=None, methodnames=None, method_order=None, 
                  fig_size=(8,6), with_lines=False, negyscale=5., bar_width=.1, texts_fontsize=10):
    #Read in files
    cat_names = []
    methods_per_cat = []
    indeces = []
    accs = []
    ris = []
    for folder, labelname in folderlabellist:
        fname_data = folder + datafile 
        # 1) Index | 2) Name | 3) Acc | 4) Rel. improvement [%] | 5) Training Time | 6) Time per 1000 it | 7) Iterations"
        curr_indeces = []
        curr_names = []
        curr_accs = []
        curr_RI = []
        with open(fname_data, 'r') as file:
            for line in file:
                if not line.startswith('#'):
                    columns = line.strip().split('\t')
                    considerline = True
                    if maxindex is not None:
                        if int(columns[0]) > maxindex:
                            considerline = False
                    if considerline:
                        curr_indeces.append(int(columns[0]))
                        curr_names.append(columns[1])
                        curr_accs.append(float(columns[2]))
                        curr_RI.append(float(columns[3]))
        indeces.append(curr_indeces)
        accs.append(curr_accs)
        ris.append(curr_RI)
        cat_names.append(labelname)
        if methodnames is None:
            methods_per_cat = curr_names
        else:
            methods_per_cat = methodnames
    
    indeces = np.array(indeces)
    accs = np.array(accs)
    ris = np.array(ris)

    # Sort data and methods for each category
    sorted_indices = np.argsort(np.argsort(-ris, axis=1), axis=1)

    x = np.arange(len(cat_names))

    # Define a color palette
    colors = plt.cm.tab10(np.arange(len(methods_per_cat)))  # Use a colormap

    plt.figure(figsize=fig_size) #default size is (8,6)

    lwidth_lines = .7
    if with_lines:
        plt.axhline(y=50, color='gray', linestyle='--', linewidth=lwidth_lines, zorder=2)
        plt.axhline(y=-100, color='gray', linestyle='--', linewidth=lwidth_lines, zorder=2)
        plt.axhline(y=90, color='gray', linestyle='--', linewidth=lwidth_lines, zorder=2)
        plt.axhline(y=-900, color='gray', linestyle='--', linewidth=lwidth_lines, zorder=2)

        plt.text(x=-.22, y=51, s=r' 0.5 $\times$ error', color='gray', fontsize=9, horizontalalignment='left')
        plt.text(x=-.22 , y=-90, s=r' 2 $\times$ error', color='gray', fontsize=9, horizontalalignment='left')
        plt.text(x=-.22 , y=91, s=r' 0.1 $\times$ error', color='gray', fontsize=9, horizontalalignment='left')
        plt.text(x=-.22 , y=-900, s=r' 10 $\times$ error', color='gray', fontsize=9, horizontalalignment='left')

    plt.axhline(y=0, color='black', linewidth=lwidth_lines,  zorder=2)



    # Create bars
    hatch_list = [None, '/////', 'xxxxx', '.....']
    bars_per_supmethod = len(methods_per_cat)//3
    for ime in range(len(methods_per_cat)):
        if method_order is None:
            i = ime
        else:
            i = method_order[ime]
        plt.bar(x + sorted_indices[x,i] * bar_width, ris[:, i], label=methods_per_cat[i], 
                edgecolor='black', width=bar_width, color=colors[i%3], zorder=3, hatch=hatch_list[ime%bars_per_supmethod])

    # Customizing the plot
    plt.ylabel('Relative improvement [%]', fontsize=texts_fontsize)
    plt.ylim(-2.*1e5, 100)

    # Adjust y-axis scaling
    def yscale_forward(x):
        return np.where(x>=0, x, -negyscale*np.log(1.+np.abs(x)))
    def yscale_backward(x):
        return np.where(x>=0, x, 1.-np.exp(-x/negyscale))
    plt.yscale("function", functions=[yscale_forward, yscale_backward])

    plt.yticks([-100000,-10000,-1000,-100,-10,20, 40, 60, 80, 100], fontsize=texts_fontsize)

    plt.xticks(x + bar_width * (len(methods_per_cat) - 1) / 2, cat_names, fontsize=texts_fontsize)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=texts_fontsize)

    # Save the plot to a file
    plt.savefig(fname_results + ".png", dpi=300, bbox_inches='tight')


def plot_all_compares(folderlabellist, fname_results, skipfactor, overalltitle="", baseind_table=0):
    """
        Call above functions to create different plots and tables to compare methods
    """
    for usetime in [True,False]: # Create plots both with time and number of training iterations on x-axis
        fnameadd = ""
        if usetime:
            fnameadd += "_time"
        plot_compare_errors(folderlabellist, fname_results + fnameadd, overalltitle=overalltitle, usetime=usetime)
        plot_compare_errors(folderlabellist, fname_results + fnameadd + "_sparsed", overalltitle=overalltitle, skipfactor=skipfactor, usetime=usetime)
        plot_compare_errors(folderlabellist, fname_results + fnameadd + "_bestloss", overalltitle=overalltitle, usebestloss=True, usetime=usetime)

    compare_accurcies_table(folderlabellist, fname_results + "_a_table_fixedtime", baseind=baseind_table)
    compare_accurcies_table(folderlabellist, fname_results + "_a_table_sameits", fix_time=False, baseind=baseind_table)



"""
    Specify general methods and hyperparameters 
"""

Nr_train_init = 500
Nr_train_boundary = 1000
Nr_train_domain = 20000
Nr_test = None

fourier_size = 20
fourier_size_larger = 50
fourier_sigma = 20.

shortnames = ["No", "BClit", "BCours", 
              f"No_Fourier{fourier_size}", f"BClit_Fourier{fourier_size}", f"BCours_Fourier{fourier_size}",
              f"No_Fourier{fourier_size_larger}", f"BClit_Fourier{fourier_size_larger}", f"BCours_Fourier{fourier_size_larger}"]
plotnames = ["Vanilla", "HC literature", "HC via Fourier 1", 
             f"Vanilla + Fourier {fourier_size}", f"HC literature + Fourier {fourier_size}", f"HC via Fourier {fourier_size}",
             f"Vanilla + Fourier {fourier_size_larger}", f"HC literature + Fourier {fourier_size_larger}", f"HC via Fourier {fourier_size_larger}"]

fourier_embeddings = [] #Initialise different Fourier embedding strategies
for ihc in range(len(shortnames)):
    fourier_embedding = None
    if ihc == 2:
        fourier_embedding = FourierEmbedding(1, fourier_sigma, only_ints=True, only_cos=True, nec_freqs=[1],scale01=False) #Use only frequency 1
    elif ihc == 3 or ihc == 4:
        fourier_embedding = FourierEmbedding(fourier_size, fourier_sigma, nec_freqs=[1.], scale01=False, alternating_cossin=True) #Use random frequencies
    elif ihc == 5:
        fourier_embedding = FourierEmbedding(fourier_size, fourier_sigma, only_ints=True, only_cos=True, nec_freqs=[1.], scale01=False) #Use random integer frequencies and cos
    elif ihc == 6 or ihc == 7:
        fourier_embedding = FourierEmbedding(fourier_size_larger, fourier_sigma, nec_freqs=[1.], scale01=False, alternating_cossin=True) #Use random frequencies
    elif ihc == 8:
        fourier_embedding = FourierEmbedding(fourier_size_larger, fourier_sigma, only_ints=True, only_cos=True, nec_freqs=[1.], scale01=False) #Use random integer frequencies and cos
    fourier_embeddings.append(fourier_embedding)

#Specify hyperparameters that are used
lrs_depths_widths = [(1.e-4, 3, 100)] #List of learning rates, NN-depths, and NN-widths 
loss_weights = [None]*len(shortnames) #List of loss weights 
trainits_list = [1_000_000]*len(shortnames) #List of training iteration numbers 

#Option for debugging
# trainits_list = [1_000]*len(shortnames) #List of training iteration numbers 




"""
    Low Frequency setting
"""

initcond = InitCos(2.)
D = 1./(2.*2.*np.pi*np.pi)
plottitle = "Low frequency"
fname_prefix = "1_LF/dat_"

for ihyp in range(len(lrs_depths_widths)): #Iterate over different hyperparameter settings (learning rate, MLP architecture)
    foldername = fname_prefix + str(ihyp)
    fnames = []
    outlist = []
    lr, Nr_hidden_layers, Nr_neurons_per_layers = lrs_depths_widths[ihyp]

    for ihc in range(len(shortnames)):
        new_fname = foldername + str(ihc) + "_" + shortnames[ihc]
        fnames.append(new_fname)

        hc_Neumann_lit = ("BClit" in shortnames[ihc])

        fourier_embedding = fourier_embeddings[ihc]
    
        train_plot(new_fname, initcond, D=D, hc_Neumann_lit=hc_Neumann_lit,
                   train_its=trainits_list[ihc], lr=lr, Nr_hidden_layers = Nr_hidden_layers, Nr_neurons_per_layers = Nr_neurons_per_layers,
                   fourier_embedding=fourier_embedding,
                   Nr_train_boundary=Nr_train_boundary, Nr_train_init=Nr_train_init, Nr_train_domain=Nr_train_domain, Nr_test=Nr_test, 
                   loss_weights=loss_weights[ihc])
        
        outlist.append((new_fname, plotnames[ihc]))

    plot_all_compares(outlist, foldername, 10, overalltitle=plottitle)
        



"""
    High Frequency solution
"""
initcond = InitCos(50.)
D = 1./(50.*50.*np.pi*np.pi)
plottitle = "High frequency"
fname_prefix = "2_HF/dat_"

for ihyp in range(len(lrs_depths_widths)): #Iterate over different hyperparameter settings (learning rate, MLP architecture)
    foldername = fname_prefix + str(ihyp)
    fnames = []
    outlist = []
    lr, Nr_hidden_layers, Nr_neurons_per_layers = lrs_depths_widths[ihyp]

    for ihc in range(len(shortnames)):
        new_fname = foldername + str(ihc) + "_" + shortnames[ihc]
        fnames.append(new_fname)

        hc_Neumann_lit = ("BClit" in shortnames[ihc])

        fourier_embedding = fourier_embeddings[ihc]
    
        train_plot(new_fname, initcond, D=D, hc_Neumann_lit=hc_Neumann_lit,
                   train_its=trainits_list[ihc], lr=lr, Nr_hidden_layers = Nr_hidden_layers, Nr_neurons_per_layers = Nr_neurons_per_layers,
                   fourier_embedding=fourier_embedding,
                   Nr_train_boundary=Nr_train_boundary, Nr_train_init=Nr_train_init, Nr_train_domain=Nr_train_domain, Nr_test=Nr_test, 
                   loss_weights=loss_weights[ihc])
        
        outlist.append((new_fname, plotnames[ihc]))

    plot_all_compares(outlist, foldername, 10, overalltitle=plottitle)
        



"""
    Multi Frequency / Multiscale solution
"""
initcond = InitMultCos([2.,50.], [1.,.1])
D = 1./(50.*50.*np.pi*np.pi)
plottitle = "Multiscale"
fname_prefix = "3_MS/dat_"

for ihyp in range(len(lrs_depths_widths)): #Iterate over different hyperparameter settings (learning rate, MLP architecture)
    foldername = fname_prefix + str(ihyp)
    fnames = []
    outlist = []
    lr, Nr_hidden_layers, Nr_neurons_per_layers = lrs_depths_widths[ihyp]

    for ihc in range(len(shortnames)):
        new_fname = foldername + str(ihc) + "_" + shortnames[ihc]
        fnames.append(new_fname)

        hc_Neumann_lit = ("BClit" in shortnames[ihc])

        fourier_embedding = fourier_embeddings[ihc]
    
        train_plot(new_fname, initcond, D=D, hc_Neumann_lit=hc_Neumann_lit,
                   train_its=trainits_list[ihc], lr=lr, Nr_hidden_layers = Nr_hidden_layers, Nr_neurons_per_layers = Nr_neurons_per_layers,
                   fourier_embedding=fourier_embedding,
                   Nr_train_boundary=Nr_train_boundary, Nr_train_init=Nr_train_init, Nr_train_domain=Nr_train_domain, Nr_test=Nr_test, 
                   loss_weights=loss_weights[ihc])
        
        outlist.append((new_fname, plotnames[ihc]))

    plot_all_compares(outlist, foldername, 10, overalltitle=plottitle)
        


"""
    Polynomial 3rd degree
"""

initcond = InitPolynom([0.,0.,3.,-2.])
D = 1./(np.pi*np.pi)
plottitle = "Poly3"
fname_prefix = "4_P3/dat_"

for ihyp in range(len(lrs_depths_widths)): #Iterate over different hyperparameter settings (learning rate, MLP architecture)
    foldername = fname_prefix + str(ihyp)
    fnames = []
    outlist = []
    lr, Nr_hidden_layers, Nr_neurons_per_layers = lrs_depths_widths[ihyp]

    for ihc in range(len(shortnames)):
        new_fname = foldername + str(ihc) + "_" + shortnames[ihc]
        fnames.append(new_fname)

        hc_Neumann_lit = ("BClit" in shortnames[ihc])

        fourier_embedding = fourier_embeddings[ihc]
    
        train_plot(new_fname, initcond, D=D, hc_Neumann_lit=hc_Neumann_lit,
                   train_its=trainits_list[ihc], lr=lr, Nr_hidden_layers = Nr_hidden_layers, Nr_neurons_per_layers = Nr_neurons_per_layers,
                   fourier_embedding=fourier_embedding,
                   Nr_train_boundary=Nr_train_boundary, Nr_train_init=Nr_train_init, Nr_train_domain=Nr_train_domain, Nr_test=Nr_test, 
                   loss_weights=loss_weights[ihc])
        
        outlist.append((new_fname, plotnames[ihc]))

    plot_all_compares(outlist, foldername, 10, overalltitle=plottitle)
        


"""
    Polynomial 4th degree
"""

initcond = InitPolynom([0.,0.,16.,-32.,16.])
D = 1./(np.pi*np.pi)
plottitle = "Poly4"
fname_prefix = "5_P4/dat_"


for ihyp in range(len(lrs_depths_widths)): #Iterate over different hyperparameter settings (learning rate, MLP architecture)
    foldername = fname_prefix + str(ihyp)
    fnames = []
    outlist = []
    lr, Nr_hidden_layers, Nr_neurons_per_layers = lrs_depths_widths[ihyp]

    for ihc in range(len(shortnames)):
        new_fname = foldername + str(ihc) + "_" + shortnames[ihc]
        fnames.append(new_fname)

        hc_Neumann_lit = ("BClit" in shortnames[ihc])

        fourier_embedding = fourier_embeddings[ihc]
    
        train_plot(new_fname, initcond, D=D, hc_Neumann_lit=hc_Neumann_lit,
                   train_its=trainits_list[ihc], lr=lr, Nr_hidden_layers = Nr_hidden_layers, Nr_neurons_per_layers = Nr_neurons_per_layers,
                   fourier_embedding=fourier_embedding,
                   Nr_train_boundary=Nr_train_boundary, Nr_train_init=Nr_train_init, Nr_train_domain=Nr_train_domain, Nr_test=Nr_test, 
                   loss_weights=loss_weights[ihc])
        
        outlist.append((new_fname, plotnames[ihc]))

    plot_all_compares(outlist, foldername, 10, overalltitle=plottitle)
        




"""
    Compare above by creating bar plots as in paper
"""


fig_size=(14,4)
bar_width=.08
texts_fontsize=15


bars_folderlabellist = [("1_LF/", "Low frequency\n(Reference: Vanilla)"),
                        ("2_HF/", "High frequency\n(Reference: Vanilla \nw/ Fourier 50)"),
                        ("3_MS/", "Multiscale\n(Reference: Vanilla \nw/ Fourier 50)"),
                        ("4_P3/", "Polynom 3rd order\n(Reference: Vanilla)"),
                        ("5_P4/", "Polynom 4th order\n(Reference: Vanilla)")]

methodnames = ["Vanilla PINN", "Existing HC method", r"HC via 1 Fourier frequency ${\bf (Ours)}$",
               "Vanilla PINN\nwith random Fourier embedding (size 20)", "Existing HC method\nwith random Fourier embedding (size 20)", r"HC via 20 random Fourier frequencies ${\bf (Ours)}$",
               "Vanilla PINN\nwith random Fourier embedding (size 50)", "Existing HC method\nwith random Fourier embedding (size 50)", r"HC via 50 random Fourier frequencies ${\bf (Ours)}$"]
method_order = [0,3,6,1,4,7,2,5,8]




bars_compares(bars_folderlabellist, "relative_improvements", datafile="dat_0_a_table_fixedtime.txt", maxindex=None, methodnames=methodnames, method_order=method_order, 
              fig_size=fig_size, bar_width=bar_width, texts_fontsize=texts_fontsize)

bars_compares(bars_folderlabellist, "relative_improvements_sameiterations", datafile="dat_0_a_table_sameits.txt", maxindex=None, methodnames=methodnames, method_order=method_order, 
              fig_size=fig_size, bar_width=bar_width, texts_fontsize=texts_fontsize)

