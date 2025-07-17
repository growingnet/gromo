import time
import torch
import torch.nn as nn
import torch.optim as optim

# Example with default parameters.
param_default = {
    # General parameters
    'reg': 1e-6,  # Regularization term for numerical stability when computing pseudo-inverse
    'batch_size': 64,  # Batch size for training
    'warning': True,  # Whether to print warnings when the training loss increases
    'analyse': False,  # Whether to compute the training time and losses at each epoch
    'delta': 'one',  # Method to compute the ponderation, either 'one' or 'delta'
    'z': 'star',  # Goal for the z optimization, either 'star' or 'mid'
    'method': 'Omega-A',  # Default optimization method, or : 'omega-SGD'
    
    # if method is Omega-A
    'nb_iterations_Om_A': 2,  # number of Omega <-> A optimization steps
    
    # if method is omega-SGD
    'lr': 5e-3,  # Learning rate for SGD (Adam optimizer
    'epochs': 20,  # Number of epochs for omega-SGD method
    'omega_step': 10,  # Number of epochs between Omega optimizations for omega-SGD method
    'optimizer': 'Adam',  # Optimizer for the omega-SGD method, either 'SGD' or 'Adam'
    
    # if activation is bijective (ie LeakyReLU)
    'negative_slope': 0.01,  # Default negative slope for LeakyReLU
    
    # if activation is not bijective (ie Softplus) : need to solve NNLSQ
    'z_optimization_method': 'PGD',  # Method to optimize z, either 'plsq', or 'PGD' ne need to specify if the activation function is LeakyReLU
    'clip': 0.05,  # Clipping for z when computing the inverse of the softplus function
    'max_iter': 15,  # for PGD or lsq_linear
}

def new_FCL_heuristic_initialization(train_loader, nb_new_neuron, activation_function, param = {}):
    """
    Perform in place optimization for the weight matrix A and Omega corresponding to the problem
    min_{A, Omega} E(||Omega* F(Ax) - y||^2)
    where F is the activation function.
    
    Args:
        dataloader: DataLoader object containing the data 
        A: First layer weight matrix.
        Omega: Second layer weight matrix.
        activation_function: Function between the two layers, either 'leakyrelu' or 'softplus'
        param: Dictionary of parameters for optimization.
    Returns:
        model: An instance of MLPHeuristicTraining with the optimized weights.
    """
    input_size = train_loader.dataset.tensors[0].shape[1]  # Input size from the training data
    hidden_size = nb_new_neuron  # Number of new neurons
    output_size = train_loader.dataset.tensors[1].shape[1]  # Output size from the training data
    param['batch_size'] = train_loader.batch_size  # Use the batch size from the DataLoader
    
    # default parameters for two methods with LeakyReLU and Softplus
    if activation_function.name == 'leakyrelu':
        if 'clip' not in param:
            param['clip'] = 0.0
        param['negative_slope'] = activation_function.negative_slope  # Use the negative slope from the LeakyReLU activation
    
    elif activation_function.name == 'softplus':
        if 'z_optimization_method' not in param:
            param['z_optimization_method'] = 'PGD'
        if 'z' not in param:
            param['z'] = 'star'
        if 'delta' not in param:
            param['delta'] = 'delta'
    
    model = MLPHeuristicTraining(input_size, hidden_size, output_size, param = param, activation = activation_function)
    model.train_model(train_loader, train_loader)  # Using train_loader for both training and testing
    
    return model
    
    

class MLPHeuristicTraining(nn.Module):
    """
    Class defining the network and the training methods.
    """
    def __init__(self, input_size, hidden_size, output_size, method="Omega-A", param = {}, activation = nn.Softplus):
        super(MLPHeuristicTraining, self).__init__()
        
        self.n = input_size
        self.k = hidden_size
        self.m = output_size
        
        # The Architecture of the new layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = activation()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Choice of the optimization method
        self.method = param['method'] if 'method' in param else "Omega-A"  # Default method is Omega-A
        assert(method in ["Omega-A"]), f"Unknown optimization method: {method}. Choose from 'Omega-A' or 'omega-SGD'."
        
        
        self.A = self.linear1.weight  # First layer weights A
        self.Omega = self.linear2.weight  # Second layer weights Ω
        
        # the activation function
        self.sigma = self.activation.__class__.__name__.lower()  # Get the activation function name in lowercase
        assert(self.sigma in ['softplus', 'leakyrelu']), "Activation function must be Softplus or LeakyReLU."
        self.negative_slope = param['negative_slope'] if 'negative_slope' in param else 0.01 # Default negative slope for LeakyReLU
        
        # General parameters
        self.reg = param['reg'] if 'reg' in param else 1e-6 # Regularization term for numerical stability when computing pseudo-inverse
        self.warning = param['warning'] if 'warning' in param else True # Whether to print warnings when the training loss increases
        self.batch_size = param['batch_size'] if 'batch_size' in param else 64 # Batch size for training
        self.analyse = param['analyse'] if 'analyse' in param else False # Whether to compute the training time and losses at each epoch
        
        # Parameters for the A optimization
        self.z = param['z'] if 'z' in param else 'star' # Goal for the z optimization, either 'star' or 'mid'
        assert(self.z in ['star', 'mid']), "z must be 'star' or 'mid'. 'star' is the optimal solution, 'mid' is the middle point between the target and the student output."
        self.z_optimization_method = param['z_optimization_method'] if 'z_optimization_method' in param else 'PGD' # Method to optimize z, either 'plsq', or 'PGD'
        assert(self.z_optimization_method in ['plsq', 'PGD', 'leakyrelu']), "z_optimization_method must be 'plsq', 'PGD' or 'leakyrelu'."
        
        # Parameters for Adam optimizer for omega-SGD method
        self.lr = param['lr'] if 'lr' in param else self.get_lr(input_size,self.batch_size) # Learning rate for SGD (Adam optimizer) with omega-SGD method
        self.epochs = param['epochs'] if 'epochs' in param else 20 # Number of epochs for omega-SGD method
        self.omega_step = param['omega_step'] if 'omega_step' in param else 10 # Number of epochs between Omega optimizations for omega-SGD method
        self.optimizer = param['optimizer'] if 'optimizer' in param else 'Adam' # Optimizer for the omega-SGD method, either 'SGD' or 'Adam'
        assert(self.optimizer in ['SGD', 'Adam']), "Optimizer must be 'SGD' or 'Adam'."
        
        self.clip = param['clip'] if 'clip' in param else 0.05 # Clipping for z when computing the inverse of the softplus fonction
        
        self.delta = param['delta'] if 'delta' in param else 'one' # Method to compute the ponderation, either 'one' or 'delta'
        
        self.max_iter = param['max_iter'] if 'max_iter' in param else 15 # for PGD or lsq_linear
        
        self.nb_iterations_Om_A = param['nb_iterations_Om_A'] if 'nb_iterations_Om_A' in param else 2 # number of Omega <-> A optimization steps 
        
        
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

    def train_model(self, train_loader, test_loader):
        """
        Train the student network using a heuristic method.
        """
        assert(self.n == train_loader.dataset.tensors[0].shape[1]), "Input size of the student must match the input size of the training data."
        assert(self.n == test_loader.dataset.tensors[0].shape[1]), "Input size of the student must match the input size of the testing data."
        
        train_losses = [self.evaluate_loss(train_loader)] 
        test_losses = [self.evaluate_loss(test_loader)] # Start by computing the accuracy 
        train_time = [0.0]
        
        # Randomly initialize the first layer weights and biases and closed form optimal solution for the second layer weights Ω
        # By default PyTorch initializes weights using kaiming uniform distribution for Linear layers, bias are initialized to 0
  
        if self.method == "omega-SGD":
            self.train_omega_SGD(train_loader, test_loader, train_losses, test_losses, train_time)
        elif self.method == "Omega-A":
            self.train_Omega_A(train_loader, test_loader, train_losses, test_losses, train_time)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from 'omega-SGD' or 'Omega-A'.")
    
        return train_losses, test_losses, train_time
        
    def train_omega_SGD(self, train_loader, test_loader, train_losses, test_losses, train_time):
        """ Train the student network using the omega-SGD method.
            This method alternates between omega optimization and SGD training steps.
            Parameters:
                train_loader: DataLoader for the training data
                test_loader: DataLoader for the testing data
                train_losses: List to store training losses
                test_losses: List to store testing losses
                train_time: List to store training times
            Returns:
                train_losses, test_losses, train_time: Updated lists with losses and times.
        """
        if self.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}. Choose 'SGD' or 'Adam'.")
        
        criterion = nn.MSELoss()
        
        for epoch in range(self.param['epochs']):
            t1 = time.time()
            epoch_loss = 0.0
            if epoch % self.param['omega_step'] == 0:
                self.Omega_opt(train_loader)
                train_losses.append(self.evaluate_loss(train_loader))
                test_losses.append(self.evaluate_loss(test_loader))
                train_time.append(time.time() - t1)
                t1 = time.time()
                
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_loader))
            train_time.append(time.time() - t1)
            test_losses.append(self.evaluate_loss(test_loader))

    def train_Omega_A(self, train_loader, test_loader, train_losses, test_losses, train_time):
        """
        Train the student network using the Omega-A method.
        This method alternates between Omega optimization and A optimization.
        Parameters:
            train_loader: DataLoader for the training data
            test_loader: DataLoader for the testing data
            train_losses: List to store training losses
            test_losses: List to store testing losses
            train_time: List to store training times
            Returns:
        train_losses, test_losses, train_time: Updated lists with losses and times.
        """
        for i in range(self.nb_iterations_Om_A):
            t1 = time.time()
            self.Omega_opt(train_loader)
            train_time.append(time.time() - t1)
            train_losses.append(self.evaluate_loss(train_loader))
            test_losses.append(self.evaluate_loss(test_loader))
            if train_losses[-1] > train_losses[-2] and self.warning: 
                print("Warning: Training loss increased after Omega optimization")
            t1 = time.time()
            self.A_opt(train_loader)
            train_time.append(time.time() - t1)
            train_losses.append(self.evaluate_loss(train_loader))
            test_losses.append(self.evaluate_loss(test_loader))
            if train_losses[-1] > train_losses[-2]*1.1 and self.warning:
                print("Warning: Training loss increased after A optimization : ", train_losses[-1], " > ", train_losses[-2])
                print("Max abs coef of A: ", torch.max(torch.abs(self.A)).item())
                print("Max abs coef of Omega: ", torch.max(torch.abs(self.Omega)).item())
                print("Norm of A: ", torch.norm(self.A, p='fro').item())
                print("Norm of Omega: ", torch.norm(self.Omega, p='fro').item())
        t1 = time.time()
        self.Omega_opt(train_loader)
        train_time.append(time.time() - t1)
        train_losses.append(self.evaluate_loss(train_loader))
        test_losses.append(self.evaluate_loss(test_loader))
        if train_losses[-1] > train_losses[-2] and self.warning: 
            print("Warning: Training loss increased after Omega optimization")

    @torch.no_grad()
    def Omega_opt(self, dataloader):
        """
        Closed-form solution for the second layer weights Ω using the expectation formulas:
            y = y - mu_y
            F = F - mu_F
            Ω* = E[y F(Ax)^T] @ (E[F(Ax) F(Ax)^T])^†
            bias_Omega = mu_y - Ω* @ mu_F
            Parameters:
                dataloader: DataLoader for the training data
            Returns:
                None
            Updates the model weights in place.
        """
        device = next(self.parameters()).device

        # Accumulate statistics for expectations
        mu_y_sum = torch.zeros(self.m, device=device)   # (m,)
        mu_F_sum = torch.zeros(self.k, device=device)   # (k,)
        num_samples = 0
        F_batches = []
        y_batches = [] # Compulsory as the dataloader might not return y in the same during a second iteration of the loop
        
        # Compute expectations over the dataset
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            F_batch = self.activation(self.linear1(X_batch))  # shape: (B, k)
            
            F_batches.append(F_batch)
            y_batches.append(y_batch)
            
            num_samples += X_batch.size(0)
            mu_y_sum += y_batch.sum(dim=0)  # shape: (m,)
            mu_F_sum += F_batch.sum(dim=0)  # shape: (k,)
            
        mu_y = mu_y_sum / num_samples  # shape: (m,)
        mu_F = mu_F_sum / num_samples  # shape: (k,)

        FFT_sum = torch.zeros(self.k, self.k, device=device)  # shape: (k, k)
        yFT_sum = torch.zeros(self.m, self.k, device=device)  # shape: (m, k)

        for F_batch, y_batch in zip(F_batches, y_batches):
            F_centered = F_batch - mu_F           # (B, k)
            y_centered = y_batch - mu_y           # (B, m)

            FFT_sum += F_centered.T @ F_centered  # (k, k)
            yFT_sum += y_centered.T @ F_centered  # (m, k)
                    
        # Estimate expectations
        E_FF_T = FFT_sum  / num_samples       # shape: (k, k)
        E_yF_T = yFT_sum / num_samples        # shape: (m, k)
        
        # Compute pseudo-inverse of E[F F^T], Using torch.linalg.pinv to run on GPU
        E_FF_T = E_FF_T + self.reg * torch.eye(E_FF_T.shape[0], device=E_FF_T.device) # for numerical stability
        E_FF_T_pinv = torch.linalg.inv(E_FF_T)
        Omega_star = E_yF_T @ E_FF_T_pinv     # (m, k)
        
        # Update model weights
        self.linear2.weight.copy_(Omega_star)
        self.linear2.bias.copy_(mu_y - Omega_star @ mu_F)

    @torch.no_grad()
    def A_opt(self, dataloader,verbose = False):
        """
        Closed-form solution for the first Layer weights A using the expectation formulas:
            A* = E[z_inv x^T] @ (E[x x^T] + reg * I_n)^†
        Parameters:
            dataloader: DataLoader for the training data
            verbose: Whether to print additional information
        Returns:
            None
        Updates the model weights in place.
        """

        device = next(self.parameters()).device
        precompute = {}
        
        U, S, V = torch.linalg.svd(self.Omega, full_matrices=False) # compute the SVD of Omega, shape for U: (m, m), S: (m,), V: (k, k)
        # cut the eigenvalues greater than 0.05 (also tried Tikhonov regularization)
        S_inv_reg = torch.where(S > 0.05, 1/S, 0)  # shape: (m,)
        omega_inv = V.T @ torch.diag(S_inv_reg) @ U.T  # shape: (k, m)
        precompute = {'omega_inv': omega_inv}
            
        if self.z_optimization_method == 'PGD':
            L = 1/torch.norm(S **2,p='fro') # inverse of the L2 norm of Omega.T @ Omega, base step for Fast Projected Gradient Descent
            Q = self.Omega.T @ self.Omega  # shape: (k, k)
            assert(Q.shape[0] == Q.shape[1] == self.k)
            theta_1 = torch.eye(self.k, device=device) - Q*L
            precompute |= {'L': L, 'theta_1': theta_1, 'Q': Q}
        
        if verbose:
            print('norm of Omega inverse: ', torch.norm(precompute['omega_inv'], p='fro').item())
        
        # Accumulate statistics for expectations
        mu_x_pond_sum = torch.zeros(self.n, device=device)   # (n,)
        mu_z_inv_pond_sum = torch.zeros(self.k, device=device)  # (k,)
        z_inv_batches = []
        X_batches = [] # Compulsory as the dataloader might not return X in the same during a second iteration of the loop
        ponderations = []
        total_mass = 0  # Total mass for normalization
        
        # Compute expectations over the dataset
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            z_batch = self.compute_z_batch(X_batch, y_batch, precompute)  # shape: (B, k)
            z_inv_batch = self.inverse_activation(z_batch)  # shape: (B, k)
            ponderation = self.ponderation(z_inv_batch, z_batch,X_batch)  # shape (B,)
            
            ponderations.append(ponderation)
            X_batches.append(X_batch)
            z_inv_batches.append(z_inv_batch)
            
            total_mass += ponderation.sum()  # Total mass for normalization
            mu_x_pond_sum += (X_batch * ponderation.unsqueeze(1)).sum(dim=0)  # shape: (n,)
            mu_z_inv_pond_sum += (z_inv_batch * ponderation.unsqueeze(1)).sum(dim=0)  # shape: (k,)
            
        mu_x_pond = mu_x_pond_sum / total_mass  # shape: (n,)
        mu_z_inv_pond = mu_z_inv_pond_sum / total_mass  # shape: (k,)
        
        xxT_sum = torch.zeros(self.n, self.n, device=device)  # shape: (n, n)
        xz_sum = torch.zeros(self.k, self.n, device=device)  # shape: (k, n)
        
        for Z_inv_batch, X_batch, ponderation in zip(z_inv_batches, X_batches, ponderations):
            x_centered = X_batch - mu_x_pond         # (B, n)
            z_inv_centered = Z_inv_batch - mu_z_inv_pond  # (B, k)

            xxT_sum += (x_centered*ponderation.unsqueeze(1)).T @ x_centered  # (n, B) @ (B, n) = (n, n)
            xz_sum += (z_inv_centered*ponderation.unsqueeze(1)).T @ x_centered  # (k, B) @ (B, n) = (k, n)
            
        # Estimate expectations
        E_xxT = xxT_sum / total_mass  # shape: (n, n)
        E_xzT = xz_sum / total_mass  # shape: (k, n)
        
        # Compute pseudo-inverse of E[xx^T], Using torch.linalg.pinv to run on GPU
        reg_E_xxT = E_xxT + self.reg* torch.eye(E_xxT.shape[0], device=E_xxT.device) # for numerical stability
        E_xxT_pinv = torch.linalg.inv(reg_E_xxT) 
        A_star = E_xzT @ E_xxT_pinv  # (k, n)
        
        # Update model weights
        self.linear1.weight.copy_(A_star)
        self.linear1.bias.copy_(mu_z_inv_pond - A_star @ mu_x_pond)
    
        if verbose:
            print('norm of E_xxT: ', torch.norm(E_xxT, p='fro').item())
            print('norm of E_xzT: ', torch.norm(E_xzT, p='fro').item())
            print('ponderation moyenne: ', torch.mean(torch.cat(ponderations)).item(), end=' ')
            print('ponderation std: ', torch.std(torch.cat(ponderations)).item(),end=' ')
            print('ponderation min: ', torch.min(torch.cat(ponderations)).item(), end=' ')
            print('ponderation max: ', torch.max(torch.cat(ponderations)).item())
    
    @torch.no_grad()
    def compute_z_batch(self, x, y, precompute):
        """
        Compute a z_batch using the given method to approximate a y_goal 
        Parameters:
            omega: Second layer weights; shape (m, k)
            omega_inv: Inverse of the second layer weights; shape (m, k) Warning : is None if z_optimization_method != 'plsq'
            x: Input data; shape (B, n)
            y: Target data; shape (B, m)
        Returns:
            z_batch: Computed z batch; shape (B, k)
        """
        if self.z == 'star':
            y_g = y - self.linear2.bias
        elif self.z == 'mid': 
            y_g = 1/2 * (y + self(x)) - self.linear2.bias 
        
        if self.z_optimization_method in ['plsq', 'leakyrelu']:
            z_batch = y_g @ precompute['omega_inv'].T  # shape: (B, k)
        elif self.z_optimization_method == 'PGD':
            z_batch = self.PGD(self.Omega, y_g, precompute['theta_1'], precompute['L'], self.max_iter, precompute['omega_inv'], self.clip)
        else:
            raise ValueError(f"Unknown z_optimization_method: {self.z_optimization_method}. Choose from 'plsq', 'PGD' or 'leakyrelu'.")
        
        if self.sigma == 'softplus':
            z_batch =  torch.clamp(z_batch, min=0.01)  # Ensure non-negativity for softplus
            
        if self.delta == 'one':
            z_batch = torch.clamp(z_batch, min=self.clip)
        
        return z_batch  # Leaky ReLU does not require non-negativity
  
    @torch.no_grad()
    def inverse_activation(self, z):
        """
        Compute the inverse of the activation function applied to z.
        Parameters:
            z: Input tensor; shape (B, k)
        Returns:
            Inverse of the activation function applied to z; shape (B, k)
        """
        if self.sigma == 'softplus':
            return self.inv_softplus(z)
        elif self.sigma == 'leakyrelu':
            return self.inv_leaky_relu(z, negative_slope=self.negative_slope)
        else:
            raise ValueError(f"Unknown activation function: {self.sigma}. Choose 'softplus' or 'leakyrelu'.")
    
    @staticmethod
    def PGD(Omega, y, theta_1, L, max_iter, omega_inv, clip):
        """
        Accelerated Projected Gradient Descent (PGD) with Nesterov acceleration and projection (clipping).

        Solves:       min_z ||Omega z - y||^2  subject to z >= clip

        Parameters:
        -----------
        Omega : torch.Tensor of shape (k, m)
            Dictionary or linear operator for the second layer.
        y : torch.Tensor of shape (B, m)
            Target signal (typically a batch of observations).
        theta_1 : torch.Tensor of shape (k, k)
            Precomputed operator: I_k - Q * L, with Q = Omega @ Omega.T and L = 1 / ||Omega^T Omega||_2
        L : float
            Step size (usually L = 1 / spectral_norm(Omega.T @ Omega))
        max_iter : int
            Maximum number of iterations for PGD.
        omega_inv : torch.Tensor of shape (16, 4)
            Regularized pseudo-inverse of Omega, used for initialization.
        clip : float
            Lower bound for projection; ensures z >= clip element-wise.

        Returns:
        --------
        z_pgd : torch.Tensor of shape (B, k)
            Solution estimate for each sample in the batch.
        """
        tol = 1e-1  # Tolerance for convergence
        
        # Precompute constant term in the update: L * Omega.T @ y
        theta_2 = y @ (Omega * L)  # shape: (B, k)

        # Initialize acceleration parameter
        alpha = 0.5

        # Initialization: warm start using pseudo-inverse, with projection
        u = torch.clamp(y @ omega_inv.T, min=clip)  # shape: (B, k)
        v = u.clone()  # auxiliary variable for Nesterov momentum

        # Main loop: Nesterov accelerated PGD
        for _ in range(max_iter):
            # Gradient update + projection (ReLU-like via clamp)
            u_new = torch.clamp(v @ theta_1.T + theta_2, min=clip)

            # Update Nesterov parameter and momentum
            alpha_new = 0.5 * ((alpha**4 + 4 * alpha**2)**0.5 - alpha**2)
            beta = (alpha * (1 - alpha)) / (alpha**2 + alpha_new)

            # Nesterov momentum step
            v_new = u_new + beta * (u_new - u)

            # Early stopping condition
            if torch.norm(u_new - u) < tol:
                break

            # Prepare for next iteration
            u = u_new
            v = v_new
            # alpha = alpha_new  # don't forget to update alpha too!

        return u_new  # final iterate is returned
        
    @staticmethod
    def inv_softplus(x):
        """
        Inverse of Softplus activation function.
        """
        return x + torch.log(-torch.expm1(-x)) # x + log(1 - exp(-x))
    
    @staticmethod
    def inv_leaky_relu(x, negative_slope=0.01):
        """
        Inverse of Leaky ReLU activation function.
        """
        return torch.where(x >= 0, x, x / negative_slope)

    def ponderation(self, Z_inv_batch, Z_batch, X_batch):
        """
        Compute sample-wise ponderation values (δ_i) for a batch, using:

            δ_i = ||F(Ax_i) - z_i||² / ||Ax_i - F⁻¹(z_i)||²

        Parameters:
        -----------
        Z_inv_batch : torch.Tensor of shape (B, k)
            Inverse activation applied to z_i, i.e., F⁻¹(z_i)
        Z_batch : torch.Tensor of shape (B, k)
            Latent representations z_i
        X_batch : torch.Tensor of shape (B, n)
            Input data samples x_i

        Returns:
        --------
        delta_batch : torch.Tensor of shape (B,)
            Ponderation weights δ_i for each sample in the batch.
        """
        if self.delta == 'one':
            return torch.ones(X_batch.size(0), device=X_batch.device)
        elif self.delta == 'delta':
            Ax = self.linear1(X_batch)  # shape (B, k)
            num = torch.norm(self.activation(Ax) - Z_batch, dim=1)**2  # shape (B,)
            denom = torch.norm(Ax - Z_inv_batch, dim=1)**2  # shape (B,)
            delta_batch = torch.where(denom != 0, num / denom, torch.zeros_like(num))  # Avoid division by zero
            delta_batch = torch.clamp(delta_batch, min=0.0, max=100) # shape (B,)
            return delta_batch  
        else:
            raise ValueError(f"Unknown normalisation method: {self.delta}. Choose 'one' or 'delta'.")

    @torch.no_grad()
    def evaluate_loss(self, dataloader):
        if self.analyse:
            criterion = nn.MSELoss()
            loss = 0.0
            for X_batch, y_batch in dataloader:
                output = self(X_batch)
                loss += criterion(output, y_batch).item()
            return loss / len(dataloader)
        else:
            return 0.0

    @staticmethod
    def get_lr(input_dim, batch_size):
        """
        Compute the learning rate based on input dimension and batch size.
        This is a heuristic to ensure stable training.
        Parameters:
            input_dim: Dimension of the input data
            batch_size: Size of the training batch
        Returns:
            Learning rate
        """
        # Base learning rate for reference settings
        base_input_dim = 64
        base_batch_size = 128
        base_lr = 5*1e-3

        # Scale down with higher input_dim, scale up with higher batch_size
        lr = base_lr * (base_batch_size / batch_size) * (base_input_dim / input_dim)
        return lr