import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic data to mimic the experimental data

#Physics Parameters
g=9.8
h0=1.0
v0=10.0

# True (analytical) solution according to Newtonian Physics - h(t) = h0+v0*t-0.5*g*t^2
def true_solution(t):
  return h0+v0*t-0.5*g*(t**2)

# Generate some time points
t_min, t_max = 0.0, 2.0
N_data = 10
t_data = np.linspace(t_min, t_max, N_data)

#Generate synthetic "experimental" heights with noise
np.random.seed(0)
noise_level=0.7
h_data_exact = true_solution(t_data)
h_data_noisy = h_data_exact + noise_level*np.random.randn(N_data)

#Convert to pytorch tensors
t_data_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1,1)
h_data_tensor = torch.tensor(h_data_noisy, dtype=torch.float32).view(-1,1)


class PINN(nn.Module):
  def __init__(self,n_hidden=20):
    super(PINN, self).__init__()
    #A simple MLP with 2 hidden layers
    self.net=nn.Sequential(
        nn.Linear(1,n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, n_hidden),
        nn.Tanh(),
        nn.Linear(n_hidden, 1)
    )
  def forward(self, t):
    """
    Forward pass: input shape(batch_size, 1) -> output shape (batch_size)
    """
    return self.net(t)

#Instantiate the model
model = PINN(n_hidden=20)

# Helper for Automatic Diff

def derivative(y,x):
  """
  Computes dy/dx using PyTorch's autograd.
  y and x must be tensors with requires_grad=True for x.
  """
  return torch.autograd.grad(
      y,x,
      grad_outputs=torch.ones_like(y),
      create_graph=True
  )[0]

# Define the Loss Components (PINN)

# We have:
#  (1). Data Loss (fit noisy data)
#  (2). ODE Loss: dh/dt = v0 - g*t
#  (3). Initial condition loss: h(0)=h0

def physics_loss(model, t):
  """
  Compare d(h_pred)/dt with the known expression (v0-gt).
  """
  # t must have requires_grad = True for autograd to work
  t.requires_grad_(True)

  h_pred = model(t)
  dh_dt_pred = derivative(h_pred, t)

  #For each t, physics says dh/dt = v0 - g*t
  dh_dt_true = v0 - g * t

  loss_ode = torch.mean((dh_dt_pred - dh_dt_true)**2)
  return loss_ode

def initial_condition_loss(model):
  """
  Enforce h(0)=h0.
  """
  #Evaluate at t=0
  t0=torch.zeros(1,1, dtype=torch.float32, requires_grad=False)
  h0_pred = model(t0)
  return (h0_pred-h0).pow(2).mean()

def data_loss(model,t_data,h_data):
  """
  MSE between predicted h(t_i) and noisy measurements h_data.
  """
  h_pred = model(t_data)
  return torch.mean((h_pred - h_data)**2)

# Training Setup

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate collocation points for physics loss (more points than experimental data)
t_collocation = np.linspace(t_min, t_max, 100).reshape(-1,1).astype(np.float32)
t_physics_tensor = torch.tensor(t_collocation, requires_grad=True)

# Adjust Hyperparameters to favor physics regularization
lambda_data = 5.0
lambda_ode = 10.0
lambda_ic = 5.0

num_epochs = 5000
print_every = 500

# Training Loop

model.train()
for epoch in range(num_epochs):
  optimizer.zero_grad()

  # 1. Data Loss (on the 10 noisy points)
  l_data = data_loss(model, t_data_tensor, h_data_tensor)
  
  # 2. Physics Loss (on 100 collocation points across the whole domain)
  l_ode = physics_loss(model, t_physics_tensor)
  
  # 3. Initial Condition Loss
  l_ic = initial_condition_loss(model)

  # Combined Loss
  loss = lambda_data * l_data + lambda_ode * l_ode + lambda_ic * l_ic

  # Backprop
  loss.backward()
  optimizer.step()

  # Print progress
  if (epoch + 1) % print_every == 0:
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Total Loss = {loss.item():.6f}, "
          f"Data Loss = {l_data.item():.6f}, "
          f"ODE Loss = {l_ode.item():.6f}, "
          f"IC Loss = {l_ic.item():.6f}")

model.eval()
t_plot=np.linspace(t_min, t_max, 100).reshape(-1,1).astype(np.float32)
t_plot_tensor = torch.tensor(t_plot, requires_grad=True)
h_pred_plot = model(t_plot_tensor).detach().numpy()

# True solution (for comparison)
h_true_plot = true_solution(t_plot)

#Plot results
plt.figure(figsize=(8,5))
plt.scatter(t_data, h_data_noisy, color="red", label="Noisy Data")
plt.plot(t_plot,h_true_plot,'k--',label="Exact Solution")
plt.plot(t_plot, h_pred_plot, 'b', label="PINN Prediction")
plt.xlabel('t')
plt.ylabel('h(t)')
plt.legend()
plt.title('PINN for ball trajectory')
plt.grid(True)
plt.show()
