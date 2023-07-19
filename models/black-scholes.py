import torch
from torch.distributions import Normal
# Description:0

# We will be using PyTorch to calculate the price of
# European put options and the associated greeks
# via the Black Scholes model

# inputs for our model, we will be using an arbitrary
# example as the defaults 

std_norm_cdf = Normal(0, 1).cdf
std_norm_pdf = lambda x: torch.exp(Normal(0, 1).log_prob(x))


def bs_price(right, K, S, T, sigma, r):
    d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
    d_2 = d_1 - sigma * torch.sqrt(T)
    
    if right == "C":
        C = std_norm_cdf(d_1) * S - std_norm_cdf(d_2) * K * torch.exp(-r * T)
        return C
        
    elif right == "P":
        P = std_norm_cdf(-d_2) * K * torch.exp(-r * T) - std_norm_cdf(-d_1) * S
        return P

# which side of the trade you are on
# C being call and P being put
right = "C"
# Present Value of Exercise Price
K = torch.tensor(100.0, requires_grad=True)
# Current Stock Price
S = torch.tensor(100.0, requires_grad=True)
# Time
T = torch.tensor(1.0, requires_grad=True)
# Volatility
sigma = torch.tensor(0.20, requires_grad=True)
# Risk-Free rate
r = torch.tensor(0.00, requires_grad=True)

# Price
price = bs_price(right, K, S, T, sigma, r)
print(price)

# Tell PyTorch to compute gradients
price.backward()

# Greeks
print(f"Delta: {S.grad}\nVega: {sigma.grad}\nTheta: {T.grad}\nRho: {r.grad}")

# probability that option will be exercised
d_1 = (1 / (sigma * torch.sqrt(T))) * (torch.log(S / K) + (r + (torch.square(sigma) / 2)) * T)
# probability that option will be exercised
d_2 = d_1 - sigma * torch.sqrt(T)

delta = std_norm_cdf(d_1)
vega = S * std_norm_pdf(d_1) * torch.sqrt(T)
theta = ((S * std_norm_pdf(d_1) * sigma) / (2 * torch.sqrt(T))) + r * K * torch.exp(-r * T) * std_norm_cdf(d_2)
rho = K * T * torch.exp(-r * T) * std_norm_cdf(d_2)

S = torch.tensor(100.0, requires_grad=True)
price = bs_price(right, K, S, T, sigma, r)

delta = torch.autograd.grad(price, S, create_graph=True)[0]
delta.backward()

print(f"Autograd Gamma: {S.grad}")

# And the direct Black-Scholes calculation
gamma = std_norm_pdf(d_1) / (S * sigma * torch.sqrt(T))
print(f"BS Gamma: {gamma}")