
import numpy as np
import matplotlib.pyplot as plt

# Define all the constants parameter
hbar = 1.0      # Reduced Planck's constant
m = 1.0         # Mass of particle
alpha = 1.0     #alpha parameter for the potential well
lambda_ = 4.0   #lambda parameter for the potential well

a = -3     # Left boundary
b = 3      # Right boundary
dx = 0.01  # Step size
x = np.arange(a,b,dx)

# Define the potential well
def potential_well(x):
    return (hbar ** 2 / (2 * m)) * (alpha ** 2) * lambda_ * (lambda_ - 1) * (0.5 - 1 / (np.cosh(alpha * x)) ** 2)
V = potential_well(x)

# Define the numerov method
def numerov(x, E, V):
    dx2 = dx**2              # Square of the step size
    psi = np.zeros_like(x)
    psi[0] = 0.0
    psi[1] = 1e-6

    k = ((2*m)/hbar**2)*(E-V)  
    for i in range(1, len(x) - 1):
        psi[i + 1] = (2 * (1 - 5/12 * dx2 * k[i]) * psi[i] - (1 + 1/12 * dx2 * k[i-1]) * psi[i-1]) / (1 + 1/12 * dx2 * k[i+1])
    return psi

def matching_function(E, x, V):
    psi_left = numerov(x, 0, 1e-6, E, V)
    psi_right = numerov(x[::-1], 0, 1e-3, E, V)[::-1]
    midpoint = len(x) // 2
    # Defining the 2 parts of the three point formula
    term_one = (psi_left[midpoint + 1] - psi_left[midpoint - 1]) / (2 * dx * psi_left[midpoint])
    term_two = (psi_right[midpoint + 1] - psi_right[midpoint - 1]) / (2 * dx * psi_right[midpoint])

    # Returning the value of the three point formula
    return term_one - term_two
     
   
# Define the eigenvalues function
def E(n):
    return ((hbar ** 2 / (2 * m)) * (alpha ** 2))*(((lambda_ * (lambda_ - 1))/2) - (lambda_ - 1 - n)**2)

# Find the eigenvalues
def find_eigenvalue(n_max):
    eigenvalues = []
    for n in range (n_max):
         eigenvalues.append(E(n))
    return eigenvalues

# Print the first three eigenvalues
n_max = 3
eigenvalues = find_eigenvalue(n_max)
print("First three computed eigenvalues:", eigenvalues)

# Plot potential and wavefunctions
plt.figure(figsize=(10, 6))
plt.plot(x, V, label="Potential V(x)")

for i, E in enumerate(eigenvalues[:3]):  # Plot first 3 eigenfunctions
    psi = numerov(x, E, V)
    psi_norm = psi / np.max(np.abs(psi))  # Normalize the wavefunction
    plt.plot(x, psi_norm + E, label=f" Ψ(x) for E={E:.4f}")

plt.title("Potential & Wavefunctions for 1D Schrodinger equation")
plt.xlabel("x")
plt.ylabel("V(x) , Ψ(x)")
plt.legend()
plt.grid()
plt.show()


# Plot eigenvalues and potential
plt.figure(figsize=(10, 6))
plt.plot(x, V, label="Potential V(x)")
for i, E in enumerate(eigenvalues):
    plt.hlines(E, xmin=a, xmax=b, colors='r', linestyles='dashed', label=f'E {i} = {E:.4f}' )

# Add plot configurations
plt.title("Eigenvalues")
plt.xlabel("x")
plt.ylabel("Energy E")
plt.legend()
plt.grid()
plt.show()

    

     