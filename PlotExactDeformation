import numpy as np
import matplotlib.pyplot as plt

# Grid of points to evaluate deformation
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y)
Q = 4

# ux(x, y) = cos(2*pi*x) * sin(pi*y)
ux_exact = np.cos(2 * np.pi * X) * np.sin(np.pi * Y)
# uy(x, y) = sin(pi*x) * Q * y^4 / 4
uy_exact = np.sin(np.pi * X) * (Q * Y**4) / 4

#PLOTTING COLOR MAP
# Plot the color map results
plt.figure(figsize=(15, 6))

# Displacement in x direction
plt.subplot(1, 2, 1)
plt.contourf(X, Y, ux_exact, levels=50, cmap='coolwarm', norm=plt.Normalize(vmin=-0.8, vmax=0.8))
plt.colorbar(label='Displacement in x direction (ux)')
plt.title('Displacement in x direction (ux)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# Displacement in y direction
plt.subplot(1, 2, 2)
plt.contourf(X, Y, uy_exact, levels=50, cmap='coolwarm', norm=plt.Normalize(vmin=-0.8, vmax=0.8))
plt.colorbar(label='Displacement in y direction (uy)')
plt.title('Displacement in y direction (uy)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

plt.tight_layout()
plt.savefig('deformation_results_exact_5000_color_map_v9.png', dpi=300)
plt.close()

#PLOTTING DEFORMED GRID
#Plot deformed grid results
plt.figure(figsize=(8,8))

#Deformed grid values
deformed_X = X + ux_exact
deformed_Y = Y + uy_exact

#Plotting both original and deformed grids
for i in range(X.shape[0]):
    plt.plot(X[i, :], Y[i, :], 'k--', lw=0.5)
    plt.plot(deformed_X[i, :], deformed_Y[i, :], 'b-', lw=1)
for j in range(X.shape[1]):
    plt.plot(X[:, j], Y[:, j], 'k--', lw=0.8)
    plt.plot(deformed_X[:, j], deformed_Y[:, j], 'b-', lw=1)
plt.title("Deformed Grid (Exact Function)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')

plt.tight_layout()
plt.savefig('Deformed Grid_Exact_5000_v9.png', dpi=300)
plt.close()
