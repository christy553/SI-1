import deepxde as dde
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pdb

def main():
    parser = argparse.ArgumentParser()

    # Add command-line arguments
    parser.add_argument('--train', action='store_true', help='Train the model and save it')
    parser.add_argument('--load', type=str, help='Path to the saved model to load')
    parser.add_argument('--model_filename', type=str, default='model_checkpoint.h5', help='Filename for saving the model')

    args = parser.parse_args()

    lmbd = 1.0
    mu = 0.5
    Q = 4.0

    sin = dde.backend.sin
    cos = dde.backend.cos
    stack = dde.backend.stack
    model_filename = args.model_filename

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    BC_type = ["hard", "soft"][0]

    def boundary_left(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_right(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1.0)

    def boundary_top(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 1.0)

    def boundary_bottom(x, on_boundary):
        return on_boundary and dde.utils.isclose(x[1], 0.0)

    # Exact solutions
    def func(x):
        ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
        uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

        E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
        E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
        E_xy = 0.5 * (
            np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
            + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
        )

        Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        Sxy = 2 * E_xy * mu

        return np.hstack((ux, uy, Sxx, Syy, Sxy))

    # Soft Boundary Conditions
    ux_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=0)
    ux_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=0)
    uy_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=1)
    uy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=1)
    uy_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=1)
    sxx_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=2)
    sxx_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=2)
    syy_top_bc = dde.icbc.DirichletBC(
        geom,
        lambda x: (2 * mu + lmbd) * Q * np.sin(np.pi * x[:, 0:1]),
        boundary_top,
        component=3,
    )

    # Hard Boundary Conditions
    def hard_BC(x, f):
        Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
        Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]

        Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
        Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * sin(np.pi * x[:, 0])
        Sxy = f[:, 4]
        return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)

    def fx(x):
        return (
            -lmbd
            * (
                4 * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
                - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
            )
            - mu
            * (
                np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
                - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
            )
            - 8 * mu * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
        )

    def fy(x):
        return (
            lmbd
            * (
                3 * Q * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
                - 2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
            )
            - mu
            * (
                2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
                + (Q * x[:, 1:2] ** 4 * np.pi**2 * sin(np.pi * x[:, 0:1])) / 4
            )
            + 6 * Q * mu * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
        )


    def jacobian(f, x, i, j):
        if dde.backend.backend_name == "jax":
            return dde.grad.jacobian(f, x, i=i, j=j)[0]
        else:
            return dde.grad.jacobian(f, x, i=i, j=j)

    def pde(x, f):
        E_xx = jacobian(f, x, i=0, j=0)
        E_yy = jacobian(f, x, i=1, j=1)
        E_xy = 0.5 * (jacobian(f, x, i=0, j=1) + jacobian(f, x, i=1, j=0))

        S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
        S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
        S_xy = E_xy * 2 * mu

        Sxx_x = jacobian(f, x, i=2, j=0)
        Syy_y = jacobian(f, x, i=3, j=1)
        Sxy_x = jacobian(f, x, i=4, j=0)
        Sxy_y = jacobian(f, x, i=4, j=1)

        momentum_x = Sxx_x + Sxy_y - fx(x)
        momentum_y = Sxy_x + Syy_y - fy(x)

        if dde.backend.backend_name == "jax":
            f = f[0]  # f[1] is the function used by jax to compute the gradients

        stress_x = S_xx - f[:, 2:3]
        stress_y = S_yy - f[:, 3:4]
        stress_xy = S_xy - f[:, 4:5]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

    if BC_type == "hard":
        bcs = []
    else:
        bcs = [
            ux_top_bc,
            ux_bottom_bc,
            uy_left_bc,
            uy_bottom_bc,
            uy_right_bc,
            sxx_left_bc,
            sxx_right_bc,
            syy_top_bc,
        ]

    data = dde.data.PDE(
        geom,
        pde,
        bcs,
        num_domain=500,
        num_boundary=500,
        solution=func,
        num_test=100,
    )

    layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.PFNN(layers, activation, initializer)
    if BC_type == "hard":
        net.apply_output_transform(hard_BC)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    # Train or load the model based on the argument
    if args.train:
        losshistory, train_state = model.train(iterations=5000)
        # Save the model after training
        model.save(model_filename)
        print(f"Model saved to {model_filename}")
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        
    elif args.load:
        if os.path.exists(args.load):
            model.restore(args.load)
            print(f"Model loaded from {args.load}")
            
            # Grid of points to evaluate deformation
            x = np.linspace(0, 1, 50)
            y = np.linspace(0, 1, 50)
            X, Y = np.meshgrid(x, y)
            points = np.vstack([X.ravel(), Y.ravel()]).T

            # Predict deformation at the grid points
            predictions = model.predict(points)

            # Predicted displacements
            ux_pred = predictions[:, 0]  # x-displacement
            uy_pred = predictions[:, 1]  # y-displacement

            # Reshape the outputs for visualization
            ux_pred = ux_pred.reshape(X.shape)
            uy_pred = uy_pred.reshape(X.shape)
            
            #PLOTTING COLOR MAP
            # Plot the color map results
            plt.figure(figsize=(15, 6))

            # Displacement in x direction
            plt.subplot(1, 2, 1)
            plt.contourf(X, Y, ux_pred, levels=50, cmap='coolwarm', norm=plt.Normalize(vmin=-0.8, vmax=0.8))
            plt.colorbar(label='Displacement in x direction (ux)')
            plt.title('Displacement in x direction (ux)')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')

            # Displacement in y direction
            plt.subplot(1, 2, 2)
            plt.contourf(X, Y, uy_pred, levels=50, cmap='coolwarm', norm=plt.Normalize(vmin=-0.8, vmax=0.8))
            plt.colorbar(label='Displacement in y direction (uy)')
            plt.title('Displacement in y direction (uy)')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')

            plt.tight_layout()
            plt.savefig('deformation_results_5000_color_map_v9.png', dpi=300)
            plt.close()
            
            #PLOTTING DEFORMED GRID
            #Plot deformed grid results
            plt.figure(figsize=(8,8))
            
            #Deformed grid values
            scale_factor = 0.1
            deformed_X = X + ux_pred*scale_factor
            deformed_Y = Y + uy_pred*scale_factor
            
            #Plotting both original and deformed grids
            for i in range(X.shape[0]):
                plt.plot(X[i, :], Y[i, :], 'k--', lw=0.5)
                plt.plot(deformed_X[i, :], deformed_Y[i, :], 'b-', lw=1)
            for j in range(X.shape[1]):
                plt.plot(X[:, j], Y[:, j], 'k--', lw=0.8)
                plt.plot(deformed_X[:, j], deformed_Y[:, j], 'b-', lw=1)
            plt.title("Deformed Grid")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.axis('equal')
            
            plt.tight_layout()
            plt.savefig('Deformed Grid_5000_v9.png', dpi=300)
            plt.close()
            
        else:
            print(f"Error: Model file {args.load} not found.")
            '''
            # Plot the vector field
            plt.figure(figsize=(6, 6))
            scale_factor = 0.03
            plt.quiver(X, Y, ux_pred*scale_factor, uy_pred*scale_factor, angles='xy', scale_units='xy', scale=1, color='blue')
            plt.title('Predicted Deformation Vector Field')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.grid()
            plt.savefig('deformation_100_vector_field_scaled.png', dpi=300)
            plt.close()
            '''
if __name__ == "__main__":
    main()
