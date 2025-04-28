import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Model definition and parameters
model = pybamm.lithium_ion.SPM(
    options={
        "cell geometry": "pouch",
        "current collector": "potential pair",
        "dimensionality": 2,
        "thermal": "x-lumped",
    },
)

# Define parameters using PyBaMM's parameter interface
E_C = pybamm.Parameter("Young modulus Cathode [Pa]")
E_A = pybamm.Parameter("Young modulus Anode [Pa]")
a_C = pybamm.Parameter("Expansion coefficient Cathode")
a_A = pybamm.Parameter("Expansion coefficient Anode")
poisson_ratio = pybamm.Parameter("Poisson ratio")
t_C = pybamm.Parameter("Positive electrode thickness [m]")
t_A = pybamm.Parameter("Negative electrode thickness [m]")

# Set parameter values
parameter_values = pybamm.ParameterValues("Marquis2019")

parameter_values.update({"Young modulus Cathode [Pa]": 10**9}, check_already_exists=False)
parameter_values.update({"Young modulus Anode [Pa]": 10**9}, check_already_exists=False)
parameter_values.update({"Expansion coefficient Cathode": 13/300}, check_already_exists=False)
parameter_values.update({"Expansion coefficient Anode": 0}, check_already_exists=False)
parameter_values.update({"Poisson ratio": 0.2}, check_already_exists=False)

# Expansion: we include an ad-hoc prediction of the swelling based on experimental data.
# The anode swells up to 13% in volume, 4.4% in strain. The cathode strain is smaller, we assume it to be 0.
#These need to be correctly parametrised.


def swelling_A(stoc):
    return a_A * stoc

def swelling_C(stoc):
    return a_C * stoc


# Define custom stress functions using PyBaMM expressions
prefactor = - (poisson_ratio / ((t_A / E_A + t_C / E_C) * (1 - poisson_ratio) * (1 - 2 * poisson_ratio)))

def sigma_A_yy_zz(alpha_A, alpha_C):
    return - prefactor * (t_C * alpha_C + t_A * alpha_A) - E_A / (1 - poisson_ratio) * alpha_A

def sigma_C_yy_zz(alpha_A, alpha_C):
    return - prefactor * (t_C * alpha_C + t_A * alpha_A) - E_C / (1 - poisson_ratio) * alpha_C

def sigma_xx(alpha_A, alpha_C):
    return prefactor * ((1 - poisson_ratio) / poisson_ratio )* (t_C * alpha_C + t_A * alpha_A)

# Retrieve stoichiometry from model variables
x_p = model.variables['Positive electrode stoichiometry']
x_n = model.variables['Negative electrode stoichiometry']

# Register stress variables
model.variables["In-plane stress Anode (yy,zz) [Pa]"] = sigma_A_yy_zz(swelling_A(x_p), swelling_C(x_n))
model.variables["In-plane stress Cathode (yy,zz) [Pa]"] = sigma_C_yy_zz(swelling_A(x_p), swelling_C(x_n))
model.variables["Through-cell stress xx [Pa]"] = sigma_xx(swelling_A(x_p), swelling_C(x_n))
model.variables["Expansion strain anode"] = swelling_C(x_n)
model.variables["Expansion strain cathode"] = swelling_A(x_p)

# Setup solver and simulation
solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-3, root_tol=1e-3, mode="fast")

experiment = pybamm.Experiment(["Discharge at 2C for 40 minutes"])

sim = pybamm.Simulation(
    model, experiment=experiment, parameter_values=parameter_values, solver=solver
)

# Solve the model
sol = sim.solve()


def plot_stress_and_expansion_distribution(solution, t):
    # Mesh for plotting
    L_y = parameter_values.evaluate(pybamm.Parameter("Electrode width [m]"))
    L_z = parameter_values.evaluate(pybamm.Parameter("Electrode height [m]"))
    y_plot = np.linspace(0, L_y, 41)
    z_plot = np.linspace(0, L_z, 41)
    
    # Evaluate stress and expansion variables at the given time
    anode_stress = np.array(solution["In-plane stress Anode (yy,zz) [Pa]"](y=y_plot, z=z_plot, t=t)).T
    cathode_stress = np.array(solution["In-plane stress Cathode (yy,zz) [Pa]"](y=y_plot, z=z_plot, t=t)).T
    through_cell_stress = np.array(solution["Through-cell stress xx [Pa]"](y=y_plot, z=z_plot, t=t)).T
    expansion_anode = np.array(solution["Expansion strain anode"](y=y_plot, z=z_plot, t=t)).T
    expansion_cathode = np.array(solution["Expansion strain cathode"](y=y_plot, z=z_plot, t=t)).T

    # Create figure with subplots arranged horizontally
    fig, axs = plt.subplots(1, 5, figsize=(25, 6))

    # Anode Stress Plot
    anode_stress_plot = axs[0].pcolormesh(y_plot, z_plot, anode_stress, shading="gouraud", cmap='viridis')
    axs[0].set_xlabel(r"$y$ [m]")
    axs[0].set_ylabel(r"$z$ [m]")
    axs[0].set_title(r"Anode Stress $\sigma_{yy/zz}$ [Pa]")
    fig.colorbar(anode_stress_plot, ax=axs[0])

    # Cathode Stress Plot
    cathode_stress_plot = axs[1].pcolormesh(y_plot, z_plot, cathode_stress, shading="gouraud", cmap='viridis')
    axs[1].set_xlabel(r"$y$ [m]")
    axs[1].set_title(r"Cathode Stress $\sigma_{yy/zz}$ [Pa]")
    fig.colorbar(cathode_stress_plot, ax=axs[1])

    # Through Cell Stress Plot
    through_cell_stress_plot = axs[2].pcolormesh(y_plot, z_plot, through_cell_stress, shading="gouraud", cmap='viridis')
    axs[2].set_xlabel(r"$y$ [m]")
    axs[2].set_title(r"Through-cell Stress $\sigma_{xx}$ [Pa]")
    fig.colorbar(through_cell_stress_plot, ax=axs[2])

    # Anode Expansion Plot
    expansion_anode_plot = axs[3].pcolormesh(y_plot, z_plot, expansion_anode, shading="gouraud", cmap='viridis')
    axs[3].set_xlabel(r"$y$ [m]")
    axs[3].set_title(r"Anode Expansion")
    fig.colorbar(expansion_anode_plot, ax=axs[3])

    # Cathode Expansion Plot
    expansion_cathode_plot = axs[4].pcolormesh(y_plot, z_plot, expansion_cathode, shading="gouraud", cmap='viridis')
    axs[4].set_xlabel(r"$y$ [m]")
    axs[4].set_title(r"Cathode Expansion")
    fig.colorbar(expansion_cathode_plot, ax=axs[4])
    
    # Adjust layout to fit the subplots neatly
    plt.tight_layout()
    plt.show()

# Example usage
plot_stress_and_expansion_distribution(sol, t=38*60)  # Replace `100` with the specific time in seconds at which you want to plot the stress and expansion
