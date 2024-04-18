import pybamm

class WSP(pybamm.BaseModel):
    """
    Single Particle model done as part of the developer workshop on 17th April 2024
    """

    def __init__(self, name="Single Particle Model"):
        super().__init__(name=name)

        #Register citation
        pybamm.citations.register("Marquis2019")

        # This command registers the article where the model comes from so it appears
        # when calling `pybamm.print_citations()`
        pybamm.citations.register("Marquis2019")

        ######################
        # Variables
        ######################
        # Define the variables of the model.
        c_i = [pybamm.Variable(f"{e.capitalize()} particle concentration [mol.m-3]", domain=f"{e} particle") for e in electrodes]

    


        ######################
        # Parameters
        ######################
        # define parameters
        electrodes = ["negative", "positive"]
        I = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
        D_i = [pybamm.Parameter(f"{e.capitalize()} particle diffusivity [m2.s-1]") for e in electrodes]
        R_i = [pybamm.Parameter(f"{e.capitalize()} particle radius [m]") for e in electrodes]
        c0_i = [pybamm.Parameter(f"Initial concentration in {e} electrode [mol.m-3]") for e in electrodes]
        delta_i = [pybamm.Parameter(f"{e.capitalize()} electrode thickness [m]") for e in electrodes]
        A = pybamm.Parameter("Electrode width [m]") * pybamm.Parameter("Electrode height [m]")  # PyBaMM takes the width and height of the electrodes (assumed rectangular) rather than the total area
        epsilon_i = [pybamm.Parameter(f"{e.capitalize()} electrode active material volume fraction") for e in electrodes]

        # define universal constants (PyBaAMM has them built in)
        F = pybamm.constants.F
        R = pybamm.constants.R

        # define variables that depend on the parameters
        a_i = [3 * epsilon_i[i] / R_i[i] for i in [0, 1]]
        j_i = [I / a_i[0] / delta_i[0] / F / A, -I / a_i[1] / delta_i[1] / F / A]

        # define temperature T
        T = pybamm.Parameter("Ambient temperature [K]")

        ######################
        # Particle model
        ######################
        # governing equations
        dcdt_i = [pybamm.div(D_i[i] * pybamm.grad(c_i[i])) for i in [0, 1]]
        model.rhs = {c_i[i]: dcdt_i[i] for i in [0, 1]}

        # boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = [-j_i[i] / D_i[i] for i in [0, 1]]
        model.boundary_conditions = {c_i[i]: {"left": (lbc, "Neumann"), "right": (rbc[i], "Neumann")} for i in [0, 1]}

        # initial conditions
        model.initial_conditions = {c_i[i]: c0_i[i] for i in [0, 1]}


        ######################
        # Output variables
        ######################


        # Populate the variables dictionary with any variables you want to compute
        self.variables = {}

    # The following attributes are used to define the default properties of the model,
    # which are used by the simulation class if unspecified.
    @property
    def default_geometry(self):
        return None

    @property
    def default_submesh_types(self):
        return None

    @property
    def default_var_pts(self):
        return None

    @property
    def default_spatial_methods(self):
        return None

    @property
    def default_solver(self):
        return None
    
    @property
    def default_quick_plot_variables(self):
        return None












