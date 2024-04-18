import pybamm
import numpy as np

x_n = pybamm.Variable("Negative electrode stochiometry")
x_p = pybamm.Variable("Positive electrode stochiometry")

i = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
x_n_0 = pybamm.Parameter("Initial negative electrode stochiometry")
x_p_0 = pybamm.Parameter("Initial positive electrode stochiometry")
U_p = pybamm.FunctionParameter("Positive electrode OCV", {"x_p": x_p})
U_n = pybamm.FunctionParameter("Negative electrode OCV", {"x_n": x_n})
Q_n = pybamm.Parameter("Negative electrode capacity [A.h]")
Q_p = pybamm.Parameter("Positive electrode capacity [A.h]")
R = pybamm.Parameter("Electrode resistance [Ohm]")

model = pybamm.BaseModel("reservoir model")
model.rhs[x_n] = -i / Q_n
model.initial_conditions[x_n] = x_n_0
model.rhs[x_p] = i / Q_p
model.initial_conditions[x_p] = x_p_0

model.variables["Voltage [V]"] = U_p - U_n -  i * R
model.variables["Negative electrode stochiometry"] = x_n
model.variables["Positive electrode stochiometry"] = x_p

model.rhs[x_n].visualise("x_n_rhs.png")

model.events = [pybamm.Event("Minimim negative stochiometry", x_n),
pybamm.Event("Maximum negative stochiometry", pybamm.Scalar(1)-x_n),
pybamm.Event("Minimim positive stochiometry", x_p),
pybamm.Event("Minimim positive stochiometry", 1-x_p)]

def graphite_LGM50_ocp_Chen2020(sto):
  u_eq = (
      1.9793 * np.exp(-39.3631 * sto)
      + 0.2482
      - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
      - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
      - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
  )

  return u_eq

def nmc_LGM50_ocp_Chen2020(sto):
  u_eq = (
      -0.8090 * sto
      + 4.4875
      - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
      - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
      + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
  )
  
  return u_eq


param = pybamm.ParameterValues({
    "Current function [A]": lambda t: 1 + 0.5 * pybamm.sin(100*t),
    "Initial negative electrode stochiometry": 0.9,
    "Initial positive electrode stochiometry": 0.1,
    "Negative electrode capacity [A.h]": 1,
    "Positive electrode capacity [A.h]": 1,
    "Electrode resistance [Ohm]": 0.1,
    "Positive electrode OCV": nmc_LGM50_ocp_Chen2020,
    "Negative electrode OCV": graphite_LGM50_ocp_Chen2020,
})

sim = pybamm.Simulation(model, parameter_values=param)
sol = sim.solve([0, 1])
sol.plot(["Voltage [V]", "Negative electrode stochiometry", "Positive electrode stochiometry"])