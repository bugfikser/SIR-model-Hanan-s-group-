import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
# Path to the COVID-19 confirmed cases time series CSV file
# Adjust the path according to your local file location
df = pd.read_csv("C:\\Users\\91962\\Desktop\\COVID-19-master\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_confirmed_global.csv")

# Extract cumulative cases for India
india_cum = df[df["Country/Region"] == "India"].iloc[:, 4:].sum(axis=0)
india_cum.index = pd.to_datetime(india_cum.index)

# Calculate daily new cases
daily = india_cum.diff().fillna(0)  # Fill NaN (first day) with 0

# 2. Define the SIR model differential equations
def sir_ode(y, t, N, beta, gamma):
    """
    SIR model ODEs.
    y: array of [S, I, R]
    t: time points
    N: total population
    beta: infection rate
    gamma: recovery rate
    Returns: [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y
    dSdt = -beta * S * I / N      # Rate of change of susceptible
    dIdt = beta * S * I / N - gamma * I  # Rate of change of infected
    dRdt = gamma * I              # Rate of change of recovered
    return [dSdt, dIdt, dRdt]

# 3. Function to run the SIR model
def run_sir(beta, gamma, S0_override=None):
    """
    Run the SIR model and return daily new cases.
    beta: infection rate
    gamma: recovery rate
    S0_override: optional override for initial susceptible population
    Returns: array of daily new cases
    """
    y0 = [S0_override if S0_override is not None else S0, I0, R0]  # Initial conditions
    sol = odeint(sir_ode, y0, t, args=(N, beta, gamma))  # Solve ODE
    S, I, R = sol.T  # Transpose solution into S, I, R arrays
    new_cases = beta * S * I / N  # Calculate daily new cases
    return new_cases

# 4. Set initial conditions and parameters
N = 1_000_000  # Effective population (scaled down for computational simplicity)
I0 = 10        # Initial number of infected individuals
R0 = 0         # Initial number of recovered individuals
S0 = N - I0 - R0  # Initial number of susceptible individuals

# Baseline parameters
beta = 0.15     # Infection rate (contacts per day * transmission probability)
gamma = 0.05    # Recovery rate (1 / infectious period, e.g., 10 days)

# Time period for simulation (e.g., first 300 days to capture the first wave)
days_to_plot = 500
t = np.arange(days_to_plot)

# 5. Run the baseline SIR model
new_cases_baseline = run_sir(beta, gamma)

# 6. Scale the model to match the peak of the real data
peak_real = daily.iloc[:days_to_plot].max()  # Maximum daily cases in first 300 days
peak_pred = new_cases_baseline.max()         # Maximum predicted daily cases
scale = peak_real / peak_pred                # Scaling factor to align model with data

# 7. Define scenarios with adjusted parameters
scenarios = {
    "Baseline": (beta, gamma, None),                    # No intervention
    "Lockdown": (beta * 0.5, gamma, None),             # 50% reduction in transmission
    "Social Distancing": (beta * 0.7, gamma, None),    # 30% reduction in transmission
    "Improved Recovery": (beta, gamma * 1.5, None),    # 50% faster recovery
    "Vaccination": (beta, gamma, int(0.7 * N) - I0)    # 30% of population vaccinated
}
# Plot with shift
t_shifted = t + 150  # Shift peak to ~day 200
plt.plot(t_shifted, new_cases_baseline * scale, label="Adjusted Baseline")

# 8. Plot the results
plt.figure(figsize=(12, 6))

# Plot real daily cases for the first 300 days
plt.plot(np.arange(days_to_plot), daily.iloc[:days_to_plot], 'k.', label="Real Daily Cases (India)")

# Plot each scenario
for name, (b, g, S0_override) in scenarios.items():
    new_cases_pred = run_sir(b, g, S0_override)
    plt.plot(t_shifted, new_cases_pred * scale, label=name)

# Add labels, title, legend, and grid
plt.title("SIR Model Scenarios vs Real COVID-19 Daily Cases in India")
plt.xlabel("Days")
plt.ylabel("Daily New Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()