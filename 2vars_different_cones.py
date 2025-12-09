import numpy as np  
import cvxpy as cp  
import matplotlib.pyplot as plt  


# problem data to make the cones important
Q0 = np.array([[12, 8], [8, 10]])  # Stronger interaction between variables
c0 = np.array([-15, -10])         # Pushes towards different regions

# Adjust quadratic constraint to be more restrictive
Q1 = np.array([[6, 3], [3, 5]])  # Making it more dominant
c1 = np.array([-6, 8])           # Adjusted for stronger effect
d1 = -10                         # Tightened constraint


# Define different cone choices 
cones = {
    "Nonnegative Orthant": lambda y, z: [y >= 0, z >= 0], #y1​≥0,y2​≥0,z1​≥0,z2​≥0
    "Second-Order Cone": lambda y, z: [cp.SOC(y[0], y[1:]), cp.SOC(z[0], z[1:])], #∣y2​∣≤y1
    "Power Cone": lambda y, z: [
        cp.geo_mean(cp.hstack([y[0], y[1]])) >= cp.abs(y[1]),  
        cp.geo_mean(cp.hstack([z[0], z[1]])) >= cp.abs(z[1])  # y1​y2​≥y2^2
    ]
}

#  Solve the problem for each cone type
fig, axes = plt.subplots(1, len(cones), figsize=(15, 5))
optimal_solutions = {}

for i, (cone_name, cone_constraints) in enumerate(cones.items()):
    
    # Standard case: Define y and z variables
    y = cp.Variable(2)
    z = cp.Variable(2)
    x = y - z  

    # Objective function: Minimize quadratic cost function
    objective = cp.Minimize(cp.quad_form(x, Q0) + c0.T @ x)

    # Constraints: Cone constraints + Quadratic constraint
    constraints = cone_constraints(y, z)
    constraints.append(cp.quad_form(x, Q1) + c1.T @ x + d1 <= 0)

    # ---- Step 4: Solve the problem ----
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    # ---- Step 5: Store and print the solution ----
    if prob.status not in ["infeasible", "unbounded"]:
        
        x_opt = y.value - z.value
        optimal_solutions[cone_name] = x_opt
        print(f"Optimal solution for {cone_name}: x* = {x_opt}, Objective value = {prob.value}")
    else:
        x_opt = None
        print(f"Solver failed for {cone_name}: {prob.status}")

    # ---- Step 6: Visualization ----
    ax = axes[i]
    ax.set_title(cone_name)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # ---- Step 7: Plot the feasible region for each cone ----
    t_vals = np.linspace(0, 10, 100)

    if cone_name == "Second-Order Cone":
        t_vals = np.linspace(0, 10, 100)
        ax.fill_between(t_vals, -t_vals, t_vals, color='lightblue', alpha=0.5, label="Feasible Region")


    elif cone_name == "Nonnegative Orthant":
        ax.fill_between([0, 10], [0, 0], [10, 10], color='lightgreen', alpha=0.5, label="Feasible Region")

    elif cone_name == "Power Cone":
        # Define the range for y1 and y2 including negative values
        y1_vals = np.linspace(-10, 10, 400)
        y2_vals = np.linspace(-10, 10, 400)
    
        # Create a meshgrid for y1 and y2
        Y1, Y2 = np.meshgrid(y1_vals, y2_vals)
    
        # Define the feasible region for y1 * y2 >= y2^2
        feasible_region = (Y1 * Y2) >= (Y2 ** 2)
    
        # Plot the feasible region
        ax.contourf(Y1, Y2, feasible_region, levels=[0.5, 1], colors='orange', alpha=0.5)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")

plt.tight_layout()
plt.show()
