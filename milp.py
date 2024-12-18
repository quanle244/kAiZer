import pandas as pd
import numpy as np
import csv
from pulp import *

# Load input data from Excel files
demand = pd.read_excel("Input_Demand.xlsx", header=None).values.astype(float)
purchase_cost = pd.read_excel("Input_PurchaseCost.xlsx", header=None).values.astype(float)
weights = pd.read_excel("Input_Weightperbox.xlsx", header=None).values.astype(float)

# Define holding cost rate
holding_cost_rate = 0.0021
num_products, num_months = demand.shape
M = demand.max() * 10  # Large constant for the binary constraint

# Define the ordering cost based on weight ranges
def get_order_cost(weight):
    if weight > 8000:
        return 8500
    elif 5000 <= weight <= 8000:
        return 7300
    elif 3000 <= weight < 5000:
        return 6000
    else:
        return 4500

# Initialize the MILP problem
lp = LpProblem("LotSizingProblem", LpMinimize)

# Decision Variables
Q = [[LpVariable(f"Q_{i}_{t}", lowBound=0, cat="Continuous") for t in range(num_months)] for i in range(num_products)] # Quantity of product i ordered in month t (continuous)
I = [[LpVariable(f"I_{i}_{t}", lowBound=0, cat="Continuous") for t in range(num_months)] for i in range(num_products)] # Ending inventory of product i in month t (continuous)
X = [[LpVariable(f"X_{i}_{t}", cat="Binary") for t in range(num_months)] for i in range(num_products)] # Binary variable indicating if product i was ordered in month t (1 = ordered, otherwise 0)

# Objective Function
objective_function = lpSum(
    purchase_cost[i, t] * Q[i][t] +  # Purchase cost
    holding_cost_rate * purchase_cost[i, t] * I[i][t] +  # Holding cost
    get_order_cost(weights[i, 0]) * X[i][t]  # Ordering cost
    for i in range(num_products) for t in range(num_months)
)
lp += objective_function

# Constraints
for i in range(num_products):
    for t in range(num_months):
        # Inventory balance constraint
        lp += ((I[i][t - 1] if t > 0 else 0) + Q[i][t] - demand[i, t] == I[i][t]) # Ending Inventory of product i in last month t + Ordered quantity of that product in month t - Demand of that product in month t 

        # BigM constraint
        lp += Q[i][t] <= M * X[i][t]

# Solve the problem
lp.solve()

# Extract results
order_plans = np.zeros((num_products, num_months), dtype=float)
total_costs = np.zeros(num_products)

for i in range(num_products):
    for t in range(num_months):
        order_plans[i, t] = Q[i][t].varValue # Insert ordered quantity into order_plans list
        total_costs[i] += (
            purchase_cost[i, t] * Q[i][t].varValue +
            holding_cost_rate * purchase_cost[i, t] * I[i][t].varValue +
            get_order_cost(weights[i, 0]) * X[i][t].varValue
        )

# Display Results
for product in range(len(order_plans)):
    print(f"Product {product + 1}")
    print(f"Order Plan (by month): {order_plans[product]}")
    print(f"Total Cost: {total_costs[product]}")
    print("-" * 40)

order_plans_matrix = np.array(order_plans)
print("***")
print("Order plan for all prodcuts:")
print("-" * 50)
print(order_plans_matrix)
print("-" * 50)
print("***")
print(f"Total cost for planning horizon using MILP is: {sum(total_costs)}")
print("***")

# Write order_plans into csv file:
with open("Output_Order_plan_using_MILP.csv", encoding="utf8", mode="w", newline='') as file_csv:
    header = [" ","8/2023", "9/2023", "10/2023", "11/2023", "12/2023", "1/2024", "2/2024", "3/2024", "4/2024", "5/2024", "6/2024", "7/2024"]
    writer = csv.writer(file_csv)
    writer.writerow(header) # Write header into csv file
        
    for i, order_plan in enumerate(order_plans, start=1):
        order_plan = order_plan.astype(object)
        updated_order_plan = np.insert(order_plan, 0, f"Item {i}")
        writer.writerow(updated_order_plan)

# Write total_cost into csv file:
with open("Output_Total_cost_using_MILP.csv", encoding="utf8", mode="w", newline='') as file_csv:
    writer = csv.writer(file_csv)  # Create a CSV writer
    for i, total_cost in enumerate(total_costs, start=1):  
        updated_total_cost = [f"Item {i}", total_cost]  
        writer.writerow(updated_total_cost) 
       