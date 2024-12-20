import pandas as pd
import numpy as np
import csv

# Load input data from Excel files
demand = pd.read_excel("Input_Demand.xlsx", header=None).values.astype(float)  
purchase_cost = pd.read_excel("Input_PurchaseCost.xlsx", header=None).values.astype(float)  
weights = pd.read_excel("Input_Weightperbox.xlsx", header=None).values.astype(float)  

# Define holding cost rate
holding_cost_rate = 0.0021 #VND

# Calculate holding costs (per unit)
holding_cost = purchase_cost * holding_cost_rate

# Define ordering cost based on weight ranges
def get_order_cost(weight): 
    if weight > 8000:
        return 8500
    elif 5000 <= weight <= 8000:
        return 7300
    elif 3000 <= weight < 5000:
        return 6000
    else:
        return 4500

# Wagner-Whitin Algorithm for Lot Sizing
def wagner_whitin(demand, purchase_cost, holding_cost, weights):
    num_products, num_months = demand.shape  
    order_plans = []  # To store order plans for each product
    total_costs = []  # To store total costs for each product

    for product in range(num_products):
        weight = weights[product, 0] # Extracting weight of a specific product from weight list  
        order_cost_per_order = get_order_cost(weight) 

        # Cost matrix for dynamic programming
        total_cost = np.full((num_months, num_months), float('inf')) 
        # Precompute costs for ordering at month `start` to cover demand until month `end`.
        for start in range(num_months): 
            cumulative_demand = 0 # The total demand from month 'start' to month 'end'.
            for end in range(start, num_months): 
                cumulative_demand += demand[product, end] # Add demand of month 'end' to cumulative demand.
                
                # Holding Cost Calculation 
                holding_cost_accum = 0                
                for k in range(start,end): # k is intermediate months between 'start' month to 'end' month 
                    holding_cost_accum += holding_cost[product,k] * sum(demand[product, k + 1:end + 1]) 
                total_cost[start, end] = (
                    cumulative_demand * order_cost_per_order  # Ordering cost
                    + cumulative_demand * purchase_cost[product, start]  # Purchase cost
                    + holding_cost_accum  # Holding cost
                )

        # Dynamic programming to find the minimum cost
        dp = [float('inf')] * num_months  # Create a cost list with 'num_months' elements, all initialized to 'inf' 
        dp[0] = total_cost[0, 0] 
        backtrack = [-1] * num_months  # For reconstructing the solution

        for t in range(1, num_months):
            for s in range(t + 1): 
                cost = (dp[s - 1] if s > 0 else 0) + total_cost[s, t]  
                if cost < dp[t]: # Check if the total cost from month s to month t is less than the current minimum cost in month t stored in dp[t] list or not
                    dp[t] = cost # If true, that total cost from month s to month t is updated.
                    backtrack[t] = s # Record the start month s of the ordering period that leads to the minimum total cost at month t

        # Calculate order quantity for recorded months
        order_plan = np.zeros(num_months, dtype=int) 
        t = num_months - 1 # Start the backtracking process from the last month (t=11 for 12-month plan)
        while t >= 0: 
            s = backtrack[t] # Indicate that an order is placed in month 's'.
            order_plan[s] = sum(demand[product, s:t + 1])  # Calculate the sum of demand from month 's' to month 't', then assign this value to order_plan[s], indicating the order quantity placed in month 's'.
            t = s - 1 # Update month 't' to point to the month before month 's', moving backward in time.

        order_plans.append(order_plan)
        total_costs.append(dp[-1])  # Store total cost for the product

    return order_plans, total_costs

# Solve the lot-sizing problem for all products
order_plans, total_costs = wagner_whitin(demand, purchase_cost, holding_cost, weights)

# Display results
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
print(f"Total cost for planning horizon using W-W is: {sum(total_costs)}")
print("***")

# Write order_plans into csv file:
with open("Output_Order_plan_using_WW.csv", encoding="utf8", mode="w", newline='') as file_csv:
    header = [" ","8/2023", "9/2023", "10/2023", "11/2023", "12/2023", "1/2024", "2/2024", "3/2024", "4/2024", "5/2024", "6/2024", "7/2024"]
    writer = csv.writer(file_csv)
    writer.writerow(header) 

    for i, order_plan in enumerate(order_plans, start=1):
        order_plan = order_plan.astype(object)
        updated_order_plan = np.insert(order_plan, 0, f"Item {i}")
        writer.writerow(updated_order_plan)

# Write total_cost into csv file:
with open("Output_Total_cost_using_WW.csv", encoding="utf8", mode="w", newline='') as file_csv:
    writer = csv.writer(file_csv)  
    for i, total_cost in enumerate(total_costs, start=1):  
        updated_total_cost = [f"Item {i}", total_cost]  
        writer.writerow(updated_total_cost) 
       

