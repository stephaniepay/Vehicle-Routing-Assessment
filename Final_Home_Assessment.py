#!/usr/bin/env python
# coding: utf-8

# # MODE FAIR HOME ASSESSMENT

# In[1]:


import random
import pandas as pd
import numpy as np
from copy import deepcopy


# #### Data Preparation

# In[2]:


customer_df = {
    "Customer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Latitude": [4.3555, 4.3976, 4.3163, 4.3184, 4.4024, 4.4142, 4.4804, 4.3818, 4.4935, 4.4932],
    "Longitude": [113.9777, 114.0049, 114.0764, 113.9932, 113.9896, 114.0127, 114.0734, 114.2034, 114.1828, 114.1322],
    "Demand": [5, 8, 3, 6, 5, 8, 3, 6, 5, 8]
}

depot = (4.4184, 114.0932)

customers_df = pd.DataFrame(customer_df)
customers_df


# In[3]:


vehicle_df = {
    "Vehicle": ["Type A", "Type B"],
    "Capacity": [25, 30],
    "Cost": [1.2, 1.5]
}

depot = (4.4184, 114.0932)

vehicles_df = pd.DataFrame(vehicle_df)
vehicles_df


# #### Calculate travel distance between each customer location and depot

# In[4]:


def euclidean_distance(lat1, lon1, lat2, lon2):
    return 100 * np.sqrt( (lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)


# In[5]:


num_locations = len(customers_df) + 1

distance_matrix = np.zeros((num_locations, num_locations))

for i in range(num_locations):
    for j in range(num_locations):
        
        if i == 0:  #depart from depot
            lat1, lon1 = depot
        else:
            lat1, lon1 = customers_df.iloc[i-1, 1], customers_df.iloc[i-1, 2]
        
        if j == 0:  #arrive depot
            lat2, lon2 = depot
        else:
            lat2, lon2 = customers_df.iloc[j-1, 1], customers_df.iloc[j-1, 2]
        
        distance_matrix[i, j] = euclidean_distance(lat1, lon1, lat2, lon2)

distance_matrix_df = pd.DataFrame(distance_matrix, columns=['Depot'] + [f'C{i}' for i in customer_df["Customer"]], 
                                  index=['Depot'] + [f'C{i}' for i in customer_df["Customer"]])

distance_matrix_df


# #### Generate initial solution

# In[6]:


def generate_initial_solution(num_customers, num_vehicles, seed=None):
    
    if seed is not None:
        random.seed(seed)
        
    customers = list(range(1, num_customers + 1))
    random.shuffle(customers)
    
    solution = [[] for _ in range(num_vehicles)]
    
    for i, customer in enumerate(customers):
        vehicle_index = i % num_vehicles  #roundrobin  
        solution[vehicle_index].append(customer)
    return solution


# #### Generate neighboring solutions

# In[7]:


def generate_neighbors(solution, vehicle_capacities):
    neighbors = []

    for i in range(len(solution)):
        for j in range(len(solution)):
            
            # If vehicles are different, first vehicle is not empty, second vehicle is not empty
            if i != j and solution[i] and solution[j]: 
                
                new_solution = deepcopy(solution)
                new_solution[i].append(new_solution[j].pop(0))

                workable = True
                
                for route in new_solution:
                    route_demand = sum(customers_df.iloc[customer - 1]['Demand'] for customer in route)
                    
                    if route_demand > vehicle_capacities[0] and route_demand > vehicle_capacities[1]:
                        workable = False
                        break

                if workable:
                    neighbors.append(new_solution)
    
    return neighbors


# #### Calculate total cost for each solution

# In[8]:


def calculate_cost(solution, distance_matrix, vehicle_costs, vehicle_capacities):
    total_distance = 0
    total_cost = 0

    for route in solution:
        if route:
            
            route_distance = 0
            route_demand = 0
            route_cost = 0

            route_distance += distance_matrix[0, route[0]]  #Depot -> First customer

            for i in range(len(route) - 1):
                route_distance += distance_matrix[route[i], route[i + 1]] 
                route_demand += customers_df.iloc[route[i] - 1]['Demand']

            route_distance += distance_matrix[route[-1], 0]  #Last cusomter -> Depot
            route_demand += customers_df.iloc[route[-1] - 1]['Demand']


            
            if len(solution) == 2:  #use Type B when there is only 2 vehicles
                route_cost = route_distance * vehicle_costs[1] 
                
            else:
                for vehicle in vehicles_df.itertuples():
                    if route_demand <= vehicle.Capacity:
                        route_cost = route_distance * vehicle.Cost
                        break

            total_distance += route_distance
            total_cost += route_cost

    return total_cost, total_distance


# #### Tabu Search algorithm

# In[9]:


def tabu_search(distance_matrix, vehicle_costs, vehicle_capacities, num_customers, num_vehicles, 
                        max_iterations=200, seed=None):
    
    current_solution = generate_initial_solution(num_customers, num_vehicles, seed)
    best_solution = current_solution
    best_cost, _ = calculate_cost(best_solution, distance_matrix, vehicle_costs, vehicle_capacities)
    
    tabu_list = []
    iteration_costs = []

    for iteration in range(max_iterations):
        
        neighbors = generate_neighbors(current_solution, vehicle_capacities)
        
        if not neighbors:
            break

        best_neighbor = neighbors[0]
        best_neighbor_cost, _ = calculate_cost(best_neighbor, distance_matrix, vehicle_costs, vehicle_capacities)

        for neighbor in neighbors:
            neighbor_cost, _ = calculate_cost(neighbor, distance_matrix, vehicle_costs, vehicle_capacities)
            
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost

        if best_neighbor_cost < best_cost:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost

        tabu_list.append(current_solution)
        
        if len(tabu_list) > 10:
            tabu_list.pop(0)

        current_solution = best_neighbor
        iteration_costs.append(best_cost)
        print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

    return best_solution, best_cost, iteration_costs


# #### Run Tabu Search multiple times to find the best solution

# In[10]:


def find_best_solution(num_runs, distance_matrix, vehicle_costs, vehicle_capacities, num_customers, num_vehicles, max_iterations):
    
    best_overall_solution = None
    best_overall_cost = float('inf')
    best_iteration_costs = None

    for run in range(num_runs):
        print(f"Run {run + 1}")
        
        best_solution, best_cost, iteration_costs = tabu_search(
            distance_matrix, vehicle_costs, vehicle_capacities, num_customers, num_vehicles, max_iterations, seed=run
        )
        
        if best_cost < best_overall_cost:
            best_overall_solution = best_solution
            best_overall_cost = best_cost
            best_iteration_costs = iteration_costs

    return best_overall_solution, best_overall_cost, best_iteration_costs






num_vehicles = 5
vehicle_costs = vehicles_df['Cost'].values
vehicle_capacities = vehicles_df['Capacity'].values

best_solution, best_cost, iteration_costs = find_best_solution(
    num_runs=20, distance_matrix=distance_matrix, vehicle_costs=vehicle_costs, 
    vehicle_capacities=vehicle_capacities, num_customers=len(customers_df), 
    num_vehicles=num_vehicles, max_iterations=100
)

print("Best Overall Solution:", best_solution)
print("Best Overall Cost:", best_cost)


# #### Summarize and Print the Best Solution
# 

# In[11]:


def summarize_solution(solution, distance_matrix, vehicle_costs, vehicle_capacities):
    
    summary = []
    total_cost = 0
    total_distance = 0
    vehicle_type_map = {0: 'Type A', 1: 'Type B'}

    for route_index, route in enumerate(solution):
        
        if route:
            route_distance = 0
            route_demand = 0
            route_cost = 0
            vehicle_type = None

            route_distance += distance_matrix[0, route[0]]
            route_demand += customers_df.iloc[route[0] - 1]['Demand']

            for i in range(len(route) - 1):
                route_distance += distance_matrix[route[i], route[i + 1]]
                route_demand += customers_df.iloc[route[i + 1] - 1]['Demand']

            route_distance += distance_matrix[route[-1], 0]

            
            if len(solution) == 2:  #use Type B when there is only 2 vehicles
                route_cost = route_distance * vehicle_costs[1]  
                vehicle_type = 'Type B'
                
            else:
                for vehicle in vehicles_df.itertuples():
                    
                    if route_demand <= vehicle.Capacity:
                        route_cost = route_distance * vehicle.Cost
                        vehicle_type = vehicle.Vehicle
                        break

            total_distance += route_distance
            total_cost += route_cost

            route_summary = f"Depot"
            
            for i in range(len(route)):
                if i == 0:
                    route_summary += f" -> C{route[i]} ({distance_matrix[0, route[i]]:.3f} km)"
                else:
                    route_summary += f" -> C{route[i]} ({distance_matrix[route[i - 1], route[i]]:.3f} km)"
            
            route_summary += f" -> Depot ({distance_matrix[route[-1], 0]:.3f} km)"

            
            summary.append({
                "Vehicle": vehicle_type,
                "Route": route_summary,
                "Round Trip Distance": route_distance,
                "Cost": route_cost,
                "Demand": route_demand
            })
            

    return summary, total_distance, total_cost






solution_summary, total_distance, total_cost = summarize_solution(best_solution, distance_matrix, 
                                                                  vehicle_costs, vehicle_capacities)

print(f"Total Distance = {total_distance:.2f} km")
print(f"Total Cost = RM {total_cost:.2f}\n")

for i, vehicle_summary in enumerate(solution_summary, 1):
    print(f"Vehicle {i} ({vehicle_summary['Vehicle']}):")
    
    print(f"Round Trip Distance: {vehicle_summary['Round Trip Distance']:.3f} km, Cost: RM {vehicle_summary['Cost']:.2f}, Demand: {vehicle_summary['Demand']}")
    
    print(vehicle_summary['Route'] + "\n")


# In[12]:


get_ipython().system('jupyter nbconvert --to script Final_ModeFair_Assessment.ipynb')


# In[ ]:




