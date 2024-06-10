# Vehicle-Routing-Assessment
## Background
You are a logistics manager for a delivery company tasked with optimizing the routing of a fleet of vehicles to efficiently deliver goods to various customer locations. Your goal is to optimize the delivery cost while ensuring that all delivery locations are visited and all demands are met.

## Task
Your task is to develop an algorithm or program that attempt to find the best route for a fleet of vehicles of different types so that the deliveries are completed at the lowest cost while satisfying all hard constraints. Ensure that your solution is scalable to support a larger number of customers beyond the provided test data.

## Requirements
- Hard Constraint: Each delivery location must be visited exactly once. The total demand of each vehicle route must not exceed its maximum capacity.
- Soft Constraint: Minimize cost required to meet all demands.

## Assumptions
- The vehicles start and end their routes at the same depot location.
- Each vehicle only travels one round trip. (depart from depot and back to the depot)
- There is no limit on the number of vehicles.
- Travel times between any two locations are the same in both directions.
- Deliveries can be made at any time, there are no time windows for deliveries.
- Vehicle travel distance is calculated using Euclidean distance formula:
- Distance in km = 100 * √((Longitude2-Longitude1)^2 + (Latitude2-Latitude1)^2)

Test Data: One Depot, 10 Customer, 2 types of vehicles.
Depot: (Latitude = 4.4184, Longitude = 114.0932)

## Outputs
### Customer Data
<img width="294" alt="Customer Data" src="https://github.com/stephaniexxx/Vehicle-Routing-Assessment/assets/76270106/5979503e-0f75-4659-871e-af5e80997041">

### Vehicle Data
<img width="190" alt="Vehicle Data" src="https://github.com/stephaniexxx/Vehicle-Routing-Assessment/assets/76270106/21b8b9a1-4761-4eff-8136-ba4377f40777">

### Distance Matrix
<img width="842" alt="Tabu Search Run" src="https://github.com/stephaniexxx/Vehicle-Routing-Assessment/assets/76270106/2a0eb077-d65f-4cef-b5bb-4dbe58ec6fc5">

### Tabu Search Algorithm Iterations
<img width="701" alt="Tabusearch Run" src="https://github.com/stephaniexxx/Vehicle-Routing-Assessment/assets/76270106/6cbca2b8-431a-4936-a47e-004854ec5785">

### Final Output
<img width="767" alt="Final Output" src="https://github.com/stephaniexxx/Vehicle-Routing-Assessment/assets/76270106/bc0f10f0-940b-43be-9283-868763e56e3c">
