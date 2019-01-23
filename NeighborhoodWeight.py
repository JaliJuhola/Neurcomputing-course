from typing import Union, Iterable, Tuple, Any
from scipy.spatial import distance
import numpy as np
import math

def getHighestWeightEuclidian(weigths: Iterable[Tuple[float, float, float]], data: Tuple[float, float, float]) -> Any:
    smallest_distance = float("inf")
    result_tuple = None
    for weigth in weigths:
        weigth_distance = distance.euclidean(weigth, data)
        if weigth_distance < smallest_distance:
            smallest_distance = weigth_distance
            result_tuple = weigth
    
    return result_tuple, smallest_distance

def gaussianDistance(winner: Tuple[float, float, float], data: Tuple[float, float, float], repeat_count: int) -> float:
    vector_diff = np.array(winner) - np.array(data)
    vectors_normalized_power_two =  np.power(np.linalg.norm(vector_diff), 2)
    divide_by_two_and_complement = vectors_normalized_power_two / (2 * repeat_count)
    result = np.power(math.e, -divide_by_two_and_complement)
    return result
    
def mexicanHatNeighborhood(winner: Tuple[float, float, float], data: Tuple[float, float, float], repeat_count: int) -> float:
    vector_diff = np.array(winner) - np.array(data)
    vectors_normalized_power_two = np.power(np.linalg.norm(vector_diff), 2)
    divide_by_two_and_complement = (vectors_normalized_power_two / (repeat_count))
    result = (1 - divide_by_two_and_complement) * gaussianDistance(winner=winner, data=data, repeat_count=repeat_count)
    return result

"""
Task 2
Let us consider that we have an SOM with only two nodes n1 and n2. Node n1 is weighted with
weight vector w1=(1,2,3) and node n2 is weighted with weight vector w2=(-1,-3,6). Search for the
winning node given a data vector x1=(2,-1,-2).
"""
data = (2,-1,-2)
weigths = [(1,2,3), (-1,-3,6)]
winner, value = getHighestWeightEuclidian(weigths=weigths, data=data)
print("Winner of first task {winner} with value {value}".format(winner=winner, value=value))
"""
Task 3
Center a Gaussian neighborhood function on the winner node of task 2. How many iterations does
it take when the strength of neighborhood falls under 10% given that a data vector is x2=(1,1,1).
Gaussian neighborhood function in iteration t is
𝑓(𝒙, 𝒄,𝑡) = 𝑒−||𝒙−𝒄||22𝑡 .
Initial strength is calculated at iteration t=1. Recall that ||x|| is the norm or length of a vector x
"""
new_data = (1,1,1)
i = 1
gaussian_distance = 0
while gaussian_distance < 0.9:
    gaussian_distance = gaussianDistance(winner=winner, data=new_data, repeat_count=i)
    if gaussian_distance < 0.9:
        i = i + 1

print("Under 10 % neighborhood ({distance}) with try {try_count}".format(distance=gaussian_distance, try_count=i))

"""
Use the winning node from task 2 and data vector x2 from task 3 and calculate how many iterations
it takes when the strength of neighborhood changes into negative using Mexican Hat neighborhood
function
𝑓(𝒙, 𝒄,𝑡) = (1 −||𝑥 − 𝑐||2𝑡) 𝑒−||𝒙−𝒄||2

"""
i = 1
mexican_hat_neighborhood = 0
while mexican_hat_neighborhood >= 0:
    mexican_hat_neighborhood = mexicanHatNeighborhood(winner=winner, data=new_data, repeat_count=i)
    if mexican_hat_neighborhood >= 0:
        i = i + 1

print("negative value from mexicanhat neighborhood ({mexican_hat_neighborhood}) with try {try_count}".format(mexican_hat_neighborhood=mexican_hat_neighborhood, try_count=i))