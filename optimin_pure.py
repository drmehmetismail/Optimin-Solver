
from sympy import Matrix
import time

def performance_function(u_1, u_2):
    """
    For each cell (x,y), find the 'performance' in the row (for P1)
    and in the column (for P2) subject to possible improvements by the opponent.
    
    Returns two flattened lists, pi_1 and pi_2, each of length row*col:
      - pi_1[i*col + j] is the performance for P1 if the outcome is (x=i, y=j).
      - pi_2[i*col + j] is the performance for P2 if the outcome is (x=i, y=j).
    """
    row = u_1.rows
    col = u_1.cols
    
    pi_1 = []
    for x in range(row):
        for y in range(col):
            val = u_1[x,y]
            # If the opponent (P2) can choose a different column z in the same row x
            # that yields a better payoff for P2 but lowers P1's payoff, do so:
            for z in range(col):
                if u_2[x,z] > u_2[x,y] and u_1[x,z] < val:
                    val = u_1[x,z]
            pi_1.append(val)
    
    pi_2 = []
    for x in range(row):
        for y in range(col):
            val = u_2[x,y]
            # If P1 can choose a different row z in the same column y
            # that yields a better payoff for P1 but lowers P2's payoff, do so:
            for z in range(row):
                if u_1[z,y] > u_1[x,y] and u_2[z,y] < val:
                    val = u_2[z,y]
            pi_2.append(val)
    
    return pi_1, pi_2

def pareto(x_values, y_values):
    """
    Given two lists x_values and y_values of the same length, 
    returns the Pareto frontier assuming we want to MAXIMIZE both x and y.
    
    Steps:
      1) Combine them into (x, y) pairs.
      2) Sort by x descending, then by y descending.
      3) Single pass to collect points that strictly improve y.
    """
    # Combine the two lists into (x, y) pairs
    points = list(zip(x_values, y_values))
    
    # Sort points by X descending, then by Y descending
    points.sort(key=lambda p: (p[0], p[1]), reverse=True)

    pareto_points = []
    best_y = float('-inf')

    # Single pass from largest x to smallest x, 
    # collecting points with strictly larger y
    for x, y in points:
        if y > best_y:
            pareto_points.append((x, y))
            best_y = y

    return pareto_points

def optimin(u_1, u_2, pi_1, pi_2):
    """
    Collect the (x,y) coordinates in the matrix that appear on the 
    Pareto frontier of (pi_1, pi_2). Then return them as a list of (row, col).
    """
    row = u_1.rows
    col = u_1.cols
    
    # Reshape pi_1, pi_2 into matrix form so we can compare them by (x,y)
    Pi_1 = Matrix(row, col, pi_1)
    Pi_2 = Matrix(row, col, pi_2)
    
    # Build the Pareto frontier from pi_1, pi_2 ONCE
    frontier = pareto(pi_1, pi_2)
    # Convert to a set for O(1) membership testing
    frontier_set = set(tuple(pt) for pt in frontier)
    
    coordinates = []
    for x in range(row):
        for y in range(col):
            if (Pi_1[x,y], Pi_2[x,y]) in frontier_set:
                coordinates.append((x+1, y+1))
    return coordinates

if __name__ == "__main__":
    start_time = time.time()

    # Example matrices
    zerosum1 = Matrix([[0,2], [3,1]])
    zerosum2 = Matrix([[0,-2], [-3,-1]])

    """
    The game matrix below is as follows: 
    3,2 0,3
    2,2 -1,0
    """
    SS1 = Matrix([[3, 0], 
                  [2, -1]])
    SS2 = Matrix([[2, 3], 
                  [2,  0]])

    # 3x3 Illustrative Example from the optimin paper
    EP1 = Matrix([[100, 100,   0],
                  [105,  95,   0],
                  [  0, 210,   1]])
    EP2 = Matrix([[100, 105,   0],
                  [100,  95, 210],
                  [  0,   0,   1]])

    # Example usage with a small matrix
    U_1 = SS1
    U_2 = SS2

    """
    print('The game matrix is:')
    print("U_1 =")
    print(U_1)
    print("U_2 =")
    print(U_2)
    """

    pi_1, pi_2 = performance_function(U_1, U_2)
    frontier = pareto(pi_1, pi_2)
    coords = optimin(U_1, U_2, pi_1, pi_2)

    print("Pareto frontier for (pi_1, pi_2):", frontier)
    for (x, y) in coords:
        print(f"Optimin coordinate: (x={x}, y={y}), original payoff = ({U_1[x-1,y-1]}, {U_2[x-1,y-1]})")
    print("Done. Computation took %.6f seconds" % (time.time() - start_time))
