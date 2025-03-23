# Find approximate mixed strategy optimin points in a two-player game
from optimin_pure import optimin, performance_function, pareto
from sympy import Matrix, N
import time

def generate_matrix2by2(U_1, U_2, k):
    # generate approximate mixed strategy sets for 2x2 mixed extension
    Delta = []
    for i in range(k+1):
        if i <=k:
            i = i/k
            Delta.append([i,1-i])
    actions = len(Delta) # = (k+1)*(k+2)/2
    # compute the expected payoffs of mixed strategies as given by Delta.
    def f(i,j):
        p = Delta[i]
        q = Delta[j]
        return p[0]*q[0]*U_1[0,0] + p[0]*q[1]*U_1[0,1] + p[1]*q[0]*U_1[1,0] +  p[1]*q[1]*U_1[1,1]

    def g(i,j):
        p = Delta[i]
        q = Delta[j]
        return p[0]*q[0]*U_2[0,0] + p[0]*q[1]*U_2[0,1] + p[1]*q[0]*U_2[1,0] +  p[1]*q[1]*U_2[1,1]
    u_1 = N(Matrix(actions,actions,f),5)
    u_2 = N(Matrix(actions,actions,g),5)
    return u_1, u_2

def generate_matrix3by3(U_1, U_2, k):
    # generates mixed strategy sets for 3x3 games
    Delta = []
    for i in range(k+1):
        for j in range(k+1):
            if i + j <=k:
                i = i/k
                j = j/k
                Delta.append([i,j,1-i-j])
    actions = len(Delta) # = (k+1)*(k+2)/2
    # compute the expected payoffs of mixed strategies in Delta.
    def f(i,j):
        p = Delta[i]
        q = Delta[j]
        return p[0]*q[0]*U_1[0,0] + p[0]*q[1]*U_1[0,1] + p[1]*q[0]*U_1[1,0] +  p[1]*q[1]*U_1[1,1] +  p[2]*q[2]*U_1[2,2] +  p[1]*q[2]*U_1[1,2] +  p[2]*q[1]*U_1[2,1] +  p[2]*q[0]*U_1[2,0]

    def g(i,j):
        p = Delta[i]
        q = Delta[j]
        return p[0]*q[0]*U_2[0,0] + p[0]*q[1]*U_2[0,1] + p[1]*q[0]*U_2[1,0] +  p[1]*q[1]*U_2[1,1] +  p[2]*q[2]*U_2[2,2] +  p[1]*q[2]*U_2[1,2] +  p[2]*q[1]*U_2[2,1] +  p[2]*q[0]*U_2[2,0]

    # Make the game matrix for each player
    u_1 = N(Matrix(actions,actions,f),5)
    u_2 = N(Matrix(actions,actions,g),5)
    return u_1, u_2


# Example 2x2 game matrices
"""
This 2x2 game is represented by two matrices below.
3,2 0,3
2,2 -1,0
"""
S1 = Matrix([[3, 0], 
                [2, -1]])
S2 = Matrix([[2, 3], 
                [2,  0]])

# Another example: zero-sum game matrices
zerosum1 = Matrix([[0,2], [3,1]])
zerosum2 = Matrix([[0,-2], [-3,-1]])

# 3x3 Illustrative Example from the optimin paper (Figure 1)
EP1 = Matrix([[100, 100,   0],
                [105,  95,   0],
                [  0, 210,   1]])
EP2 = Matrix([[100, 105,   0],
                [100,  95, 210],
                [  0,   0,   1]])

if __name__ == "__main__":
    start_time = time.time()
    # Enter a degree of precision for approximate mixed-strategy extension. k=1 produces the game itself, 
    # k=10 produces a 11x11 matrix for a 2x2 game in pure strategies. 
    k = 10

    
    # Example usage that generates approximate mixed strategy sets for 2x2 games
    Player1_matrix, Player2_matrix = S1, S2
    Player1_matrix, Player2_matrix = generate_matrix2by2(Player1_matrix, Player2_matrix, k)
    """
    print('The game matrix is:')
    print("Player1_matrix =")
    print(Player1_matrix)
    print("Player2_matrix =")
    print(Player2_matrix)
    """
    print('Mixed strategy matrix dimensions:' , '%sx%s' %(len(Player1_matrix.row(0)), len(Player1_matrix.col(0))))
    pi_1, pi_2 = performance_function(Player1_matrix, Player2_matrix)
    frontier = pareto(pi_1, pi_2)
    coords = optimin(Player1_matrix, Player2_matrix, pi_1, pi_2)

    print("For each player, each action is enumerated as 1,2,...:")
    optimin_count = 1
    for (x, y) in coords:
        print(f"Optimin {optimin_count} action profile: (x={x}, y={y}) and its payoff profile = ({Player1_matrix[x-1,y-1]}, {Player2_matrix[x-1,y-1]})")
        optimin_count += 1
    print("Optimin profile performances (pi_1, pi_2):", frontier)
    print("Done. Computation took %.6f seconds" % (time.time() - start_time))
