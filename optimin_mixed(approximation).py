
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


if __name__ == "__main__":
    start_time = time.time()
    # Enter a degree of precision for approximate mixed-strategy extension. k=1 produces the game itself, 
    # k=10 produces a 11x11 matrix for a 2x2 matrix. 
    k = 10

    # Example matrices
    zerosum1 = Matrix([[0,2], 
                       [3,1]])
    zerosum2 = Matrix([[0,-2], 
                       [-3,-1]])

    """
    The game matrix below is as follows: 
    3,2 0,3
    2,2 -1,0
    """
    S1 = Matrix([[3, 0], 
                  [2, -1]])
    S2 = Matrix([[2, 3], 
                  [2,  0]])
    # 3x3 Illustrative Example from the optimin paper
    EP1 = Matrix([[100, 100,   0],
                  [105,  95,   0],
                  [  0, 210,   1]])
    EP2 = Matrix([[100, 105,   0],
                  [100,  95, 210],
                  [  0,   0,   1]])
    
    # Example usage with a mixed strategy matrix
    U_1, U_2 = S1, S2
    U_1, U_2 = generate_matrix2by2(U_1, U_2, k)
    """
    print('The game matrix is:')
    print("U_1 =")
    print(U_1)
    print("U_2 =")
    print(U_2)
    """
    print('Mixed strategy matrix:' , '%sx%s' %(len(U_1.row(0)), len(U_1.col(0))))
    pi_1, pi_2 = performance_function(U_1, U_2)
    frontier = pareto(pi_1, pi_2)
    coords = optimin(U_1, U_2, pi_1, pi_2)

    print("Pareto optimal performance values (pi_1, pi_2):", frontier)
    for (x, y) in coords:
        print(f"Optimin coordinate: (x={x}, y={y}), original payoff = ({U_1[x-1,y-1]}, {U_2[x-1,y-1]})")
    print("Done. Computation took %.6f seconds" % (time.time() - start_time))
