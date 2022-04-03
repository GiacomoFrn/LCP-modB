import numpy as np



# function for the random step, using lambda construction
# int() for cleaner look and for mimiking a detector with finite resolution
def jump(
    drift,
    stddev
): 
    return int(np.random.normal(drift,stddev))

def pattern(i,z,a):
    return int(a*np.sin((np.pi*i)/z))

def generate_data(
    A    = 500,
    Z    = 12,
    N    = 10000,
    L    = 60,
    DX   = 50,
    bias = 5,
    verbose = False,
    seed = 12345
    ):
    # random seed for reproducibility
    np.random.seed(seed)

    y = np.zeros(N)
    x = np.zeros((N,L))                # each sample (tot=N) has L elements, shape->[N][L]
    
    for i in range(N):
        ## if first element of the sample, jump from the last element of the previous sample
        if i>0:
            x[i][0] = x[i-1][-1] + jump(bias,DX)
        
        ## for the others jump with "discrete continuity" inside a sample
        for j in range(1,L):
            x[i][j] = x[i][j-1] + jump(bias,DX)

        ## assign a label (0,1,2) to each sample
        y[i] = i%3 
        #y[i] = random.randint(0,2)

        if y[i]>0:
            ### sort the starting index of the pattern inside the sample
            j0 = np.random.randint(0,L-1-Z)
            #print(i,j0,j1)

            ### create the pattern following(pattern*sign)
            sign = 3-2*y[i]                        # sign = +-1
            for j in range(Z):
                x[i][j0+j] += sign*pattern(j,Z,A)
               
    # save file
    str0 = f'ts_L{L}_Z{Z}_A{A}_DX{DX}_bias{bias}_N{N}.dat'
    if verbose:
        print(str0)

    fname='DATA/x_'+str0
    np.savetxt(fname,x,fmt="%d")
    fname='DATA/y_'+str0
    np.savetxt(fname,y,fmt="%d")
        
    return str0
