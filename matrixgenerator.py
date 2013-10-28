from numpy import *
import numpy as np

def FiniteDifferenceMatrixModelAtom(probSize = 3000, mass = 1.0, xmin=-200, xmax=200):
    """
    Finite difference matrix for 1D model atom: V(x) = - 1/(1+x**2),

    H = -1/2m * d**2/dx**2 + V(x)
    """
    #problem matrix: H = p**2/2 + V(x)
    H = zeros((probSize, probSize))

    #create grid
    gridSize = probSize
    grid = linspace(xmin, xmax, gridSize)
    dx = grid[1] - grid[0]

    #kinetic energy term -1/2 * d2/dx2
    H[:] =  -1.0/(2 * mass) * (-2 * eye(probSize, k=0) + eye(probSize, k=-1) + eye(probSize, k=1)) / dx**2

    #potential term
    potential = diag( -1.0 / (1 + grid**2) )
    #potential = diag( 0.5 * grid**2 )
    H[:] = H[:] + potential[:]

    #Eigenstate guess
    b = exp(-grid**2)

    return H, b, grid


def ProblemEasy(matSize, **args):
    """
    Create random symmetric matrix B
    """
    A = np.random.random((matSize, matSize))
    B = np.dot(A.T, A)

    return B


def ProblemHard(matSize, **args):
    """
    Create random diagonal matrix with small gap
    """
    gapSize = args["gapSize"]
    A = 2 * diag(ones(matSize))
    A[-1,-1] += gapSize

    return A


def BandedSymmetricMatrix(matSize, bandNum):
    """
    Symmetric banded matrix
    """
    A = zeros((matSize, matSize))
    for k in range(bandNum):
        A += diag(random(matSize-k), k=k)
        A += diag(random(matSize-k), k=-k)

    return A


def BandedSymmetricPositiveDefiniteMatrix(matSize, bandNum):
    """
    Symmetric banded matrix
    """
    B = random((matSize, matSize))
    Q, R = linalg.qr(B)
    D = diag(random(matSize)) * 0.01
    A = dot(Q, dot(D, transpose(Q)))

    AA = diag(diag(A))
    for k in range(1,bandNum):
        AA += diag(diag(A, k=k), k=k)
        AA += diag(diag(A, k=-k), k=-k)

    return AA
