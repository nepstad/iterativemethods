from numpy import linalg as linalg
from numpy import finfo as finfo
from numpy import *
import  numpy as np

class ArnoldiIterations:

    def __init__(self, A, multiplyMatrix, krylovSize):
        self.MultiplyMatrix = multiplyMatrix
        self.Matrix = A
        self.NumRows = shape(A)[0]
        self.NumCols = shape(A)[1]
        self.KrylovSize = krylovSize
        self.EPS = finfo(double).eps ** (2.0 / 3.0)
        #self.EPS = 1e-6

    def Setup(self, startVector = []):
        #Random start vector if not specified
        b = startVector[:]
        if len(startVector) == 0:
            b = random(self.NumRows)

        #Setup rectangular matrix Q
        self.ArnoldiVectors = zeros((self.NumRows, self.KrylovSize+1), dtype=double)

        #Setup hessenberg matrix H
        self.Hessenberg = zeros((self.KrylovSize+1, self.KrylovSize), dtype=double)

        #Normalize start vector
        self.NextArnoldiVector = b / linalg.norm(b)

        #Breakdown flag
        self.Breakdown = False
        self.BreakdownStep = self.KrylovSize

    def ArnoldiIterations(self):
        for i in range(self.KrylovSize):
            self.ArnoldiStep(i)
            if self.Breakdown:
                self.BreakdownStep = i+1
                break

    def ArnoldiStep(self, stepNum):
        self.ArnoldiVectors[:,stepNum] = self.NextArnoldiVector[:]

        #Compute next krylov vector A**i * b
        #v = dot(self.Matrix, self.NextArnoldiVector)
        v = self.MultiplyMatrix(self.NextArnoldiVector)

        for j in range(stepNum+1):
            #Compute next element in H from projection of v on q_j,
            #i.e. q_j* * A * q_i
            self.Hessenberg[j,stepNum] = dot(self.ArnoldiVectors[:,j], v)

            #Remove current projection from v (orthonorm. on prev. q's)
            v[:] =  v[:] - self.Hessenberg[j,stepNum] * self.ArnoldiVectors[:,j]

        #Double orthogonalization
        for j in range(stepNum+1):
            #Compute next element in H from projection of v on q_j,
            #i.e. q_j* * A * q_i
            proj = dot(self.ArnoldiVectors[:,j], v)

            #Add new projection to Hessenberg matrix
            self.Hessenberg[j, stepNum] += proj

            #Remove current projection from v (orthonorm. on prev. q's)
            v[:] =  v[:] - proj * self.ArnoldiVectors[:,j]

        #if stepNum < self.KrylovSize-1:
        self.Hessenberg[stepNum+1, stepNum] = linalg.norm(v)
        if abs(self.Hessenberg[stepNum+1, stepNum]) < self.EPS:
            print "Breakdown!"
            self.Breakdown = True

        self.NextArnoldiVector[:] = v[:] / self.Hessenberg[stepNum+1, stepNum]


def GMRES(A, b, krylovSize=10, useQR = True):
    def MultiplyMatrix(x):
        return dot(A, x)

    arnoldi = ArnoldiIterations(A, MultiplyMatrix, krylovSize)
    arnoldi.Setup(startVector = b)
    arnoldi.ArnoldiIterations()

    #converged = False
    #while not converged:
        #arnoldi step

        #check residual

    #Solve least square problem
    x = None
    bdStep = arnoldi.BreakdownStep
    if useQR:
        Q,R = linalg.qr(arnoldi.Hessenberg[:bdStep+1,:bdStep])
        Qb = dot(transpose(arnoldi.ArnoldiVectors[:,:bdStep+1]), b)
        Qbb = dot(transpose(Q), Qb)
        y = linalg.solve(R[:bdStep+1,:bdStep], Qbb)
        x = dot(arnoldi.ArnoldiVectors[:,:bdStep], y)
    else:
        HH = dot(transpose(arnoldi.Hessenberg), arnoldi.Hessenberg)
        bb = dot(transpose(arnoldi.Hessenberg), dot(transpose(arnoldi.ArnoldiVectors), b))
        y = linalg.solve(HH, bb)
        x = dot(arnoldi.ArnoldiVectors[:,:-1], y)

    return x


def FindEigenvaluesArnoldi(A, krylovSize=10, shift = 0, inverseIterations = False):
    def MultiplyMatrix(x):
        if inverseIterations:
            return solve(A - shift * eye(shape(A)[0]), x)
        else:
            return dot(A, x)

    arnoldiIterator = ArnoldiIterations(A, MultiplyMatrix, krylovSize=krylovSize)
    arnoldiIterator.Setup()
    arnoldiIterator.ArnoldiIterations()
    ritzVals, ritzVecs = linalg.eig(arnoldiIterator.Hessenberg[:-1,:])
    eigVecs = dot(arnoldiIterator.ArnoldiVectors[:,:-1], ritzVecs)

    if inverseIterations:
        ritzVals = 1.0 / ritzVals + shift

    return ritzVals, eigVecs, arnoldiIterator


def TestArnoldiIterations(A, krylovSize = 10, numEigs = 2):
    eigValArnoldi, eigVecArnoldi, AI = FindEigenvaluesArnoldi(A, krylovSize=krylovSize)
    E, V = eig(A)
    eigValArnoldiSorted = reversed(sort(eigValArnoldi.real))
    eigValExactSorted = reversed(sort(E.real))
    eigError = [abs(e1-e2) for e1,e2 in zip(eigValArnoldiSorted, eigValExactSorted)]

    return eigError[:numEigs], AI


def TestGMRESSimple():
    probSize = 2000
    A = 2*eye(probSize) + 0.5 * np.random.standard_normal((probSize,probSize)) / sqrt(probSize)
    b = ones(probSize)
    print "Condition number of A = %s" % np.linalg.cond(A)

    #Exact numerical solution
    print "Finding exact solution..."
    exactSol = np.linalg.solve(A, b)

    #Solve by GMRES
    print "Solving Ax = b with GMRES..."
    residual = []
    error = []
    for ks in range(1,30):
        xx = GMRES(A, b, krylovSize = ks)
        residual += [linalg.norm(dot(A, xx) - b, ord=2) / linalg.norm(b, ord=2)]
        error += [linalg.norm(xx - exactSol, ord=2) / linalg.norm(exactSol, ord=2)]

    return residual, error


def TestGMRES(A, b, preconditioners = [], **args):
    probSize = shape(A)[0]
    print "Condition number of A = %s" % np.linalg.cond(A)

    #Exact solution
    print "Solving Ax = b exactly..."
    exactSol = np.linalg.solve(A, b)

    #Preconditioners
    preA = A.copy()
    preb = b.copy()

    #Apply selected preconditioner
    for pc in preconditioners:
        preA, preb = pc(preA, preb, **args)

    print "Condition number of M**-1 * A = %s" % np.linalg.cond(preA)

    #Solve by GMRES iteration
    print "Solving iteratively..."
    residual = []
    error = []
    for ks in range(1,50,2):
        xx = GMRES(preA, preb, krylovSize = ks)
        Ax = dot(preA, xx)
        normAx = linalg.norm(Ax, ord=2)
        residual += [linalg.norm(preb - Ax, ord=2) / linalg.norm(preb, ord=2)]
        error += [linalg.norm(xx - exactSol, ord=2) / linalg.norm(exactSol, ord=2)]

    return residual, error


class Preconditioners:

    class Jacobi:
        def __call__(self, A, b, **args):
            print "Applying Jacobi preconditioner..."
            #A = A - diag(diag(A)) + eye(probSize)
            invM = diag(1.0 / diag(A))
            preA = dot(invM, A)
            preb = dot(invM, b)
            return preA, preb

    class Block:
        def __call__(self, A, b, **args):
            print "Applying Block preconditioner..."
            blockSize = args["blockSize"]
            probSize = b.size
            invM = zeros(shape(A))
            for k in range(probSize / blockSize):
                curSlice = [slice(k*(blockSize), (k+1)*blockSize)] * 2
                matrixBlock = A[curSlice]
                invM[curSlice] = np.linalg.inv(matrixBlock)
            preA = dot(invM, A)
            preb = dot(invM, b)
            return preA, preb

    class SymmetricBlock:
        def __call__(self, A, b, **args):
            print "Applying Symmetric Block preconditioner..."
            probSize = A.shape[0]
            blockSize = args["blockSize"]
            if mod(probSize, blockSize) != 0:
                raise Exception("Preconditioner block size must divide matrix size!")
            C = zeros(shape(A))
            invC = zeros(shape(A))
            for k in range(probSize / blockSize):
                curSlice = [slice(k*(blockSize), (k+1)*blockSize)] * 2
                matrixBlock = A[curSlice]
                L = linalg.cholesky(matrixBlock)
                C[curSlice] = L[:]
                invC[curSlice] = inv(L)

            preA = dot(invC, A)
            preA = dot(preA, transpose(invC))
            preb = dot(invC, b)
            #return preA, preb, invC, C
            return preA, preb

