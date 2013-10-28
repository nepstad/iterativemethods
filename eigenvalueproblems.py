from numpy import *

def RayleighQuotient(A, x):
    """
    Compute the Rayleigh quotient x**T * A * x / ||x||**2
    """

    return dot(x, dot(A, x)) / linalg.norm(x)**2


def PowerIterations(A, x, tol = 1e-6, maxIterations = 100000):
    """
    Perform power iterations to estimate the largest eigenvalue of A and its eigenvector.
    """

    #Normalize x
    w = x.copy() / linalg.norm(x)

    terminate = False
    eigValEstimateNew = 0
    eigValEstimateOld = 0
    numIterations = 0
    while not terminate:
        numIterations += 1

        #Calculate next power of A, A**n * x
        w = dot(A, w)

        #Normalize
        w /= linalg.norm(w)

        #Estimate eigenvalue
        eigValEstimateOld = eigValEstimateNew
        eigValEstimateNew = dot(w, dot(A, w))

        #Termination condition
        if abs(eigValEstimateNew - eigValEstimateOld)**2 < tol:
            print "Convergence reached in %s iteration" % numIterations
            terminate = True

        if numIterations > maxIterations:
            print "Maximum number of iterations reached!"
            terminate = True

    print "Eigenvalue estimate = %s" % eigValEstimateNew

    return numIterations, eigValEstimateNew, w


def RayleighQuotientIterations(A, x, tol = 1e-5, maxIterations = 1e5, method="Direct", \
    EigenvalueEstimator = RayleighQuotient):
    """
    Perform Rayleigh quotient iterations to estimate the eigenvalue of A
    closest to x**T * A * x / ||x|| and its eigenvector.
    """

    def DirectSolver(A, w, **args):
        return linalg.solve(A, w)

    #Normalize x
    w = x.copy() / linalg.norm(x)

    I = diag(ones(shape(A)[0]))

    terminate = False
    eigValEstimateNew = EigenvalueEstimator(A, w)
    eigValEstimateOld = 0
    numIterations = 0
    oldEigvec = w.copy()
    while not terminate:
        numIterations += 1
        #print linalg.cond(A - eigValEstimateNew * I)

        #Solve inverse iteration step
        oldEigvec[:] = w[:]
        if method == "Direct":
            w = linalg.solve(A - eigValEstimateNew * I, w)
        else:
            w = GMRES(A - eigValEstimateNew * I, w, krylovSize = 5)

        #Normalize
        w /= linalg.norm(w)

        #Update eigenvalue estimate
        eigValEstimateOld = eigValEstimateNew
        eigValEstimateNew = EigenvalueEstimator(A, w)

        #Termination condition
        #if abs(eigValEstimateNew - eigValEstimateOld)**2 < tol:
        if linalg.norm(w - oldEigvec) < tol or linalg.norm(w + oldEigvec) < tol:
            print "Convergence reached in %s iteration" % numIterations
            terminate = True

        if numIterations > maxIterations:
            print "Maximum number of iterations reached!"
            finalError = numpy.min([linalg.norm(w + oldEigvec), linalg.norm(w - oldEigvec)])
            print "Error in final iteration ||x* - x|| = %s" % finalError
            terminate = True

    print "Eigenvalue estimate = %s" % eigValEstimateNew

    return numIterations, eigValEstimateNew, w


def QRAlgorithm():
    pass


def TestPowerIterations(matSize = 100, tol = 1e-15, ProblemMatrix = ProblemEasy, \
    maxIterations = 1e5, **args):
    #Setup problem matrix
    A = ProblemMatrix(matSize, **args)

    #Create random start vector
    x = random(matSize)

    #Find largest eigenvalues by power iterations
    numIterations, eigVal, eigVec = PowerIterations(A, x, tol = tol, maxIterations = maxIterations)

    #Find "exact" eigenvalues
    E, V = linalg.eig(A)
    I = argsort(E)
    eigValExact = E[I[-1]]
    eigVecExact = V[:,I[-1]]

    #Error
    relErrorEigVal = abs(eigValExact - eigVal) / abs(eigValExact)
    signDiff = sign(eigVec[0] / eigVecExact)
    relErrorEigVec = linalg.norm(eigVec - signDiff * eigVecExact) / linalg.norm(eigVecExact)
    print "Relative error in eigenvalue = %s" % relErrorEigVal
    print "Relative error in eigenvector = %s" % relErrorEigVec
    print "Expected eigenvalue accuracy, O(|l2/l1|**2k) = %s" \
        % abs(E[I[-2]]/eigValExact)**(2*numIterations)
    print "Expected eigenvector accuracy, O(|l2/l1|**k) = %s" \
        % abs(E[I[-2]]/eigValExact)**(numIterations)

    semilogy(E[I], "x")
