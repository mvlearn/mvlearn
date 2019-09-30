import numpy

class DiagGaussianKernel(object):
    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma
    
    def __call__(self, X1, X2):

        #if X1.shape[1] != X2.shape[1]:
                    #raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))
                
        K = numpy.exp(- numpy.sum( (X1-X2)**2, 1)/(2*self.sigma**2))
        K = numpy.array(K, ndmin=2).T
        return K

class GaussianKernel(object):
    """
    A class to find gaussian kernel evaluations k(x, y) = exp (-||x - y||^2/2 sigma^2)
    """
    def __init__(self, sigma=1.0):
        """
        Initialise object with given value of sigma >= 0

        :param sigma: kernel width parameter.
        :type sigma: :class:`float`
        """
        self.sigma = sigma

    def __call__(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """
        
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        j1 = numpy.ones((X1.shape[0], 1))
        j2 = numpy.ones((X2.shape[0], 1))

        diagK1 = numpy.sum(X1**2, 1)
        diagK2 = numpy.sum(X2**2, 1)

        X1X2 = numpy.dot(X1, X2.T)

        Q = (2*X1X2 - numpy.outer(diagK1, j2) - numpy.outer(j1, diagK2) )/ (2*self.sigma**2)

        return numpy.exp(Q)

    def __str__(self):
        return "GaussianKernel: sigma = " + str(self.sigma)

class PolyKernel(object):
    """
    A class to find linear kernel evaluations k(x, y) = <x, y> 
    """
    def __init__(self, c, p):
        """
        Intialise class. 
        """
        self.c = c
        self.p = p

    def __call__(self, X1, X2):
        """
        Find kernel evaluation between two matrices X1 and X2 whose rows are
        examples and have an identical number of columns.


        :param X1: First set of examples.
        :type X1: :class:`numpy.ndarray`

        :param X2: Second set of examples.
        :type X2: :class:`numpy.ndarray`
        """

        if X1.shape[1] != X2.shape[1]:
            raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

        return (numpy.dot(X1, X2.T) + self.c) ** self.p

class LinearKernel(PolyKernel):
    def __init__(self):
        super(LinearKernel, self).__init__(0, 1)
        #linearkernel inherits from polykernel with parameter (0,1)
        #super(LinearKernel, self) = super()
            #first parameter refers to subclass LinearKernel
            #second parameter refers to a LinearKernel object, which is self