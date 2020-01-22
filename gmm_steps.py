import numpy
import math

def log_likelihood(X, mean, variance):  

    # [TODO] Complete the calculation of the LL point-estimate
    # Tip: Use math.pi as a replacement for the Pi constant
    #return numpy.ones(shape=len(X), dtype=float)
    
    #### ANSWER ####
    
    # [TODO] Complete the calculation of the LL point-estimate
    # Tip: Use math.pi as a replacement for the Pi constant    
    # Tip 2: In this task each line corresponds to one samples and each column one dimension
    # Tip 3: Remember that the inverse of a diagonal matrix A is 1/A (only diagonal elements).
    #        This is true for \Sigma    
    # Tip 4: Don't forget LL corresponds to one scalar per example

    return -0.5 * (X.shape[1]*math.log(2*math.pi) + numpy.log(variance).sum()) \
    -0.5 * ( (X-mean)**2 * (1./variance)).sum(axis=1)


def e_step(X, weights, means, variances):  

    ## log-likelihoods = 2D numpy.ndarray with as many rows as examples and
    ## as many columns as dimensions in X
    # [TODO] calculate the first_terms first:
    #        first_terms = log ( w_i ) + log_likelihood(x_t; mu_i, Sigma_i)
    first_terms = numpy.array([math.log(w) + log_likelihood(X, mu, sigma) for (w,mu,sigma) in zip(weights, means, variances)])

    # [TODO] compute the denominator - i.e., the second term:
    #        log ( sum_j^M [ w_j * Normal(x_t; mu_j, Sigma_j) ] )

    # this is the place where we would like to use the log-add-exp trick
    # to do so, you have to operate in pairs and accumulate - use a ``for`` loop
    # The shape of second_term is (len(means), len(X))
    second_term = first_terms[0]
    for k in first_terms[1:]: second_term = logadd(second_term, k)

    # WARNING: Do not change the following return string
    #return numpy.exp(first_terms - second_term).T, second_term
    
    responsibilities = numpy.exp(first_terms - second_term).T 
    
    zeroth_order_stats = numpy.sum(responsibilities, axis=0)
                
    first_order_stats = numpy.dot(responsibilities.T, X).T
    
    return zeroth_order_stats, first_order_stats



def acc_stats(gmm_stats):

    zeroth_order_stats = numpy.zeros(gmm_stats[0][0].shape)
    first_order_stats =  numpy.zeros(gmm_stats[0][1].shape)

    for stats in gmm_stats:
        zeroth_order_stats += stats[0]
        first_order_stats += stats[1]
        
    return zeroth_order_stats, first_order_stats




def m_step(gmm_stats):

    zeroth_order_stats = gmm_stats[0]
    first_order_stats = gmm_stats[1]    
    
    means = first_order_stats / zeroth_order_stats
    
    return means.T



def logadd(log_a, log_b):
    """Computes log(a+b) given log(a) and log(b) as numerically stable as possible

    You can compute it, in a numerically stable way, like this:

    log (a + b) = log(a) + log(1 + b/a)
                = log(a) + log(1 + exp(log(b) - log(a)))
                = log(a) + log1p(exp(log(b) - log(a)))

    Please notice that for this to work correctly, it is expected that a > b. If
    that is not the case (i.e., b > a), it is more stable if you compute:

    log (b + a) = log(b) + log(1 + a/b)
                = log(b) + log(1 + exp(log(a) - log(b)))
                = log(b) + log1p(exp(log(a) - log(b)))

    So, a test must be done in every addition. Also note that, iff:

    a > b => log(a) > log(b)

    because log(.) is a monotonically increasing function. This will simplify our
    comparisons.


    Parameters:

    log_a (numpy.ndarray): A 1D array with the logarithms of another 1D array
      of the same size

    log_b (numpy.ndarray): A second 1D array with the logarithms of another 1D
      array of the same size


    Returns:

    numpy.ndarray: A 1D array with the same size as the input vectors,
    representing the logarithm of the sum of the two terms ``a`` and ``b``.

    """

    # use numpy.where to select all elements that are bigger or smaller
    smallest_log = numpy.where(log_a < log_b, log_a, log_b)
    biggest_log = numpy.where(log_a > log_b, log_a, log_b)

    # we now perform the addition
    diff = smallest_log - biggest_log
    return numpy.where(
      diff < -39.14, #why? => exp(-39.14) = 1e-17, so log(1+1e-17) ~ log(1) = 0
      biggest_log,
      biggest_log + numpy.log1p(numpy.exp(diff))
      )
