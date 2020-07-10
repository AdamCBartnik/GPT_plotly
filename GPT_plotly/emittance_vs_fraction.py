import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from .ParticleGroupExtension import core_emit_calc

def emittance_vs_fraction(pg, var, number_of_points=25, plotting=True, verbose=False, show_core_emit_plot=False):

    min_particle_count = 100   # minimum number of particles required to compute a fraction emittance

    var1 = var
    var2 = 'p' + var
    
    # Check input and perform initializations:
    x = getattr(pg, var1)
    y = getattr(pg, var2)/pg.mass
    w = pg.weight
    
    particle_count = len(x)

    if particle_count < min_particle_count:
        raise ValueError(f'Too few particles for emittance vs fraction {particle_count} < {min_particle_count}')

    if (min_particle_count/particle_count > 0.05 and verbose):
        print('emittance_vs_fraction_for_particles -> the minimum estimated fraction to be computed is > 5%.')
        print('This could effect the calculation of the core emittance and core fraction. Suggestions:')
        print('(1) Increase the total particle count in the input distribution [x,y]')
        print('(2) Decrease the user defined minimum allowed particle count')

    emit = calculate_emittance(x, y, w)[0]
    
    fs = np.linspace(0,1,number_of_points)
    fc = 1.0
    ec = 0.0

    if (emit < 1e-15):
        if (verbose):
            print('Possible zero emittance beam, assuming emit = 0')
        es = np.zeros(number_of_points)
        return (fs, es, fc, ec)

    # Calculate maximum ellipse emittance:
    (emittance, alpha, beta, center, gamma) = minboundellipse(x,y,tolerance=1.0e-3,plot_on=False)
    twiss_parameters = np.array([emittance, alpha, beta, center[0], center[1]])
    
    # Make sure to capture all particles:
    while (get_number_of_excluded_particles(twiss_parameters,x,y) > 0):
        twiss_parameters[0] = 1.01*twiss_parameters[0]

    fc = fs[0:-1]
    es = 2.0*np.log(1.0/(1.0-fc))*emit
    es = np.concatenate((es, [emit]))
    fs[1:-1]=0

    aa = np.empty(len(fs))
    bb = np.empty(len(fs))
    cx = np.empty(len(fs))
    cp = np.empty(len(fs))
    aa[:] = np.nan
    bb[:] = np.nan
    cx[:] = np.nan
    cp[:] = np.nan
    
    # Computation of emittance vs. fractions
    
    # Run through bounding ellipse areas (largest to smallest) and compute the
    # enclosed fraction and emittance of inclosed beam.  The Twiss parameters
    # computed for the minimum bounding ellipse for the entire distribution is
    # used as an initial guess:

    if verbose:
       print('')
       print('   computing emittance vs. fraction curve...') 


    indices = np.arange(len(es)-2,0,-1)
    for ind, ii in enumerate(indices):

        iteration = 1 + indices[0] - ii

        if verbose:
            print(f'      Iteration: {iteration} / {len(indices)}')


        # use previous ellipse as a guess point to compute next one:
        twiss_parameter_guess = twiss_parameters
        twiss_parameter_guess[0] = es[ii]

        twiss_parameter_bounds = ((es[ii], es[ii]), (None, None), (0, None), (np.min(x), np.max(x)), (np.min(y), np.max(y)))
                
        bnds = ((0, None), (0, None), (0, None))
        #res = minimize(lambda xx: get_number_of_excluded_particles(xx,x,y), twiss_parameter_guess, method='SLSQP', bounds=twiss_parameter_bounds, 
        #              tol=0.1, options={'maxiter': 100, 'ftol': 1.0})  # In Matlab, ('TolX', 1e-1, 'TolFun', 1)
        #twiss_parameters = res.x
        
        res = fmin(lambda xx: get_number_of_excluded_particles(np.concatenate([[es[ii]],xx]),x,y), twiss_parameter_guess[1:], args=(), xtol=0.1, ftol=1, maxiter=None, disp=verbose)
               
        # get fraction and emittance of included beam:
        included_particle_count = get_number_of_included_particles(twiss_parameters,x,y)
        f = included_particle_count/particle_count   
        if included_particle_count > min_particle_count:
            fs[ii] = f   
            included_particles = find_particles_in_twiss_ellipse(twiss_parameters, x, y)
            es[ii] = calculate_emittance(x[included_particles], y[included_particles], w[included_particles])[0]
            aa[ii] = twiss_parameters[1]
            bb[ii] = twiss_parameters[2]
            cx[ii] = twiss_parameters[3]
            cp[ii] = twiss_parameters[4]

        else:
            if verbose:
                print(f'Number of included particles was below user defined minimum ({included_particle_count} < {min_particle_count}).')
                print(f'Stopping emittance vs. fraction computation at: f = {f}].')

            # if quiting early, pad the remaining values of the [f,e] curve with nans:
            fs[2:ii] = np.nan
            es[2:ii] = np.nan

            break
            
    if verbose:
        print('   ...done.')

    # Compute core fraction and emittance:

    if verbose:
        print('')
        print('   computing core emittance and fraction: ')

    ec = core_emit_calc(x, y, w, show_fit=show_core_emit_plot)
                    
    if verbose:
        print('done.')
    
    fs = fs[np.logical_not(np.isnan(fs))]
    es = es[np.logical_not(np.isnan(es))]

    # remove duplicate values
    (fs, ifs) = np.unique(fs, return_index=True)
    es = es[ifs]  
    
    fc = np.interp(ec,es,fs)
    ac = np.interp(fc,fs,aa)
    bc = np.interp(fc,fs,bb)
    gc = (1.0+ac**2)/bc
        
    # Plot results

    if plotting:
        if verbose:
            print('   plotting data: ')

        plot_points=100
            
        fc1s = np.ones(plot_points)*fc
        ec1s = np.linspace(0.0,1.0,plot_points)*ec

        ec2s = np.ones(plot_points)*ec
        fc2s = np.linspace(0.0,1.0,plot_points)
        
        plt.figure(dpi=100)

        plt.plot(fc1s, ec1s, 'r--')
        plt.plot(fc2s, ec2s, 'r--')
        plt.plot(fs, ec*fs, 'r')
        plt.plot(fs, es, 'b.-')
                
        plt.xlim([0,1])
        plt.ylim(bottom=0)
        
        plt.xlabel('Fraction')
        plt.ylabel('Emittance')

        plt.title(f'$\epsilon_{{core}} = {ec:.3g}, f_{{core}} = {fc:.3f}$')
        
        if verbose:
            print('done.')


#        figure(2)
#        clf;
#        rho = hist3([x',y'], [sqrt(length(x)),sqrt(length(y))]);
#        imagesc(x,y,rho)
#        set(gca,'YDir','normal')

#        xs = linspace(-1,1,100);
#        yp = +sqrt(1-xs.^2);
#        ym = -sqrt(1-xs.^2);

#        xfp = sqrt(bc)*xs;
#        pfp = (-ac/sqrt(bc))*xs + (1/sqrt(bc))*yp;
#        hold on
#        plot(xfp,pfp)
#        hold off 
#    end

    return (es, fs, ec, fc)

                   

            
# Generate emittance vs fraction curve for Gaussian distribution:
def gaussian_emittance_vs_fraction(number_of_points, full_emittance):

    fs = np.linspace(0.0, 1.0, number_of_points)
    fc = fs[1:-1]

    nes = (1.0-(1.0-fc)*(1.0-np.log(1.0-fc)))/fc
    nes = np.concatenate([[0.0], nes, [1.0]])

    es = full_emittance*nes

    return (es, fs)
                    
    
                    
def calculate_emittance(x, y, w):
    # e = rms emittance, a = alpha Twiss parameter, b = beta Twiss parameter,
    # x0 = coordinate 1 centroid, y0 = coordinate 2 centroid

    w_sum = np.sum(w)

    # Compute 1st moments:
    x0=np.sum(x*w)/w_sum
    y0=np.sum(y*w)/w_sum

    dx=x-x0
    dy=y-y0

    # Compute 2nd moments:
    x2 = np.sum(x**2*w)/w_sum
    y2 = np.sum(y**2*w)/w_sum
    xy = np.sum(x*y*w)/w_sum

    # Compute Twiss parameters
    e=np.sqrt(x2*y2-xy**2)
    a = -xy/e
    b =  x2/e

    return (e,a,b,x0,y0)

                    
                    
def minboundellipse( x_all, y_all, tolerance=1.0e-3, plot_on=False):

    # x_all and y_all are rows of points

    # reduce set of points to just the convex hull of the input
    ch = ConvexHull(np.array([x_all,y_all]).transpose())
    
    x = x_all[ch.vertices]
    y = y_all[ch.vertices]

    d = 2
    N = len(x)
    P = np.array([x, y])
    Q = np.array([x, y, np.ones(N)])

    # Initialize
    count = 1
    err = 1
    u = (1.0/N) * np.array([np.ones(N)]).transpose()

    # Khachiyan Algorithm
    while (err > tolerance):
        X = Q @ np.diag(u.reshape(len(u))) @ Q.transpose()        
        M = np.diag(Q.transpose() @ np.linalg.solve(X, Q))

        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum-d-1.0)/((d+1.0)*(maximum-1.0))

        new_u = (1.0 - step_size)*u
        new_u[j] = new_u[j] + step_size

        err = np.linalg.norm(new_u - u)

        count = count + 1
        u = new_u

    U = np.diag(u.reshape(len(u)))

    # Compute the twiss parameters    
    A = (1.0/d) * np.linalg.inv(P @ U @ P.transpose() - (P @ u) @ (P @ u).transpose() )

    (U, D, V) = np.linalg.svd(A)
    
    a = 1/np.sqrt(D[0]) # major axis
    b = 1/np.sqrt(D[1]) # minor axis

    # make sure V gives pure rotation
    if (np.linalg.det(V) < 0):
        V = V @ np.array([[-1, 0], [0, 1]])

    emittance = a*b

    gamma = A[0,0]*emittance;
    beta = A[1,1]*emittance;
    alpha = A[1,0]*emittance;

    # And the center
    c = P @ u
    center = c

    if (plot_on):

        plt.figure(dpi=100)

        theta = np.linspace(0,2*np.pi,100)

        state = np.array([a*np.cos(theta), b*np.sin(theta)])

        X = V @ state
        X[0,:] = X[0,:] + c[0]
        X[1,:] = X[1,:] + c[1]

        plt.plot(X[0,:], X[1,:], 'r-')
        plt.plot(c[0], c[1], 'r*')
        plt.plot(x_all, y_all, 'b.')
                
    
    return (emittance, alpha, beta, center, gamma)
    
             
             
            
def find_particles_in_twiss_ellipse(twiss_parameters, x, y):
    # unpack Twiss parameters:
    emittance = twiss_parameters[0]
    alpha = twiss_parameters[1]
    beta = twiss_parameters[2]
    x0 = twiss_parameters[3]
    y0 = twiss_parameters[4]
    
    
    # subtract out centroids:
    dx=x-x0
    dy=y-y0

    # compute and compare single particle emittances to emittance from Twiss parameters
    gamma=(1+alpha**2)/beta
    e_particles = gamma*dx**2 + beta*dy**2 + 2*alpha*dx*dy
    return (e_particles < emittance)

def get_number_of_included_particles(twiss_parameters, x, y):
    return np.count_nonzero(find_particles_in_twiss_ellipse(twiss_parameters, x, y)) 

def get_number_of_excluded_particles(twiss_parameters, x, y):
    return len(x) - get_number_of_included_particles(twiss_parameters, x, y)

