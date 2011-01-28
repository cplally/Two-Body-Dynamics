#### A two-body dynamics simulator, using scipy and pylab

import pylab

from scipy import array, arange, zeros, inner
from scipy.integrate import odeint

from math import sqrt, pow

## The gravitational constant
## N.B. -- This program uses S.I. units
G = 6.67428E-11

def norm(x):
    """Calculate the norm of a vector x"""
    return sqrt(inner(x, x))

## The following class contains the method called at each time step to
## calculate the RHS of the equations of motion.
## That function does not stand on its own, as it must have knowledge of the two
## masses (m1, m2), which are here represented by properties of an RHS instance.
class RHS:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        
    def rhs_func(self, x, args):
        """Differentiates the given state vector x with respect to time,
        returning the RHS of the equations of motion"""
        def a_ji(x, mj):
            """Acceleration of mi due to gravitational force of mj on mi"""
            return (G*mj*x)/pow(norm(x), 3)
        
        # set up the state vector at time t'
        y = zeros(8)
        
        # first, calculate the positions of the masses
        y[0:2] = x[4:6]                            # y[0:2] = x1(t')
        y[2:4] = x[6:8]                            # y[2:4] = x2(t')
        # then calculate the velocities
        y[4:6] = a_ji(x[2:4] - x[0:2], self.m2)    # y[4:6] = v1(t')
        y[6:8] = a_ji(x[0:2] - x[2:4], self.m1)    # y[6:8] = v2(t')
        
        return y

def plot_trajectory(trajectories):
    """Uses pylab to plot the trajectories of m1 (in blue) and m2 (in red)"""
    pylab.plot(trajectories[:,0], trajectories[:,1], 'b.')
    pylab.plot(trajectories[:,2], trajectories[:,3], 'r.')
    
    pylab.xlabel("x")
    pylab.ylabel("y")
    pylab.title("Trajectories (m1's is blue)")
    pylab.grid()
    
    pylab.show()

def two_body_sim(masses, initial_positions, initial_velocities, time_step,
                 end_time):
    """Simulate the dynamics of a two-body system specified by masses and
    intial conditions"""
    
    # sanity checks
    assert len(masses) == 2, "Exactly two masses must be specified."
    assert len(initial_positions) == 2, "An initial position must be specified\
 for each mass."
    assert len(initial_velocities) == 2, "An initial velocity must be specified\
 for each mass."
    
    # set up the system at time t=0
    r0 = zeros(8)
    r0[0:2] = array(initial_positions[0])     # r0[0:2] = x1(0)
    r0[2:4] = array(initial_positions[1])     # r0[2:4] = x2(0)
    r0[4:6] = array(initial_velocities[0])    # r0[4:6] = v1(0)
    r0[6:8] = array(initial_velocities[1])    # r0[6:8] = v2(0)
    
    # create an array of times at which to find x(t)
    t = arange(0, end_time+time_step, time_step)
    # now solve the system of ODE and get the trajectory of each mass
    trajectories = odeint(RHS(masses[0], masses[1]).rhs_func, r0, t)
    # finally, plot the trajectories of the masses
    plot_trajectory(trajectories)
