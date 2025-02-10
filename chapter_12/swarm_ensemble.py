import time
import sys
import numpy as np

#
#  Swarm Intelligence framework:
#

################################################################
#  Bounds
#
class Bounds:
    """Base bounds class"""

    #-----------------------------------------------------------
    #  __init__
    #
    #  Supply lower and upper bounds for each dimension.
    #
    def __init__(self, lower, upper, enforce="clip"):
        """Constructor"""

        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.enforce = enforce.lower() # clip | resample


    #-----------------------------------------------------------
    #  Upper
    #
    #  Return a vector of the per dimension upper limits.
    #
    def Upper(self):
        """Upper bounds"""

        return self.upper


    #-----------------------------------------------------------
    #  Lower
    #
    def Lower(self):
        """Lower bounds"""

        return self.lower


    #-----------------------------------------------------------
    #  Limits
    #
    def Limits(self, pos):
        """Apply the selected boundary conditions"""

        npart, ndim = pos.shape

        for i in range(npart):
            if (self.enforce == "resample"):
                for j in range(ndim):
                    if (pos[i,j] <= self.lower[j]) or (pos[i,j] >= self.upper[j]):
                        pos[i,j] = self.lower[j] + (self.upper[j]-self.lower[j])*np.random.random()
            else:  # clip
                for j in range(ndim):
                    if (pos[i,j] <= self.lower[j]):
                        pos[i,j] = self.lower[j]
                    if (pos[i,j] >= self.upper[j]):
                        pos[i,j] = self.upper[j]
            
            #  Also validate
            pos[i] = self.Validate(pos[i])

        return pos


    #-----------------------------------------------------------
    #  Validate
    #
    #  For example, override this to enforce a discrete position
    #  for a particular vector.
    #
    def Validate(self, pos):
        """Validate a given position vector"""

        return pos


################################################################
#  RandomInitializer
#
class RandomInitializer:
    """Initialize a swarm uniformly"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, npart=10, ndim=3, bounds=None):
        """Constructor"""

        self.npart = npart
        self.ndim = ndim
        self.bounds = bounds


    #-----------------------------------------------------------
    #  InitializeSwarm
    #
    def InitializeSwarm(self):
        """Return a randomly initialized swarm"""

        if (self.bounds == None):
            #  No bounds given, just use [0,1)
            self.swarm = np.random.random((self.npart, self.ndim))
        else:
            #  Bounds given, use them
            self.swarm = np.zeros((self.npart, self.ndim))
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()

            for i in range(self.npart):
                for j in range(self.ndim):
                    self.swarm[i,j] = lo[j] + (hi[j]-lo[j])*np.random.random()        
            self.swarm = self.bounds.Limits(self.swarm)

        return self.swarm


################################################################
#  LinearInertia
#
class LinearInertia:
    """A linear inertia class"""

    #-----------------------------------------------------------
    #  __init___
    #
    def __init__(self, hi=0.9, lo=0.6):
        """Constructor"""
        
        if (hi > lo):
            self.hi = hi
            self.lo = lo
        else:
            self.hi = lo
            self.lo = hi


    #-----------------------------------------------------------
    #  CalculateW
    #
    def CalculateW(self, w0, iterations, max_iter):
        """Return a weight value"""

        return self.hi - (iterations/max_iter)*(self.hi-self.lo)


################################################################
#  DE
#
class DE:
    """Differential evolution"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 #  defaults from Tvrdik(2007) "Differential Evolution with Competitive
                 #  Setting of Control Parameters":
                 CR=0.5,          # cross-over probability
                 F=0.8,           # mutation factor, [0,2]
                 mode="rand",     # v1 variant: "rand" or "best"
                 cmode="bin",     # crossover variant: "bin" or "GA"
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None):    # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.CR = CR
        self.F = F
        self.mode = mode.lower()
        self.cmode = cmode.lower()
        self.tmode = False
        self.initialized = False


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vpos": self.vpos,              # and objective function values
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos= self.Evaluate(self.pos)      # and objective function values

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]].copy())
        self.giter.append(0)


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Evaluate
    #
    def Evaluate(self, pos):
        """Evaluate a set of positions"""

        p = np.zeros(self.npart)
        for i in range(self.npart):
            p[i] = self.obj.Evaluate(pos[i])
        return p


    #-----------------------------------------------------------
    #  Candidate
    #
    def Candidate(self, idx):
        """Return a candidate vector for the given index"""

        k = np.argsort(np.random.random(self.npart))
        while (idx in k[:3]):
            k = np.argsort(np.random.random(self.npart))
        
        v1 = self.pos[k[0]]
        v2 = self.pos[k[1]]
        v3 = self.pos[k[2]]

        if (self.mode == "best"):
            v1 = self.gpos[-1]
        elif (self.mode == "toggle"):
            if (self.tmode):
                self.tmode = False
                v1 = self.gpos[-1]
            else:
                self.tmode = True
        
        #  Donor vector
        v = v1 + self.F*(v2 - v3)

        #  Candidate vector
        u = np.zeros(self.ndim)
        I = np.random.randint(0, self.ndim-1)

        if (self.cmode == "bin"):
            #  Bernoulli crossover
            for j in range(self.ndim):
                if (np.random.random() <= self.CR) or (j == I):
                    u[j] = v[j]
                elif (j != I):
                    u[j] = self.pos[idx,j]
        else:
            #  GA-style crossover
            u = self.pos[idx].copy()
            u[I:] = v[I:]

        return u


    #-----------------------------------------------------------
    #  CandidatePositions
    #
    def CandidatePositions(self):
        """Return a set of candidate positions"""

        pos = np.zeros((self.npart, self.ndim))

        for i in range(self.npart):
            pos[i] = self.Candidate(i)

        if (self.bounds != None):
            pos = self.bounds.Limits(pos)

        return pos


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        new_pos = self.CandidatePositions() # get new candidate positions
        p = self.Evaluate(new_pos)          # and evaluate them

        #  For each particle
        for i in range(self.npart):
            if (p[i] < self.vpos[i]):               # is new position better?
                self.vpos[i] = p[i]                 # keep the function value
                self.pos[i] = new_pos[i]            # and new position
            if (p[i] < self.gbest[-1]):             # is new position global best?
                self.gbest.append(p[i])             # new position is new swarm best
                self.gpos.append(new_pos[i].copy()) # keep the position
                self.gidx.append(i)                 # particle number
                self.giter.append(self.iterations)  # and when it happened

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


################################################################
#  PSO
#
class PSO:
    """Particle swarm optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 c1=1.49,         # cognitive parameter
                 c2=1.49,         # social parameter
                 #  best if w > 0.5*(c1+c2) - 1:
                 w=0.729,         # base velocity decay parameter
                 inertia=None,   # velocity weight decay object (None == constant)
                 #  Bare-bones from:
                 #    Kennedy, James. "Bare bones particle swarms." In Proceedings of 
                 #    the 2003 IEEE Swarm Intelligence Symposium. SIS'03 (Cat. No. 03EX706), 
                 #    pp. 80-87. IEEE, 2003.
                 bare=False,      # if True, use bare-bones update
                 bare_prob=0.5,   # probability of updating a particle's component
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 ring=False,      # use ring topology if True
                 neighbors=2,     # number of particle neighbors for ring, must be even
                 vbounds=None,    # velocity bounds object
                 bounds=None):    # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.vbounds = vbounds
        self.bounds = bounds
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.bare = bare
        self.bare_prob = bare_prob
        self.inertia = inertia
        self.ring = ring
        self.neighbors = neighbors
        self.initialized = False

        if (ring) and (neighbors > npart):
            self.neighbors = npart


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "c1": self.c1,                  # cognitive parameter
            "c2": self.c2,                  # social parameter
            "w": self.w,                    # base velocity decay parameter
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vel": self.vel,                # velocities
            "xpos": self.xpos,              # per particle best positions
            "xbest": self.xbest,            # per particle bests
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()       # initial swarm positions
        self.vel = np.zeros((self.npart, self.ndim)) # initial velocities
        self.xpos = self.pos.copy()                  # these are the particle bests
        self.xbest= self.Evaluate(self.pos)          # and objective function values

        #  Swarm and particle bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.xbest))
        self.gbest.append(self.xbest[self.gidx[-1]])
        self.gpos.append(self.xpos[self.gidx[-1]].copy())
        self.giter.append(0)


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Evaluate
    #
    def Evaluate(self, pos):
        """Evaluate a set of positions"""

        p = np.zeros(self.npart)
        for i in range(self.npart):
            p[i] = self.obj.Evaluate(pos[i])
        return p


    #-----------------------------------------------------------
    #  RingNeighborhood
    #
    def RingNeighborhood(self, n):
        """Return a list of particles in the neighborhood of n"""

        idx = np.array(range(n-self.neighbors//2,n+self.neighbors//2+1))
        i = np.where(idx >= self.npart)
        if (len(i) != 0):
            idx[i] = idx[i] % self.npart
        i = np.where(idx < 0)
        if (len(i) != 0):
            idx[i] = self.npart + idx[i]

        return idx


    #-----------------------------------------------------------
    #  NeighborhoodBest
    #
    def NeighborhoodBest(self, n):
        """Return neighborhood best for particle n"""

        if (not self.ring):
            return self.gbest[-1], self.gpos[-1]

        # Using a ring, return best known position of the neighborhood
        lbest = 1e9
        for i in self.RingNeighborhood(n):
            if (self.xbest[i] < lbest):
                lbest = self.xbest[i]
                lpos = self.xpos[i]

        return lbest, lpos


    #-----------------------------------------------------------
    #  BareBonesUpdate
    #
    def BareBonesUpdate(self):
        """Apply a bare-bones update to the positions"""

        pos = np.zeros((self.npart, self.ndim))

        for i in range(self.npart):
            lbest, lpos = self.NeighborhoodBest(i)
            for j in range(self.ndim):
                if (np.random.random() < self.bare_prob):
                    m = 0.5*(lpos[j] + self.xpos[i,j])
                    s = np.abs(lpos[j] - self.xpos[i,j])
                    pos[i,j] = np.random.normal(m,s)
                else:
                    pos[i,j] = self.xpos[i,j]

        return pos


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        #  Weight for this iteration
        if (self.inertia != None):
            w = self.inertia.CalculateW(self.w, self.iterations, self.max_iter)
        else:
            w = self.w

        if (self.bare):
            #  Bare-bones position update
            self.pos = self.BareBonesUpdate()
        else:
            #  Canonical position/velocity update
            for i in range(self.npart):
                lbest, lpos = self.NeighborhoodBest(i)
                c1 = self.c1 * np.random.random(self.ndim)
                c2 = self.c2 * np.random.random(self.ndim)
                self.vel[i] = w*self.vel[i] +                    \
                              c1*(self.xpos[i] - self.pos[i]) +  \
                              c2*(lpos - self.pos[i])

            #  Keep velocities bounded
            if (self.vbounds != None):
                self.vel = self.vbounds.Limits(self.vel)

            #  Update the positions
            self.pos = self.pos + self.vel

        #  Keep positions bounded
        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)

        #  Evaluate the new positions
        p = self.Evaluate(self.pos)

        #  Check if any new particle and swarm bests
        for i in range(self.npart):
            if (p[i] < self.xbest[i]):                  # is new position a particle best?
                self.xbest[i] = p[i]                    # keep the function value
                self.xpos[i] = self.pos[i]              # and position
            if (p[i] < self.gbest[-1]):                 # is new position global best?
                self.gbest.append(p[i])                 # new position is new swarm best
                self.gpos.append(self.pos[i].copy())    # keep the position
                self.gidx.append(i)                     # particle number
                self.giter.append(self.iterations)      # and when it happened

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


################################################################
#  GWO
#
class GWO:
    """Grey wolf optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 eta=2.0,         # scale factor for a
                 npart=10,        # number of particles in the swarm (> 3)
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None):    # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.eta = eta
        self.initialized = False


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle id of global best
            "giter": self.giter,            # iteration number of global best
            "pos": self.pos,                # current particle positions
            "vpos": self.vpos,              # and objective function values
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos= np.zeros(self.npart)
        for i in range(self.npart):
            self.vpos[i] = self.obj.Evaluate(self.pos[i])

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []
        idx = np.argmin(self.vpos)
        self.gidx.append(idx)
        self.gbest.append(self.vpos[idx])
        self.gpos.append(self.pos[idx].copy())
        self.giter.append(0)

        #  1st, 2nd, and 3rd best positions
        idx = np.argsort(self.vpos)
        self.alpha = self.pos[idx[0]].copy()
        self.valpha= self.vpos[idx[0]]
        self.beta  = self.pos[idx[1]].copy()
        self.vbeta = self.vpos[idx[1]]
        self.delta = self.pos[idx[2]].copy()
        self.vdelta= self.vpos[idx[2]]


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        #  a from eta ... zero (default eta is 2)
        a = self.eta - self.eta*(self.iterations/self.max_iter)

        #  Update everyone
        for i in range(self.npart):
            A = 2*a*np.random.random(self.ndim) - a
            C = 2*np.random.random(self.ndim)
            Dalpha = np.abs(C*self.alpha - self.pos[i]) 
            X1 = self.alpha - A*Dalpha

            A = 2*a*np.random.random(self.ndim) - a
            C = 2*np.random.random(self.ndim)
            Dbeta = np.abs(C*self.beta - self.pos[i]) 
            X2 = self.beta - A*Dbeta

            A = 2*a*np.random.random(self.ndim) - a
            C = 2*np.random.random(self.ndim)
            Ddelta = np.abs(C*self.delta - self.pos[i]) 
            X3 = self.delta - A*Ddelta 
            
            self.pos[i,:] = (X1+X2+X3) / 3.0

        #  Keep in bounds
        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)

        #  Get objective function values and check for new leaders
        for i in range(self.npart):
            self.vpos[i] = self.obj.Evaluate(self.pos[i])
			
            #  new alpha?
            if (self.vpos[i] < self.valpha):
                self.vdelta = self.vbeta
                self.delta = self.beta.copy()
                self.vbeta = self.valpha
                self.beta = self.alpha.copy()
                self.valpha = self.vpos[i]
                self.alpha = self.pos[i].copy()

            #  new beta?
            if (self.vpos[i] > self.valpha) and (self.vpos[i] < self.vbeta):
                self.vdelta = self.vbeta
                self.delta = self.beta.copy()
                self.vbeta = self.vpos[i]
                self.beta = self.pos[i].copy()
            
            #  new delta?
            if (self.vpos[i] > self.valpha) and (self.vpos[i] < self.vbeta) and (self.vpos[i] < self.vdelta):
                self.vdelta = self.vpos[i]
                self.delta = self.pos[i].copy()

            #  is alpha new swarm best?
            if (self.valpha < self.gbest[-1]):
                self.gidx.append(i)
                self.gbest.append(self.valpha)
                self.gpos.append(self.alpha.copy())
                self.giter.append(self.iterations)

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


################################################################
#  Jaya
#
class Jaya:
    """Jaya optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None):    # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.initialized = False


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vpos": self.vpos,              # and objective function values
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos= self.Evaluate(self.pos)      # and objective function values

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]].copy())
        self.giter.append(0)


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Evaluate
    #
    def Evaluate(self, pos):
        """Evaluate a set of positions"""

        p = np.zeros(self.npart)
        for i in range(self.npart):
            p[i] = self.obj.Evaluate(pos[i])
        return p


    #-----------------------------------------------------------
    #  CandidatePositions
    #
    def CandidatePositions(self):
        """Return a set of candidate positions"""

        pos = np.zeros((self.npart, self.ndim))

        f = np.argsort(self.vpos)
        best = self.pos[f[0]]
        worst= self.pos[f[-1]]

        for i in range(self.npart):
            r1 = np.random.random(self.ndim)
            r2 = np.random.random(self.ndim)

            pos[i] = self.pos[i] + r1*(best  - np.abs(self.pos[i])) -  \
                                   r2*(worst - np.abs(self.pos[i]))

        if (self.bounds != None):
            pos = self.bounds.Limits(pos)

        return pos


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        new_pos = self.CandidatePositions() # get new candidate positions
        p = self.Evaluate(new_pos)          # and evaluate them

        #  For each particle
        for i in range(self.npart):
            if (p[i] < self.vpos[i]):               # is new position better?
                self.vpos[i] = p[i]                 # keep the function value
                self.pos[i] = new_pos[i]            # and new position
            if (p[i] < self.gbest[-1]):             # is new position global best?
                self.gbest.append(p[i])             # new position is new swarm best
                self.gpos.append(new_pos[i].copy()) # keep the position
                self.gidx.append(i)                 # particle number
                self.giter.append(self.iterations)  # and when it happened

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


################################################################
#  RO
#
class RO:
    """Parallel random optimization"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 eta=0.1,         # max fractional change for candidate positions
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None):    # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.eta = eta
        self.initialized = False


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,                # tolerance value, if any
            "eta": self.eta,                # max candidate fraction
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vpos": self.vpos,              # and objective function values
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos= self.Evaluate(self.pos)      # and objective function values

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]])
        self.giter.append(0)


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Evaluate
    #
    def Evaluate(self, pos):
        """Evaluate a set of positions"""

        p = np.zeros(self.npart)
        for i in range(self.npart):
            p[i] = self.obj.Evaluate(pos[i])
        return p


    #-----------------------------------------------------------
    #  CandidatePositions
    #
    def CandidatePositions(self):
        """Return a set of candidate positions"""

        n = np.random.normal(size=(self.npart, self.ndim))/5.0
        pos = self.pos + self.eta*self.pos*n

        if (self.bounds != None):
            pos = self.bounds.Limits(pos)

        return pos


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        new_pos = self.CandidatePositions() # get new candidate positions
        p = self.Evaluate(new_pos)          # and evaluate them

        #  For each particle
        for i in range(self.npart):
            if (p[i] < self.vpos[i]):               # is new position better?
                self.vpos[i] = p[i]                 # keep the function value
                self.pos[i] = new_pos[i]            # and new position
            if (p[i] < self.gbest[-1]):             # is new position global best?
                self.gbest.append(p[i])             # new position is new swarm best
                self.gpos.append(new_pos[i])        # keep the position
                self.gidx.append(i)                 # particle number
                self.giter.append(self.iterations)  # and when it happened

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


################################################################
#  GA
#
class GA:
    """Genetic algorithm"""

    #-----------------------------------------------------------
    #  __init__
    #
    def __init__(self, obj,       # the objective function (subclass Objective)
                 npart=10,        # number of particles in the swarm
                 ndim=3,          # number of dimensions in the swarm
                 max_iter=200,    # maximum number of steps
                 CR=0.8,          # cross-over probability
                 F=0.05,          # mutation probability
                 top=0.5,         # top fraction (only breed with the top fraction)
                 tol=None,        # tolerance (done if no done object and gbest < tol)
                 init=None,       # swarm initialization object (subclass Initializer)
                 done=None,       # custom Done object (subclass Done)
                 bounds=None):    # swarm bounds object

        self.obj = obj
        self.npart = npart
        self.ndim = ndim
        self.max_iter = max_iter
        self.init = init
        self.done = done
        self.bounds = bounds
        self.tol = tol
        self.CR = CR
        self.F = F
        self.top = top
        self.initialized = False


    #-----------------------------------------------------------
    #  Results
    #
    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,            # number of particles
            "ndim": self.ndim,              # number of dimensions 
            "max_iter": self.max_iter,      # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,                # tolerance value, if any
            "gbest": self.gbest,            # sequence of global best function values
            "giter": self.giter,            # iterations when global best updates happened
            "gpos": self.gpos,              # global best positions
            "gidx": self.gidx,              # particle number for new global best
            "pos": self.pos,                # current particle positions
            "vpos": self.vpos,              # and objective function values
        }


    #-----------------------------------------------------------
    #  Initialize
    #
    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0
       
        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos= self.Evaluate(self.pos)      # and objective function values

        #  Swarm bests
        self.gidx = []
        self.gbest = []
        self.gpos = []
        self.giter = []

        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]].copy())
        self.giter.append(0)


    #-----------------------------------------------------------
    #  Done
    #
    def Done(self):
        """Check if we are done"""

        if (self.done == None):
            if (self.tol == None):
                return (self.iterations == self.max_iter)
            else:
                return (self.gbest[-1] < self.tol) or (self.iterations == self.max_iter)
        else:
            return self.done.Done(self.gbest,
                        gpos=self.gpos,
                        pos=self.pos,
                        max_iter=self.max_iter,
                        iteration=self.iterations)


    #-----------------------------------------------------------
    #  Evaluate
    #
    def Evaluate(self, pos):
        """Evaluate a set of positions"""

        p = np.zeros(self.npart)
        for i in range(self.npart):
            p[i] = self.obj.Evaluate(pos[i])
        return p


    #-----------------------------------------------------------
    #  Mutate
    #
    def Mutate(self, idx):
        """Return a mutated position vector"""

        j = np.random.randint(0,self.ndim)
        if (self.bounds != None):
            self.pos[idx,j] = self.bounds.lower[j] + np.random.random()*(self.bounds.upper[j]-self.bounds.lower[j])
        else:
            lower = self.pos[:,j].min()
            upper = self.pos[:,j].max()
            self.pos[idx,j] = lower + np.random.random()*(upper-lower)


    #-----------------------------------------------------------
    #  Crossover
    #
    def Crossover(self, a, idx):
        """Mate with another swarm member"""

        #  Get the partner in the top set
        n = int(self.top*self.npart)
        b = idx[np.random.randint(0, n)]
        while (a==b):
            b = idx[np.random.randint(0, n)]

        #  Random cut-off position
        d = np.random.randint(0, self.ndim)

        #  Crossover
        t = self.pos[a].copy()
        t[d:] = self.pos[b,d:]
        self.pos[a] = t.copy()


    #-----------------------------------------------------------
    #  Evolve
    #
    def Evolve(self):
        """Evolve the swarm"""

        idx = np.argsort(self.vpos)

        for k,i in enumerate(idx):
            if (k == 0):
                continue    #  leave the best one alone
            if (np.random.random() < self.CR):
                #  Breed this one with one of the better particles
                self.Crossover(i, idx)
            if (np.random.random() < self.F):
                #  Random mutation
                self.Mutate(i)

        if (self.bounds != None):
            self.pos = self.bounds.Limits(self.pos)


    #-----------------------------------------------------------
    #  Step
    #
    def Step(self):
        """Do one swarm step"""

        self.Evolve()                               # evolve the swarm
        self.vpos = self.Evaluate(self.pos)         # and evaluate the new positions

        #  For each particle
        for i in range(self.npart):
            if (self.vpos[i] < self.gbest[-1]):         # is new position global best?
                self.gbest.append(self.vpos[i])         # new position is new swarm best
                self.gpos.append(self.pos[i].copy())    # keep the position
                self.gidx.append(i)                     # particle number
                self.giter.append(self.iterations)      # and when it happened

        self.iterations += 1


    #-----------------------------------------------------------
    #  Optimize
    #
    def Optimize(self):
        """Run a full optimization and return the best"""

        self.Initialize()

        while (not self.Done()):
            self.Step()

        return self.gbest[-1], self.gpos[-1]


#
#  Swarm weight vector search:
#
class Objective:
    def __init__(self, probs, y):
        self.probs = probs
        self.y = y
        self.fcount = 0

    def Evaluate(self, p):
        self.fcount += 1
        w = p / p.sum()
        prob = 0.0
        for i in range(6):
            prob += w[i]*self.probs[i]
        v = np.argmax(prob, axis=1)
        cc = np.zeros((10,10))
        for i in range(len(self.y)):
            cc[self.y[i],v[i]] += 1
        return 1.0 -  (np.diag(cc).sum() / cc.sum())


if (len(sys.argv) == 1):
    print()
    print("swarm_ensemble <npart> <niter> <alg>")
    print()
    print("  <npart>  - number of particles")
    print("  <niter>  - number of iterations")
    print("  <alg>    - PSO,JAYA,DE,RO,GWO,GA")
    print()
    exit(0)

#  Get the parameters
ndim = 6
npart = int(sys.argv[1])
niter = int(sys.argv[2])
alg = sys.argv[3].upper()

b = Bounds([0.0]*ndim, [1.0]*ndim, enforce="resample")
i = RandomInitializer(npart, ndim, bounds=b)
tol = 0.0

#  predictions from separate models
p0 = np.load("prob_run0.npy")
p1 = np.load("prob_run1.npy")
p2 = np.load("prob_run2.npy")
p3 = np.load("prob_run3.npy")
p4 = np.load("prob_run4.npy")
p5 = np.load("prob_run5.npy")
y = np.load("../data/audio/ESC-10/esc10_spect_test_labels.npy")

obj = Objective([p0,p1,p2,p3,p4,p5], y)

if (alg == "PSO"):
    swarm = PSO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b, bare=True)
elif (alg == "JAYA"):
    swarm = Jaya(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b)
elif (alg == "DE"):
    swarm = DE(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b)
elif (alg == "RO"):
    swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b)
elif (alg == "GWO"):
    swarm = GWO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b)
elif (alg == "RO"):
    swarm = RO(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b)
elif (alg == "GA"):
    swarm = GA(obj=obj, npart=npart, ndim=ndim, init=i, tol=tol, max_iter=niter, bounds=b)
else:
    raise ValueError("Unknown algorithm: %s" % alg)

st = time.time()
k = 0
swarm.Initialize()
while (not swarm.Done()):
    swarm.Step()
    res = swarm.Results()
    k += 1
    if ((k % 100) == 0) or (k == 1):
        print("%04d: accuracy = %0.8f" % (k, 100.0*(1.0 - res['gbest'][-1])))
res = swarm.Results()
en = time.time()

print()
print("Accuracy: %0.9f" % (100.0*(1.0 - res["gbest"][-1]),))
print()
print("Weights:")
w = res['gpos'][-1] / sum(res['gpos'][-1])
for k,p in enumerate(w):
    print("%2d: %0.9f" % (k,p))
print()
print("(%d best updates, %d function calls, time: %0.3f seconds)" % (len(res["gbest"]), swarm.obj.fcount, en-st))

