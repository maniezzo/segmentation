import numpy as np
import random as rnd

class PSO:
   def __init__(self, nump, objFunc, ndim, maxiter, numneigh, 
                 x_min, x_max, isMax, c0=0.25, c1=2, c2=2, isVerbose = False):
      '''
      nump:     number of particles
      objFunc:  the objective funtion to maximize / minimize
      ndim:     number of coefficints to optimize
      maxiter:  max num of iterations
      numneigh: number of neighbors of each particle
      x_min, x_max: arrays of max and min dimensions values
      isMax: true:  maximizatin problem, false minimization
      '''
      self.fitbest    = np.NINF if isMax else np.inf
      self.x_min      = x_min
      self.x_max      = x_max
      self.objFunc    = objFunc
      self.nump       = nump
      self.ndim       = ndim
      self.maxiter    = maxiter
      self.numneigh   = numneigh
      self.isMax      = isMax
      self.isVerbose  = isVerbose
      
      # seed for reproducibility
      rnd.seed(550)
      
      # global best solution
      self.xsolbest = np.zeros(ndim, dtype=float)
      
      # particles init
      self.prtcls = []
      
      for _ in range(nump):
          p = Particle(self.ndim, c0, c1, c2, self.numneigh, self.objFunc, 
                       self.x_max, self.x_min, self.isMax)
          self.prtcls.append(p)
          
      for particle in self.prtcls:
          particle.initialize_neighborhood(self.prtcls, numneigh, nump)
            

   def runPSO(self):
      # loop until iteration end
      noimpr=0
      for iter in range(self.maxiter):
         # debug print
         if (self.isVerbose and iter%1000 == 0):
             print("- iter {0}, global best {1}".format(iter, self.fitbest))
         
         # particles update
         maxnoimpr = 100
         for i in range(self.nump):
            if(noimpr==maxnoimpr): break
            # update velocity and position
            self.prtcls[i].updateVelPos(self.x_min, self.x_max)
 
            # update particle fitness
            self.prtcls[i].fit = self.objFunc(self.prtcls[i].x)
            
            # update personal best position
            self.prtcls[i].update_pbest(self.isMax)
                    
            # update neighborhood best position
            self.prtcls[i].update_nbest(self.prtcls, self.isMax)
            
            # update global best
            if ((not self.isMax and self.prtcls[i].fit < self.fitbest) 
                 or (self.isMax and self.prtcls[i].fit > self.fitbest)):
               noimpr = 0
               self.fitbest = self.prtcls[i].fit
               for j in range(self.ndim):
                   self.xsolbest[j] = self.prtcls[i].x[j]
            else:
               noimpr+=1
              
      # return result
      return self.xsolbest
        
class Particle:
    def __init__(self, ndim, _c0, _c1, _c2, numneigh, objFunc, x_max, x_min, 
                 isMax):
        self.c0  = _c0 # velocity coefficient
        self.c1  = _c1  # pbest coefficient
        self.c2  = _c2  # nbest coefficient
        self.fit = 0    # current fitness
        self.fitnbest = 0 # neighborhood best fitness
        self.fitbest  = 0 # personal best fitness
        self.v = np.zeros(ndim, dtype=float) # personal velocity
        self.x = np.zeros(ndim, dtype=float) # personal position
        self.xbest  = np.zeros(ndim, dtype=float) # personal best
        self.nxbest = np.zeros(ndim, dtype=float) # neighborhood best
        self.neighbors = np.zeros(numneigh, dtype=int) # neighbor ids

        # initialize positions and velocities
        for j in range(ndim):
            self.x[j] = rnd.random() * (x_max[j] - x_min[j])
            self.v[j] = (rnd.random() - rnd.random()) * 0.5 * (x_max[j] - x_min[j])
            self.xbest[j] = self.x[j]
            self.nxbest[j] = self.x[j]
        
        # compute fitness based on the new configuration
        self.fit = objFunc(self.x)
        self.fitbest = self.fit

    def initialize_neighborhood(self, pars, numneigh, nump):
        # initialize neighborhood (randomly choosed neighbors)
        for j in range(numneigh):
            id = rnd.randrange(nump)
            while(id in self.neighbors):
                id = rnd.randrange(nump)
            else:
                self.neighbors[j] = id

    def updateVelPos(self, x_min, x_max):
        # for each dimension
        for d in range(len(self.x)):
            # stochastic coefficienrts
            rho1 = self.c1 * rnd.random()
            rho2 = self.c2 * rnd.random()
                    
            # update velocity
            self.v[d] = self.c0 * self.v[d] + rho1 * (self.xbest[d] - self.x[d]) + rho2 * (self.nxbest[d] - self.x[d])
                        
            # update position
            self.x[d] += self.v[d]
                    
            # clamp position within bounds
            if (self.x[d] < x_min[d]):
                self.x[d] = x_min[d]
                self.v[d] = -self.v[d]
            elif (self.x[d] > x_max[d]):
                self.x[d] = x_max[d]
                self.v[d] = -self.v[d]

    def update_pbest(self, isMax):
        if ((not isMax and self.fit <  self.fitbest) 
            or  (isMax and self.fit >= self.fitbest)):

            self.fitbest = self.fit
            for j in range(len(self.x)):
                self.xbest[j] = self.x[j]

    def update_nbest(self, pars, isMax):
        self.fitnbest = 0 if isMax else np.inf
        for j in range(len(self.neighbors)):
            if ((not isMax and pars[self.neighbors[j]].fit <  self.fitnbest)
                or (isMax and pars[self.neighbors[j]].fit >= self.fitnbest)):
                
                self.fitnbest = pars[self.neighbors[j]].fit
                for k in range(len(self.x)):
                    self.nxbest[k] = pars[ self.neighbors[j] ].x[k]
