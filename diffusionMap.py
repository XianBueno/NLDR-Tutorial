import numpy as np
from numpy import sqrt
from numpy.matlib import zeros as matzeros
from numpy import linalg as LA

# According to: http://stackoverflow.com/questions/6684238/whats-the-fastest-way-to-find-eigenvalues-vectors-in-python
# we should consider using scipy.linalg instead for speed

class DiffusionMap:
    # This class will contain data, the diffusion embedding of the data,
    # and the Out-of-Sample-Extension (OoSE) mapping for new points.
    # We use "dfm" as an abbreviation of diffusion maps. 
    
    def __init__(self, pts):
        # pts for diffusion maps,
        # rows are observations, columns are features
        self.pts        = pts  
        self.pts_dfm    = None
        self.pts_nyst   = None
        self.pts_tree   = None
        self.N          = pts.shape[0]
        
     
        
    ###########################################################################
    ######################### Main User Functions #############################
    ###########################################################################
        
    def train(self, alpha=1, eps=0.1, t=1, p=None, h=lambda x:np.exp(-x)):
        # The training consists of finding the diffusion embedding for 
        # the training points. Method ends up storing the embedding as 
        # an array called self.pts_dfm (also to be used for Nystrom extension).
        # 
        #   alpha: 
        #     Diffusion parameter which is between 0 and 1 (inclusuve).                                 
        #   eps: 
        #     The proximity parameter.
        #   t: 
        #     How much time diffusion evolves, >0.
        #   p: 
        #     Means first p coordinates are kept, including the trivial one
        #   h: 
        #     A one-dimensional function used to define the isotropic kernel.
        #         k(x,y) = h(||x-y||^2/eps)
        
        self.alpha = alpha      # alpha is diffusion parameter, >=0 and <=1
        self.eps   = eps        # eps is the kernel proximity parameter
        self.t     = t          # t is time steps of diffusion, >0
        self.h     = h          # h is a 1D function to define isotropic kernel
        if p is None:           # p is number of cooordinates to keep
            self.p=self.N
        else:
            self.p=p          
        
        
        ##################################################################
        ### Phase 1: Construct Coifman-Kernel Matrix and Degree Vector ###
        ##################################################################
        
        self.Compute_KernSymMatrix_and_DegreeVec()
                
        #################################################
        ### Phase 2: Get Eigenvalues and Eigenvectors ###
        #################################################
        
        # Note: eigVec is matrix whose columns are the eigenvectors, as desired
        #print('Computing eigenvectors...')
        eigVal, eigVec = LA.eigh(self.M) 
        #print('Done computing eigenvectors.')
                                         
        # Sort eigVal in descending order (largest =1) and eigVec respectively
        idx = eigVal.argsort()[::-1]
        eigVal = eigVal[idx]
        eigVec = eigVec[:,idx]
        # Store these arrays
        self.eigval = eigVal[:self.p]
        self.eigvec = eigVec[:,:self.p]
        
        # Convert eigenvecs of M to eigenvecs of D^{-1}K (as was desired)
        # Do this by D^{-1/2}eigvector for each eigvector of M
        # These degree-adjusted eigenvectors are will form diffusion coords.
        eigVec_dfm = (eigVec.T/sqrt(self.deg)).T
        
        #####################################################
        ### Phase 3: Create Embedding for Training Points ###
        #####################################################
        
        # i-th pt -> [lam_1^t psi_1(i), lam_2^t psi_2(i),..., lam_n^t psi_n(i)]
        pts_dfm = np.zeros([self.N,self.p])
        for j in range(self.p):
            #j-th column is j-th eigVec (deg-adj). Just need to scale each column
            pts_dfm[:,j] = eigVal[j]**t * eigVec_dfm[:,j]
        
        # Truncate and store embedding, and eigenvalues
        self.pts_dfm = np.asarray(pts_dfm)  

    def computeP(self):
        # Computes the P matrix P(x_i,x_j) i.e. the markov transition matrix.
        # Avoids diffusion map computation.
        # Example use-case: Visualize transition matrix, data space denoising.
        
        # Need to compute at least to get q[j] values
        Compute_KernSymMatrix_and_DegreeVec(self,computeP=False)
        
        # Compute proxy vectors y_i with components j 
        y = np.zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(self.N):
                d = LA.norm(self.pts[i,:]-self.pts[j,:])
                y[i,j] = -d**2/self.eps - self.alpha * np.log(self.q[j])
        
        # Compute P(z_i,x_j) as softmax(y_i)_j
        # Shift y so max of each row is 0. 
        # Doesn't affect softmax value. Denominator is ensured to be >1.
        y = y - np.amax(y, axis=1, keepdims=True)
        exp_y = np.exp(y)
        P = exp_y/np.sum(exp_y, axis=1, keepdims=True)    # Softmax (numstable)
        
    def P_map(self,newpts):
        # Computes P(z,x_j) for each z in newpts.
        
        if self.alpha is None:  
            print('No alpha was chosen.')
        if self.eps is None:
            print('No eps was chosen.')
        if self.q is None:
            print('No q was computed')
            
        
        # Compute proxy vectors y_i with components j
        Npts = newpts.shape[0] 
        y = np.zeros([Npts,self.N])
        for i in range(Npts):
            for j in range(self.N):
                d = LA.norm(newpts[i,:]-self.pts[j,:])
                y[i,j] = -d**2/self.eps - self.alpha * np.log(self.q[j])
        
        # Compute P(z_i,x_j) as softmax(y_i)_j
        # Shift y so max of each row is 0. 
        # Doesn't affect softmax value. Denominator is ensured to be >1.
        y = y - np.amax(y, axis=1, keepdims=True)
        exp_y = np.exp(y)
        P = exp_y/np.sum(exp_y, axis=1, keepdims=True)    # Softmax (numstable)
                    
        return(P)
    
    def Nystrom(self, newpts,select=None,returnP=False): #New
        # Use the Nystrom extension formula to compute apprx of 
        # diffusion coords for previously unseen point x.
        " Need to deal with 0th extension since trivial "
        
        if self.pts_dfm is None:
            print("No embedding was ever learned! Use train() method first.")
        
        if select is None:  
            select = np.arange(0,self.p)
            
        PsyLambda = self.pts_dfm[:,select]/self.eigval[None,select]
            
        return self.P_map(newpts).dot(PsyLambda)
        
    def Nystrom_on_pts(self,saveP=False):
        if saveP:
            self.pts_nyst, self.P_nyst = self.Nystrom(self.pts,select=None,returnP=saveP)
        else:
            self.pts_nyst = self.Nystrom(self.pts,select=None,returnP=saveP)
    
    def CheckOutP(x):
        # Computes value of P(x,x_j) using the simplified formula
        # x can be a vector
        Npts = x.shape[0]
        Kzx = np.zeros([Npts,self.N])
        for i in range(Npts):
            for j in range(self.N):
                Kzx[i,j] = self.Kern1(newpts[i,:],self.pts[j,:])
        Z = (Kzx/self.q[None,:]**self.alpha).sum(axis=1)
        
        return Kzx/Z[:,None]
        
     
    ###########################################################################
    ################### Kernel-Related Functions ##############################
    ###########################################################################            
                                    
    # Isotropic kernel function.
    def Kern1(self,x,y):
        d = LA.norm(x-y)
        return self.h( d**2/self.eps )
    
    # Isotropic 'degree' of x. 'Row sum'.
    def d1(self,x):
        total = 0
        for i in range(self.N):
            total = total + self.Kern1(x,self.pts[i,:])
        return total
    
    # Anisotropic kernel function. 
    def Kern2(self,x,y):
        return self.Kern1(x,y)/(self.d1(x)*self.d1(y))**self.alpha
    
    # Anisotropic 'degree' of x. 'Row sum'.
    def d2(self,x):
        total = 0
        for i in range(self.N):
            total = total + self.Kern2(x,self.pts[i,:])
        return total
    
    # Symmetric normalization of Kern2. Plays role of kernel for KPCA and OoSE.
    def KernSym(self,x,y):
        return self.Kern2(x,y)/sqrt(self.d2(x)*self.d2(y))
    
    #######################################################
    ### Compute deg[i] = d(x_i) and K[i,j] = S(x_i,x_j) ###
    #######################################################
    def Compute_KernSymMatrix_and_DegreeVec(self,computeP=False):
        # To avoid redundant computations we construct most of the arrays
        " Can be sped up for alpha=0 (Laplacian Eigenmaps) "
        
        ##Step 1: Define Kernel matrix
        K = np.zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(self.N):
                K[i,j] = self.Kern1( self.pts[i,:], self.pts[j,:] )
        
        ##Step 2: Compute deg = row sums (isotropic degree)
        deg = K.sum(axis=1)
        self.q = deg
        
        ##Step 3: Coifman's normalization. Equivalent to Kern1 over data
        ##        (decouples statistics and geometry if alpha=1)
        K = K/(deg[:,None]*deg[None,:])**self.alpha
        
        ##Step 4: Compute row sums again (anisotropic degree)
        deg = K.sum(axis=1)
        self.deg = deg  
        
        ##Step 5: Symmetric normalization of K. 
        ##        Call it Coifman-Matrix, KernSym-Matrix, Coifman-Kernal Matrix
        self.M = K/sqrt(deg[:,None]*deg[None,:])   
        
        ##Step 6(optional): Compute Probability Kernel (POTENTIALLY UNSTABLE!)
        self.P = K/self.deg[:,None]   
        
        
