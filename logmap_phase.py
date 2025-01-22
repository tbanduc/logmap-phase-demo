import potpourri3d as pp3d
import pyvista as pv
import numpy as np
import scipy.sparse
from petsc4py import PETSc
import meshio
from tqdm import trange

length_tol = 1

# this is the monodomain conductivity
sigma = 0.4 # sigmai*sigmae/(sigmai+sigmae)

def Bmatrix(nodeCoords):
    "Create building block matrices"
    
    e1 = (nodeCoords[1,:] - nodeCoords[0,:])/np.linalg.norm(nodeCoords[1,:] - nodeCoords[0,:])
    e2 = ((nodeCoords[2,:] - nodeCoords[0,:]) - np.dot((nodeCoords[2,:] - nodeCoords[0,:]),e1)*e1)
    # normalize
    e2 = e2/np.linalg.norm(e2)

    x21 = np.dot(nodeCoords[1,:] - nodeCoords[0,:],e1)
    x13 = np.dot(nodeCoords[0,:] - nodeCoords[2,:],e1)
    x32 = np.dot(nodeCoords[2,:] - nodeCoords[1,:],e1)

    y23 = np.dot(nodeCoords[1,:] - nodeCoords[2,:],e2)
    y31 = np.dot(nodeCoords[2,:] - nodeCoords[0,:],e2)
    y12 = np.dot(nodeCoords[0,:] - nodeCoords[1,:],e2)

    J = x13*y23 - y31*x32

    B = np.array([[y23, y31, y12],[x32, x13, x21]])

    return B, J

def localStiffnessMatrix(B,J):
    "Assemble the local stiffness matrix"
    return (B.T @ B)/(2.*J)

def localMassMatrix(J):
    "Assemble the local mass matrix"
    return np.eye(3)*J/6

def assembleParabolic(pts,elm):
    "Assemble the global mass and stiffness matrices"

    I,J,Vm,Vk = [],[],[],[]
    for k,tri in enumerate(elm):
        j, i = np.meshgrid(tri,tri)
        I.extend(list(i.ravel()))
        J.extend(list(j.ravel()))
        B, Jac = Bmatrix(pts[tri])
        Mloc = localMassMatrix(Jac)
        Kloc = localStiffnessMatrix(B,Jac)
        Vm.extend(list(Mloc.ravel()))
        Vk.extend(list(Kloc.ravel()))

    n = pts.shape[0]
    M = scipy.sparse.coo_matrix((Vm,(I,J)),shape=(n,n)).tocsr()
    K = scipy.sparse.coo_matrix((Vk,(I,J)),shape=(n,n)).tocsr()

    # convert to PETSc
    Kp = PETSc.Mat()
    Kp.createAIJWithArrays((n,n),(K.indptr,K.indices,K.data))
    Kp.assemble()

    Mp = PETSc.Mat()
    Mp.createAIJWithArrays((n,n),(M.indptr,M.indices,M.data))
    Mp.assemble()

    return Mp, Kp

def initialize(M:PETSc.Mat,
               K:PETSc.Mat,
               phi:np.ndarray,
               Cm:float,
               dt:float,
               ndt:int) -> np.ndarray:

    """
    Applies diffusion step to phase

    In:
        M : mass matrix
        K : stiffness matrix
        phi : array of phase. Assumed to be of the form exp(ia), where i is the imaginary unit
        Cm : capacitance
        dt : time step
        ndt : number of steps 

    Out:
        np.angle(phi_real.array + 1j*phi_imag.array) : angle between real and imaginary component after diffusion
    """

    # lhs
    A_ =  M*Cm + K*dt*sigma

    # initialize real and imaginary components
    phi_real = A_.createVecRight()
    phi_imag = A_.createVecRight()

    # assign values
    phi_real.array[:] = phi.real
    phi_imag.array[:] = phi.imag

    # create list to iterate
    phis = [phi_real, phi_imag]

    for phi_i in phis:

        # create solver
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_)
        ksp.setType("cg")
        ksp.setConvergenceHistory()
        ksp.getPC().setType("hypre")

        # solve for ndt steps

        for _ in range(1, ndt):

            #rhs
            b_i = M*phi_i*Cm
            #solve
            ksp.solve(b_i, phi_i)

    return np.angle(phi_real.array + 1j*phi_imag.array)


def phase_s1(ps_loc:int,
             solver:pp3d.mesh.MeshVectorHeatSolver,
             mesh:pv.core.pointset.UnstructuredGrid,
             M:PETSc.Mat,
             K:PETSc.Mat,
             Cm:float,
             r0:float,
             dt:float = 0.05,
             ndt:int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Computes complex phase for a single spiral
    
    In:
        ps_loc : index corresponding to the source point
        solver : vector heat method solver for logmap
        mesh : mesh representing cardiac surface (cells are assumed to be triangles)
        M : mass matrix for diffusion
        K : stiffness matrix for diffusion
        Cm : capacitance
        r0 : cycle length
        dt : time step for diffusion
        ndt : number of iterations for diffusion

    Out:
        r : geodesic distance from source
        s : arccos(r0/r) outside of the ball of geodesic radius
        t : angle component (theta) from logmap
        phase : phase map exp(i((Theta(s) - s) + theta))
    """

    # assertion for ps_loc type
    assert type(ps_loc) == int, f"ps_loc must be an integer, not {type(ps_loc).__name__}"

    # logmap, r and theta (in range [-pi,pi])
    logmap = solver.compute_log_map(ps_loc) 
    r = np.hypot(logmap[:,0],logmap[:,1])
    t = np.arctan2(logmap[:,0],logmap[:,1])

    # initialize s and set values outside and inside of ball of geodesic radius r0
    s = np.empty_like(r)
    s[r > r0] = np.arccos(r0/r[r > r0]) # s = arccos(r0/r)
    s[r <= r0] = 0.0

    # Theta(s) = tan(s) - s
    phase = np.tan(s) - s 
    # Theta(s) + theta 
    phase += t 
    # Phi = exp(i(Theta(s) + theta))
    phase = np.exp(1j*phase) 
    
    # apply diffusion 
    phase = initialize(M = M, K = K, phi = phase, Cm = Cm, dt = dt, ndt = ndt)

    return r, s, t, phase

def phase_s2(ps_locs:list[int],
             signs:list[int],
             solver:pp3d.mesh.MeshVectorHeatSolver,
             mesh:pv.core.pointset.UnstructuredGrid,
             M:PETSc.Mat,
             K:PETSc.Mat,
             Cm:float,
             r0:float,
             dt:float = 0.05,
             ndt:int = 200) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    
    """
    Compute complex phase for multiple spirals with sequential synchronization 

    In:
        ps_loc : list of indices for source locations
        signs : list for rotation directions (1 or -1)
        solver : vector heat method solver for logmap
        mesh : mesh representing cardiac surface (cells are assumed to be triangles)
        M : mass matrix for diffusion 
        K : stiffness matrix for diffusion
        Cm : capacitance
        r0 : cycle length
        dt : time step for diffusion
        ndt : number of iterations for diffusion

    Out:
        r_min: minimum geodesic distance from sources
        s : arccos(r0/r_min)
        t : angle component (theta) from logmap. t = ti wherever ri = r_min
        phase : phase map exp(i((Theta(s) - s) + theta))
    """
       
    # assert that number of ps locations matches provided signs
    assert len(ps_locs) == len(signs), f"number of sources ({len(ps_locs)}) is different from number of charges ({len(signs)})"

    # function for pairwise synchronization between spirals
    def synch_seq(shift, *params):

        # i,j <- indices ; sign_i,sign_j <- signs of rotation
        i, j, sign_i, sign_j = params 
        # ti <- theta_i (fixed) ; tj <- theta_j + angle (we want to find the optimal shift)
        ti, tj = t_arr[i], (np.arctan2(logmap[j,:,0],logmap[j,:,1]) + shift) 
        # ri,rj <- geodesic distances
        ri, rj = r[i], r[j] 

        # initialize theta
        t = np.zeros_like(ti) 
        # assign values according to geodesic distance
        t[r_min == ri] = sign_i * ti[r_min == ri] 
        t[r_min == rj] = sign_j * tj[r_min == rj]
        
        # Theta(s) = tan(s) - s
        phase = np.tan(s) - s 
        # Theta(s) + theta 
        phase += t
        # Phi = exp(i(Theta(s) + theta))
        phase = np.exp(1j*phase) 

        # assign to mesh angle of phase
        mesh["phase"] = np.angle(phase) 

        # cut locus with tolerance length_tol
        cut_locus = 1*(np.abs(ri - rj) < length_tol) 
        # derivative
        dphase = mesh.compute_derivative(scalars = "phase")["gradient"] 

        # return sum of norm of gradient (we measure phase jumps with this)
        return np.sum(np.linalg.norm(dphase, axis = 1)*cut_locus) 
    
    # initialize logmap and geodesic distances array
    logmap = [] 
    r = []

    # iterate over sources list
    for ps in ps_locs: 

        # compute logmap
        lm = solver.compute_log_map(ps) 
        
        # append logmap and geodesic distance to corresponding lists
        logmap.append(lm) 
        r.append(np.hypot(lm[:,0],lm[:,1]))

     # cast to array
    logmap = np.array(logmap)
    r = np.array(r)

    # minimum geodesic distance
    r_min = np.min(r, axis = 0) 

    # initialize s and theta
    s = np.empty_like(r_min) 
    t = np.empty_like(r_min)

    # s = arccos(r0/r_min)
    s[r_min > r0] = np.arccos(r0/(r_min[r_min > r0]))
    s[r_min <= r0] = 0.0

    # initialize array for {sign_i*theta_i + shift_i}_i with shifts   
    t_arr = np.empty_like(r)
    # fix first spiral 
    t_arr[0,:] = signs[0]*np.arctan2(logmap[0,:,0],logmap[0,:,1])

    # iterate from first spiral to (N-1)th spiral
    for i in range(r.shape[0]-1): 
        
        # find global optimum for angle shift of theta_(i+1)
        shift_i = scipy.optimize.brute(synch_seq, ranges = [(slice(0, 2*np.pi, 0.05))], args = (i,
                                                                                                i+1,
                                                                                                signs[i],
                                                                                                signs[i+1])) 
        
        # assign sign_(i+1)*theta_(i+1) + shift_(i+1)
        t_arr[i+1] = signs[i+1]*(np.arctan2(logmap[i+1,:,0],logmap[i+1,:,1]) + shift_i[0]) 

    # assign theta <- sign_i*theta_i + shift_i at the points where ri = r_min
    for i in range(len(r)):
        ri = r[i]
        t[r_min == ri] = t_arr[i][r_min == ri]
    
    # Theta(s) = tan(s) - s
    phase = np.tan(s) - s 
    # Theta(s) + theta 
    phase += t 
    # Phi = exp(i(Theta(s) + theta))
    phase = np.exp(1j*phase) 

    # apply diffusion 
    phase = initialize(M = M, K = K, phi = phase, Cm = Cm, dt = dt, ndt = ndt) 

    return r_min, s, t, phase

def phase_s4(ps_locs:list[list[int]],
             signs:list[list[int]],
             solver:pp3d.mesh.MeshVectorHeatSolver,
             mesh:pv.core.pointset.UnstructuredGrid,
             M:PETSc.Mat,
             K:PETSc.Mat,
             Cm:float,
             r0:float,
             dt:float = 0.05,
             ndt:int = 200) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    
    """
    Compute complex phase for groups of spirals with sequential synchronization within and between groups

    In:
        ps_loc : list of indices for source locations. Assumed to be in format [[ps_0,0 ; ... ; ps_0,k0], [ps_1,0 ; ... ; ps_1,k1], ... , [ps_N,0 ; ... ; ps_M,kM]]
        signs : list for rotation directions (1 or -1). Assumed to be in same format as ps_loc
        solver : vector heat method solver for logmap
        mesh : mesh representing cardiac surface (cells are assumed to be triangles)
        M : mass matrix for diffusion 
        K : stiffness matrix for diffusion
        Cm : capacitance
        r0 : cycle length
        dt : time step
        ndt : number of iterations

    Out:
        r_min: minimum geodesic distance from sources
        s : arccos(r0/r_min)
        t : angle component (theta) from logmap. t = ti wherever ri = r_min
        phase : phase map exp(i((Theta(s) - s) + theta))
    """

    # assert that number of ps locations matches provided signs
    assert len(np.array(ps_locs).flatten()) == len(np.array(signs).flatten()), f"number of sources ({len(np.array(ps_locs).flatten())}) is different from number of charges ({len(np.array(signs).flatten())})"
    
    # function for pairwise synchronization between groups of spirals
    def synch_seq(shift, *params):

        # i,j <- indices of groups
        i, j = params 
        # ti <-  theta_i of group i ; tj <- theta_j + angle of group j
        ti, tj = t_arr[i], t_arr[j] + shift 
        # ri <-  minimum geodesic distance within group i ; rj <- ri <-  minimum geodesic distance within group j
        ri, rj = r_arr[i], r_arr[j]

        # initialize theta
        t = np.zeros_like(ti) 

        # assign values according to geodesic distance
        t[r_min == ri] = ti[r_min == ri]
        t[r_min == rj] = tj[r_min == rj]

        # Theta(s) = tan(s) - s
        phase = np.tan(s) - s 
        # Theta(s) + theta 
        phase += t 
        # Phi = exp(i(Theta(s) + theta))
        phase = np.exp(1j*phase) 

        # assign to mesh angle of phase
        mesh["phase"] = np.angle(phase)

        # cut locus with tolerance length_tol
        cut_locus = 1*(np.abs(ri - rj) < length_tol) 
        # derivative
        dphase = mesh.compute_derivative(scalars = "phase")["gradient"] 

        # return sum of norm of gradient (we measure phase jumps with this)
        return np.sum(np.linalg.norm(dphase, axis = 1)*cut_locus) 
    
    # initialize lists for synched groups
    r_arr, s_arr, t_arr = [], [], [] 

    # iterate over groups of ps
    for i in range(len(ps_locs)): 

        # synchronize group and return corresponding arrays. Here, ndt = 0 since diffusion is applied once final phase is computed
        r_, s_, t_, phase_ = phase_s2(ps_locs[i],
                                      signs = signs[i],
                                      solver = solver,
                                      mesh = mesh,
                                      M = M,
                                      K = K,
                                      Cm = Cm,
                                      r0 = r0,
                                      dt = dt,
                                      ndt = 0) 
        r_arr.append(r_), s_arr.append(s_), t_arr.append(t_)

    # cast to array
    r_arr, s_arr, t_arr = np.array(r_arr), np.array(s_arr), np.array(t_arr)

    # minimum geodesic distance
    r_min = np.min(r_arr, axis = 0)
    # initialize s and theta
    s = np.empty_like(r_min) 
    t = np.empty(t_arr.shape[1])

    # iterate over groups of ps
    for i in range(len(ps_locs)):
        
        # assign corresponding values of s according to geodesic distance
        s[r_min == r_arr[i]] = s_arr[i][r_min == r_arr[i]] 

    # initialize array for thetas with shift across groups
    t_arr_new = np.empty_like(r_arr) 
    # fix first group of spirals
    t_arr_new[0] = t_arr[0] 

    # iterate from first group to (M-1)th group
    for i in range(r_arr.shape[0]-1): 
        
        # obtain shift angle of the group
        shift_i = scipy.optimize.brute(synch_seq, ranges = [(slice(0, 2*np.pi, 0.05))], args = (i,i+1)) 

        # assign theta + shift for the subsequent group
        t_arr_new[i+1] = t_arr[i+1] + shift_i 

    # iterate over groups of ps
    for i in range(len(r_arr)):
        # assign theta <- theta_i + shift_i at the points where ri = r_min
        ri = r_arr[i]
        t[r_min == ri] = t_arr_new[i][r_min == ri]

    # Theta(s) = tan(s) - s  
    phase = np.tan(s) - s
    # Theta(s) + theta 
    phase += t
    # Phi = exp(i(Theta(s) + theta))
    phase = np.exp(1j*phase)

    phase = initialize(M = M, K = K, phi = phase, Cm = Cm, dt = dt, ndt = ndt) # apply diffusion

    return r_min, s, t, phase

def run_monodomain(phi_0:np.ndarray,
                   mesh:pv.core.pointset.UnstructuredGrid,
                   u_state:np.ndarray,
                   r_state:np.ndarray,
                   T_grid:np.ndarray,
                   Cm:np.ndarray,
                   Vrest:float = 0.0,
                   Vdep:float  = 1,
                   Vthre:float = 0.15,
                   c1:float  = 8,
                   c2:float = 1,
                   gamma:float = 0.002,
                   mu1:float = 0.2,
                   mu2:float = 0.3,
                   b:float = 0.5,
                   dt:float = 0.01,
                   Tend:float = 500.0,
                   fout:str = "monodomain_spirals_logmap.xdmf") -> None:

    """
    Run monodomain simulation for provided phase and save correpsonding .xdmf file

    In:
        phi_0 : initialization phase. Assumed to be in the range [-pi,pi]
        mesh : mesh representing cardiac surface (cells are assumed to be triangles)
        u_state : action potential profile for voltage
        r_state : action potential profile for gating variable
        T_grid : grid for interpolation
        Cm : capacitance
        Vrest, Vdep, Vthre, c1, c2, gamma, mu1, mu2 : ionic model parameters
        dt : timestep for simulation
        Tend : time of simulation
        fout : name of .xdmf filename
    
    Out:
        None 
    """

    # set number of iterations
    ndt = int(np.rint(Tend/dt)+1)

    # extract points and cells from geometry
    points = mesh.points
    cells = mesh.cells.reshape(-1,4)[:,1:]

    # mass and stiffness matrices
    M, K = assembleParabolic(points, cells)
    # lhs
    A =  M*Cm + K*dt*sigma

    # create solution vectors
    u = A.createVecRight()
    r = A.createVecRight()

    # create solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.setConvergenceHistory()
    ksp.getPC().setType("hypre")

    # Aliev-Panfilov ionic model 
    fion = lambda Vm , r: (Vm-Vdep)*(Vm-Vrest)*(Vm-Vthre)*c1 + Vm*r*c2
    rdot = lambda Vm, r: (r*mu1/(Vm + mu2) + gamma)*(-r -Vm*c1*(Vm - Vthre - 1))

    # interpolate states to scaled phase
    u_inic = np.interp((phi_0+np.pi)/(2*np.pi), T_grid, u_state[::-1])
    r_inic = np.interp((phi_0+np.pi)/(2*np.pi), T_grid, r_state[::-1])

    mesh["phi0"] = phi_0
    mesh["u0"] = u_inic
    
    # set initial condition
    u.array[:] = u_inic
    r.array[:] = r_inic

    # list for saving solutions across iterations
    Us = [u.copy().array]
    Rs = [r.copy().array]

    with meshio.xdmf.TimeSeriesWriter(fout) as writer:
        writer.write_points_cells(points, [("triangle",cells)])
        writer.write_data(0.0, point_data={"u": u.array,"r": r.array})

        # iterate over time
        for i in trange(1,ndt):
            t = dt * i
            # compute ionic and source currents
            Iion = fion(u,r)
            r = r + rdot(u,r)*dt
            # rhs
            b = M*(u*Cm - Iion*dt)
            # solve Ax = b and save solution to u
            ksp.solve(b, u)

            # write solution every 50 steps (change this parameter so the solutions look closer)
            if (i % 50) == 0: 
            
                Us.append(u.copy().array)
                Rs.append(r.copy().array)
                writer.write_data(t, point_data={"u": u.array, "r": r.array})

