# we will convert this only to a 2D example on the regular Cartesian grid, and we will also save auxiliary state i.e. velocity so that we can plot it later.

# %%
import numpy as np
from clawpack import riemann
import os
from matplotlib import colormaps as cm

# %% 

# def qinit(state):
#     X, Y = state.grid.p_centers
#     state.q[0,:,:] = 0.9*(0.4<X)*(X<0.6)*(0.1<=Y)*(Y<0.3) + 0.1

raw_data = np.load("/home/ajivani/WLROM_new/WhiteLight/validation_data/CR2161_validation_PolarTensor.npy")
raw_data.shape

sample_rd = raw_data[:126, :, 25, 20]

def mapc2p_annulus(xc, yc):
    """
    Specifies the mapping to curvilinear coordinates.

    Inputs: c_centers = Computational cell centers
                 [array ([Xc1, Xc2, ...]), array([Yc1, Yc2, ...])]

    Output: p_centers = Physical cell centers
                 [array ([Xp1, Xp2, ...]), array([Yp1, Yp2, ...])]
    """  
    p_centers = []

    # Polar coordinates (first coordinate = radius,  second coordinate = theta)
    p_centers.append(xc[:]*np.cos(yc[:]))
    p_centers.append(xc[:]*np.sin(yc[:]))
    
    return p_centers

# def calc_interpolation_coefficients_in_time(dtInitial, tFinal, tStartSim, tEndSim):
#     # times_compute = np.linspace(0.0, tFinal, num_output_times)
#     times_compute = 
#     times_sim = 

def qinit(state):
    """
    Initialize with two Gaussian pulses.
    """
    # # First gaussian pulse
    # A1     = 1.    # Amplitude
    # beta1  = 40.   # Decay factor
    # r1     = -0.5  # r-coordinate of the center
    # theta1 = 0.    # theta-coordinate of the center

    # # Second gaussian pulse
    # A2     = -1.   # Amplitude
    # beta2  = 40.   # Decay factor
    # r2     = 0.5   # r-coordinate of the centers
    # theta2 = 0.    # theta-coordinate of the centers

    R, Theta = state.grid.p_centers
    # state.q[0,:,:] = A1*np.exp(-beta1*(np.square(R-r1) + np.square(Theta-theta1)))\
                #    + A2*np.exp(-beta2*(np.square(R-r2) + np.square(Theta-theta2)))

    state.q[0,:,:] = raw_data[2:126, :, 25, 20]

def edge_velocities_and_area(R_nodes,Theta_nodes,dx,dy):
    """This routine fills in the aux arrays for the problem:

        aux[0,i,j] = u-velocity at left edge of cell (i,j)
        aux[1,i,j] = v-velocity at bottom edge of cell (i,j)
        aux[2,i,j] = physical area of cell (i,j) (relative to area of computational cell)
    """
    # X, Y = state.grid.p_centers

    u0 = 20
    u1 = 10
    # u1 = 0.0

    mx = R_nodes.shape[0]-1
    my = R_nodes.shape[1]-1

    # bottom left corners
    Xp0 = R_nodes[:mx,:my]
    Yp0 = Theta_nodes[:mx,:my]

    # top left corners
    Xp1 = R_nodes[:mx,1:]
    Yp1 = Theta_nodes[:mx,1:]

    # bottom right
    Xp2 = R_nodes[1:,1:]
    Yp2 = Theta_nodes[1:,1:]

    # top right
    Xp3 = R_nodes[1:,:my]
    Yp3 = Theta_nodes[1:,:my]

    aux = np.empty((3,mx,my), order='F')

    # aux[0, :mx, :my] = (-u0 * Xp0 / np.sqrt(Xp0**2 + Yp0**2)) - u1 * Xp0

    aux[0, :mx, :my] = u0 + u1 * np.sqrt(Xp0**2 + Yp0**2)
    aux[1, :mx, :my] = 0.0
    # aux[1, :mx, :my] = (-u0 * Yp0 / np.sqrt(Xp0**2 + Yp0**2)) - u1 * Yp0

    # rrr = np.sqrt(Xp0**2 + Yp0**2)

    # u = np.zeros_like(Xp0)
    # u[Xp0 >=0] = u0 * Xp0[Xp0 >=0] / rrr[Xp0 >=0]
    # u[Xp0 < 0] = -u0 * Xp0[Xp0 < 0] / rrr[Xp0 < 0]

    # v = np.zeros_like(Yp0)
    # v[Yp0 >= 0] = u0 * Yp0[Yp0 >= 0] / rrr[Yp0 >= 0]
    # v[Yp0 < 0] = -u0 * Yp0[Yp0 < 0] / rrr[Yp0 < 0]

    # aux[0, :, :] = u
    # aux[1, :, :] = v



    # Compute area of the physical element
    area = 1./2.*( (Yp0+Yp1)*(Xp1-Xp0) +
                     (Yp1+Yp2)*(Xp2-Xp1) +
                        (Yp2+Yp3)*(Xp3-Xp2) +
                        (Yp3+Yp0)*(Xp0-Xp3) )
    
    aux[2, :mx, :my] = area / (dx * dy) # capacity function 

    return aux

def ghost_velocities_upper(state,dim,t,qbc,auxbc,num_ghost):
    """
    Set the velocities for the ghost cells outside the outer radius of the annulus.
    In the computational domain, these are the cells at the top of the grid.
    """
    grid=state.grid
    if dim == grid.dimensions[0]:
        dx, dy = grid.delta
        R_nodes,Theta_nodes = grid.p_nodes_with_ghost(num_ghost=2)

        auxbc[:,-num_ghost:,:] = edge_velocities_and_area(R_nodes[-num_ghost-1:,:],Theta_nodes[-num_ghost-1:,:],dx,dy)

    else:
        raise Exception('Custom BC for this boundary is not appropriate!')


def ghost_velocities_lower(state,dim,t,qbc,auxbc,num_ghost):
    """
    Set the velocities for the ghost cells outside the inner radius of the annulus.
    In the computational domain, these are the cells at the bottom of the grid.
    """
    grid=state.grid
    if dim == grid.dimensions[0]:
        dx, dy = grid.delta
        R_nodes,Theta_nodes = grid.p_nodes_with_ghost(num_ghost=2)

        auxbc[:,0:num_ghost,:] = edge_velocities_and_area(R_nodes[0:num_ghost+1,:],Theta_nodes[0:num_ghost+1,:],dx,dy)

    else:
        raise Exception('Custom BC for this boundary is not appropriate!')



# def auxinit(state):
#     # Initialize petsc Structures for aux
#     # xc=state.grid.x.centers
#     X, Y = state.grid.p_centers

#     rnodes, theta_nodes = state.grid.p_nodes
#     dx, dy = state.grid.delta
#     mx = rnodes.shape[0] - 1
#     my = rnodes.shape[1] - 1

#     # u0 = 5 # imagine radial velocity is fixed and doesn't change with r for case 1.
#     # ur_vals = u0 + 
#     u0 = 5
#     # u1 = 2.5 # for now we will keep constant radial velocity.

#     # state.aux[0, :, :] = 0.0
#     # state.aux[1, :, :] = np.sin(2.*np.pi*X)+2
#     # state.aux[1, :, :] = (0.5 - 0.5 * Y) + 0.1

#     # Bottom-left corners
#     Xp0 = rnodes[:mx,:my]
#     Yp0 = theta_nodes[:mx,:my]

#     # Top-left corners
#     Xp1 = rnodes[:mx,1:]
#     Yp1 = theta_nodes[:mx,1:]

#     # Top-right corners
#     Xp2 = rnodes[1:,1:]
#     Yp2 = theta_nodes[1:,1:]

#     # Top-left corners
#     Xp3 = rnodes[1:,:my]
#     Yp3 = theta_nodes[1:,:my]

#     # Compute area of the physical element
#     area = 1./2.*( (Yp0+Yp1)*(Xp1-Xp0) +
#                    (Yp1+Yp2)*(Xp2-Xp1) +
#                    (Yp2+Yp3)*(Xp3-Xp2) +
#                    (Yp3+Yp0)*(Xp0-Xp3) )
    
#     state.aux[0, :, :] = -u0 * X / np.sqrt(X**2 + Y**2)

#     # state.aux[0, :, :] = (u0 * X / np.sqrt(X**2 + Y**2)) + u1 * Y
#     state.aux[1, :, :] = -u0 * Y / np.sqrt(X**2 + Y**2)
#     state.aux[2, :, :] = area / (dx * dy)






# def inlet_BC(state, dim, t, qbc, auxbc, num_ghost):
def inlet_BC(state, dim, t, qbc, auxbc, num_ghost):
    """
    Set inflow conditions with interpolation at the inner radial boundary.
    """
 
    # Get the radial and angular coordinates
    R, Theta = state.grid.p_centers

    # time_ratio = int(t // 0.015625) # finds the exact or closest index of time in raw data to set the IC.

    # Find the time index for the raw data
    time_index = int(t // 0.015625)  # after standardizing the time scale for this particular sim, 0.015625 s is the gap between observations.
    print("Time index: ", time_index)

    st_index_sample_data = 25
    sample_rd_all = raw_data[2:126, :, st_index_sample_data:, 20]
    sample_rd_edge = raw_data[:2, :, st_index_sample_data:, 20]

    last_time_index = sample_rd_all.shape[2] - 1

    # Interpolate between the two time steps if necessary
    if time_index >= (raw_data.shape[2] - st_index_sample_data - 1):
        # Use the last available time step
        qbc[0, :num_ghost, num_ghost:-num_ghost] = sample_rd_edge[:, :, last_time_index]
        # auxbc[0, :num_ghost, :] = np.zeros_like(auxbc[0, :num_ghost, :])
    else:
        t1 = time_index * 0.015625
        t2 = (time_index + 1) * 0.015625
        alpha = (t - t1) / (t2 - t1)
        print("Weight for previous time step: ", 1 - alpha)
        qbc[0, :num_ghost, num_ghost:-num_ghost] = (1 - alpha) * sample_rd_edge[:, :, time_index] + alpha * sample_rd_edge[:, :, time_index + 1]
        # auxbc[0, :num_ghost, :] = np.zeros_like(auxbc[0, :num_ghost, :])

        # auxbc[:] = np.zeros_like(auxbc)  # Set auxiliary values to zero at the boundary (is this well-posed?????) # for upper boundary perhaps regular extrap is fine!

def setup(use_petsc=False,outdir='./_output',solver_type='classic'):
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    solver = pyclaw.ClawSolver2D(riemann.vc_advection_2D) # vc_advection for variable coefficient advection!

    solver.dimensional_split = False
    solver.transverse_waves = 2
    solver.order = 2
    # solver.limiters = pyclaw.limiters.tvd.vanleer

    # https://github.com/clawpack/pyclaw/blob/151aefa92a613d952f3b49070cd80826a81b1178/src/pyclaw/limiters/tvd.py
    solver.limiters = pyclaw.limiters.tvd.MC
    # Use MC / Koren


    # solver.bc_lower[0] = pyclaw.BC.extrap
    # solver.bc_upper[0] = pyclaw.BC.extrap

    # solver.bc_lower[0] = pyclaw.BC.periodic
    # solver.bc_upper[0] = pyclaw.BC.periodic
    # solver.aux_bc_lower[0] = pyclaw.BC.periodic
    # solver.aux_bc_upper[0] = pyclaw.BC.periodic

    solver.bc_lower[0] = pyclaw.BC.custom
    solver.bc_upper[0] = pyclaw.BC.extrap

    solver.aux_bc_lower[0] = pyclaw.BC.custom
    solver.aux_bc_upper[0] = pyclaw.BC.custom


    # solver.aux_bc_lower[0] = pyclaw.BC.extrap
    # solver.aux_bc_upper[0] = pyclaw.BC.extrap

    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    solver.aux_bc_lower[1] = pyclaw.BC.periodic
    solver.aux_bc_upper[1] = pyclaw.BC.periodic

    # solver.aux_bc_lower[1] = None
    # solver.aux_bc_upper[1] = None

    solver.user_bc_lower = inlet_BC
    solver.user_aux_bc_lower = ghost_velocities_lower
    solver.user_aux_bc_upper = ghost_velocities_upper

    # solver.bc_lower[1] = pyclaw.BC.periodic
    # solver.bc_upper[1] = pyclaw.BC.periodic

    # solver.bc_lower[1] = pyclaw.BC.extrap
    # solver.bc_upper[1] = pyclaw.BC.extrap

    # 2d var velocity
    solver.dt_initial = 0.005
    solver.dt_variable=True

    # solver.dt_initial = 0.005
    # solver.dt_variable = False # 0.005 should used for all timesteps - see https://github.com/clawpack/pyclaw/blob/151aefa92a613d952f3b49070cd80826a81b1178/src/pyclaw/solver.py#L88


    solver.cfl_max = 0.5
    solver.cfl_desired = 0.4
    r_lower = 3.903302085636277
    r_upper = 23.465031329617336
    # m_r = 126
    m_r = 124

    theta_lower = 0.0
    theta_upper = np.pi*2.0
    m_theta = 512

    r = pyclaw.Dimension(r_lower, r_upper, m_r, name='x')
    theta = pyclaw.Dimension(theta_lower, theta_upper, m_theta, name='y')
    domain = pyclaw.Domain([r, theta])
    domain.grid.mapc2p = mapc2p_annulus
    domain.grid.num_ghost = solver.num_ghost

    num_eqn = 1
    num_aux = 3

    # state = pyclaw.State(domain, num_eqn, num_aux)
    state = pyclaw.State(domain, num_eqn)

    qinit(state)
    # auxinit(state) # remove if using ghost velocity functions.

    dx, dy = state.grid.delta
    p_corners = state.grid.p_nodes
    state.aux = edge_velocities_and_area(p_corners[0],p_corners[1],dx,dy)
    state.index_capa = 2

    claw = pyclaw.Controller()
    # claw.tfinal = 2.0
    # claw.tfinal = 1.5
    claw.tfinal = 1.04
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True

    # write solution for 50 steps.
    claw.num_output_times = 50
    # other options: check https://www.clawpack.org/pyclaw/output.html


    # writing values in aux array?
    claw.write_aux_init = True
    # claw.write_aux_always = True
    # https://www.clawpack.org/pyclaw/output.html

    # check solution at specific coordinates (adding gauges) - https://github.com/clawpack/pyclaw/blob/151aefa92a613d952f3b49070cd80826a81b1178/examples/psystem_2d/psystem_2d.py#L221
    return claw

# %%
def setplot(plotdata):
    """ 
    Plot solution using VisClaw.
    """
    from mapc2p_annulus import mapc2p_annulus
    import numpy as np
    # from clawpack.visclaw import colormaps

    plotdata.clearfigures()  # clear any old figures,axes,items data
    # plotdata.plotdir = '_plots_2d_var_whitelight2'
    # plotdata.plotdir = '_plots_2d_var_whitelight9'
    # plotdata.plotdir = '_plots_2d_var_whitelight11'
    # plotdata.plotdir = '_plots_2d_var_whitelight12'
    plotdata.plotdir = '_plots_2d_var_whitelight15'

    # plotdata.mapc2p = mapc2p_annulus
    plotdata.mapc2p = mapc2p_annulus

    # Figure for pcolor plot
    # plotfigure = plotdata.new_plotfigure(name='q[0]', figno=1)
    plotfigure = plotdata.new_plotfigure(name='q[0]', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = 'q[0]'
    plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 0

    cmap = cm.get_cmap('viridis')
    plotitem.pcolor_cmap = cmap
    # plotitem.pcolor_cmin = -1.0
    # plotitem.pcolor_cmax = 1.0
    plotitem.add_colorbar = True
    plotitem.MappedGrid = True


    return plotdata

# %%
pyclaw_kwargs = {'use_petsc':False,
                'outdir':'./_output',
                'solver_type':'classic'}

clawSolution = setup()

# %%
status = clawSolution.run()
status

# %%
from clawpack import pyclaw

outdir = pyclaw_kwargs.get('outdir','./_output')
htmlplot = pyclaw_kwargs.get('htmlplot',False) # return False if key doesn't exist
iplot    = pyclaw_kwargs.get('iplot',False)
outdir, htmlplot, iplot

pyclaw.plot.html_plot(outdir=outdir,setplot=setplot)
# %%
### Solution for plotting aux data instead of q data????
# https://github.com/clawpack/visclaw/blob/2f439bb7a2669eb76eeec20265994dd0d2377168/src/python/visclaw/data.py#L40

# or just try this: plotitem.plot_var = velocity
# or this: plotitem.plot_var = 1?

        # if controller:
        #     controller.plotdata = self
        #     # inherit some values from controller
        #     self.add_attribute('rundir',copy.copy(controller.rundir))
        #     self.add_attribute('outdir',copy.copy(controller.outdir))
        #     if len(controller.frames)>0:
        #         for i,frame in enumerate(controller.frames):
        #             self.framesoln_dict[str(i)] = frame
        #     self.add_attribute('format',copy.copy(controller.output_format))
        #     self.add_attribute('file_prefix',copy.copy(controller.output_file_prefix))