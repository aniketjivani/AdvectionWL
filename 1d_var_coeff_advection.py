# we will convert this only to a 2D example on the regular Cartesian grid, and we will also save auxiliary state i.e. velocity so that we can plot it later.

# %%
import numpy as np
from clawpack import riemann
import os
from matplotlib import colormaps as cm

# %% 

def qinit(state):

    # Initial Data parameters
    ic = 3
    beta = 100.
    gamma = 0.
    x0 = 0.3
    x1 = 0.7
    x2 = 0.9

    x =state.grid.x.centers
    
    # Gaussian
    qg = np.exp(-beta * (x-x0)**2) * np.cos(gamma * (x - x0))
    # Step Function
    qs = (x > x1) * 1.0 - (x > x2) * 1.0
    
    if   ic == 1: state.q[0,:] = qg
    elif ic == 2: state.q[0,:] = qs
    elif ic == 3: state.q[0,:] = qg + qs

def auxinit(state):
    # Initialize petsc Structures for aux
    xc=state.grid.x.centers
    state.aux[0,:] = np.sin(2.*np.pi*xc)+2 # is it okay to initialize velocities at cell centers instead of edges?



# def qinit(state):
#     """
#     Initialize with two Gaussian pulses.
#     """
#     # First gaussian pulse
#     A1     = 1.    # Amplitude
#     beta1  = 40.   # Decay factor
#     r1     = -0.5  # r-coordinate of the center
#     theta1 = 0.    # theta-coordinate of the center

#     # Second gaussian pulse
#     A2     = -1.   # Amplitude
#     beta2  = 40.   # Decay factor
#     r2     = 0.5   # r-coordinate of the centers
#     theta2 = 0.    # theta-coordinate of the centers

#     # R, Theta = state.grid.p_centers
#     # state.q[0,:,:] = A1*np.exp(-beta1*(np.square(R-r1) + np.square(Theta-theta1)))\
#     #                + A2*np.exp(-beta2*(np.square(R-r2) + np.square(Theta-theta2)))

#     R, Theta = state.grid.p_centers
#     state.q[0,:,:] = A1*np.exp(-beta1*(np.square(R-r1) + np.square(Theta-theta1)))

#     # state.q[0,:,:] = sample_rd


# def ghost_velocities_upper(state,dim,t,qbc,auxbc,num_ghost):
#     """
#     Set the velocities for the ghost cells outside the outer radius of the annulus.
#     In the computational domain, these are the cells at the top of the grid.
#     """
#     grid=state.grid
#     if dim == grid.dimensions[0]:
#         dx, dy = grid.delta
#         R_nodes,Theta_nodes = grid.p_nodes_with_ghost(num_ghost=2)

#         auxbc[:,-num_ghost:,:] = edge_velocities_and_area(R_nodes[-num_ghost-1:,:],Theta_nodes[-num_ghost-1:,:],dx,dy)

#     else:
#         raise Exception('Custom BC for this boundary is not appropriate!')


# def ghost_velocities_lower(state,dim,t,qbc,auxbc,num_ghost):
#     """
#     Set the velocities for the ghost cells outside the inner radius of the annulus.
#     In the computational domain, these are the cells at the bottom of the grid.
#     """
#     grid=state.grid
#     if dim == grid.dimensions[0]:
#         dx, dy = grid.delta
#         R_nodes,Theta_nodes = grid.p_nodes_with_ghost(num_ghost=2)

#         auxbc[:,0:num_ghost,:] = edge_velocities_and_area(R_nodes[0:num_ghost+1,:],Theta_nodes[0:num_ghost+1,:],dx,dy)

#     else:
#         raise Exception('Custom BC for this boundary is not appropriate!')


# def edge_velocities_and_area(R_nodes, Theta_nodes, dx, dy):
#     """This routine fills in the aux arrays for the problem:

#         aux[0,i,j] = u-velocity at left edge of cell (i,j)
#         aux[1,i,j] = v-velocity at bottom edge of cell (i,j)
#         aux[2,i,j] = physical area of cell (i,j) (relative to area of computational cell)
#     """
#     mx = R_nodes.shape[0]-1
#     my = R_nodes.shape[1]-1
#     aux = np.empty((3,mx,my), order='F')

#     # Bottom-left corners
#     Xp0 = R_nodes[:mx,:my]
#     Yp0 = Theta_nodes[:mx,:my]

#     # Top-left corners
#     Xp1 = R_nodes[:mx,1:]
#     Yp1 = Theta_nodes[:mx,1:]

#     # Top-right corners
#     Xp2 = R_nodes[1:,1:]
#     Yp2 = Theta_nodes[1:,1:]

#     # Top-left corners(bottom right?)
#     Xp3 = R_nodes[1:,:my]
#     Yp3 = Theta_nodes[1:,:my]
#     u0 = 2.5
#     u1 = 1.5
#     #     u1 = 50

#     aux[0, :mx, :my] = u0 * (Xp1 / (np.sqrt(Xp1**2 + Yp1**2))) + u1 * Xp1
#     aux[1, :mx, :my] = u0 * (Yp3 / (np.sqrt(Xp3**2 + Yp3**2))) + u1 * Yp3


#     # Compute area of the physical element
#     area = 1./2.*( (Yp0+Yp1)*(Xp1-Xp0) +
#                    (Yp1+Yp2)*(Xp2-Xp1) +
#                    (Yp2+Yp3)*(Xp3-Xp2) +
#                    (Yp3+Yp0)*(Xp0-Xp3) )
    
#     # Compute capa 
#     aux[2,:mx,:my] = area/(dx*dy)

#     return aux

def setup(use_petsc=False,outdir='./_output',solver_type='classic'):
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type == 'classic':
        # people use dimensional_split if no transverse solver is defined??? see https://github.com/clawpack/pyclaw/blob/151aefa92a613d952f3b49070cd80826a81b1178/src/pyclaw/classic/solver.py#L411 - if dimensional split is false, then transverse riemann solver is used. see 21.6 of LeVeque's book.

        solver = pyclaw.ClawSolver1D(riemann.vc_advection_1D_py.vc_advection_1D) # vc_advection for variable coefficient advection!

        # solver = pyclaw.ClawSolver2D(riemann.advection_2D) # vc_advection for variable coefficient advection!
        # solver.dimensional_split = 1 
        # solver.transverse_waves = 2
        # solver.order = 2 # 1st order is Godunov's method and 2nd order is Lax-Wendroff-LeVeque method (default is 2)
    
    
    # solver.limiters = pyclaw.limiters.tvd.vanleer
    solver.kernel_language = 'Python'

    solver.limiters = pyclaw.limiters.tvd.MC



    # solver.bc_lower[0] = pyclaw.BC.extrap
    # solver.bc_upper[0] = pyclaw.BC.extrap

    solver.bc_lower[0] = 2
    solver.bc_upper[0] = 2

    solver.aux_bc_lower[0] = 2
    solver.aux_bc_upper[0] = 2

    # solver.bc_lower[1] = pyclaw.BC.periodic
    # solver.bc_upper[1] = pyclaw.BC.periodic

    # solver.bc_lower[1] = pyclaw.BC.extrap
    # solver.bc_upper[1] = pyclaw.BC.extrap

    # solver.dt_initial = 0.1
    # solver.cfl_max = 0.5
    # solver.cfl_desired = 0.4

    solver.cfl_max = 1.0
    solver.cfl_desired = 0.9


    # mx = 50
    # my = 50

    # x = pyclaw.Dimension(0.0, 1.0, mx, name='x')
    # y = pyclaw.Dimension(0.0, 1.0, my, name='y')
    # domain = pyclaw.Domain([x, y])

    xlower=0.0; xupper=1.0; mx=100
    x    = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain(x)

    # r_lower = 0.2
    # r_upper = 1.0
    # m_r = 40

    # theta_lower = 0.0
    # theta_upper = np.pi*2.0
    # m_theta = 120

    # r     = pyclaw.Dimension(r_lower,r_upper,m_r,name='r')
    # theta = pyclaw.Dimension(theta_lower,theta_upper,m_theta,name='theta')
    # domain = pyclaw.Domain([r,theta])
    # domain.grid.mapc2p = mapc2p_annulus
    # domain.grid.num_ghost = solver.num_ghost

    num_eqn = 1
    num_aux = 1
    state = pyclaw.State(domain,num_eqn, num_aux)

    # state.problem_data['u'] = 0.0
    # state.problem_data['v'] = 1.0
    # state.problem_data['v'] = 0.5

    qinit(state)
    auxinit(state)

    # dx, dy = state.grid.delta
    # p_corners = state.grid.p_nodes
    # state.aux = edge_velocities_and_area(p_corners[0],p_corners[1],dx,dy)
    # state.index_capa = 2 # aux[2,:,:] holds the capacity function

    claw = pyclaw.Controller()
    claw.tfinal = 1.0
    claw.solution = pyclaw.Solution(state,domain)
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
    # from clawpack.pyclaw.examples.advection_2d_annulus.mapc2p import mapc2p
    import numpy as np
    from clawpack.visclaw import colormaps

    plotdata.clearfigures()  # clear any old figures,axes,items data
    # plotdata.mapc2p = mapc2p
    # plotdata.plotdir = '_plots_wl'
    # plotdata.plotdir = '_plots_wl_theta'
    # plotdata.plotdir = '_plots_wl_k1pt5'
    # plotdata.plotdir = '_plots_wl_krktheta'
    # plotdata.plotdir = '_plots_nostream'
    # plotdata.plotdir = '_plots_blob_nostream'
    plotdata.plotdir = '_plots_1d_gauss'


    
    # # Figure for contour plot
    # plotfigure = plotdata.new_plotfigure(name='contour', figno=0)

    # # Set up for axes in this figure:
    # plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = 'auto'
    # plotaxes.ylimits = 'auto'
    # plotaxes.title = 'q[0]'
    # plotaxes.scaled = True

    # # Set up for item on these axes:
    # plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    # plotitem.plot_var = 0
    # plotitem.contour_levels = np.linspace(-0.9, 0.9, 10)
    # plotitem.contour_colors = 'k'
    # plotitem.patchedges_show = 1
    # plotitem.MappedGrid = True

    # Figure for pcolor plot
    # plotfigure = plotdata.new_plotfigure(name='q[0]', figno=1)
    plotfigure = plotdata.new_plotfigure(name='q[0]', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    # plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-.1, 1.1]
    plotaxes.title = 'q[0]'
    # plotaxes.scaled = True

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    # plotitem.plot_var = 1
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':2,'markersize':5}


    # cmap = cm.get_cmap('viridis')
    # plotitem.pcolor_cmap = cmap
    # plotitem.pcolor_cmin = 0.0
    # plotitem.pcolor_cmax = 1.0
    # plotitem.add_colorbar = True
    # plotitem.MappedGrid = True


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