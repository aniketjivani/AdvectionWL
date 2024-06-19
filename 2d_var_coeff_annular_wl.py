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

    state.q[0,:,:] = sample_rd




def auxinit(state):
    # Initialize petsc Structures for aux
    # xc=state.grid.x.centers
    X, Y = state.grid.p_centers

    rnodes, theta_nodes = state.grid.p_nodes
    dx, dy = state.grid.delta
    mx = rnodes.shape[0] - 1
    my = rnodes.shape[1] - 1

    u0 = 15 # imagine radial velocity is fixed and doesn't change with r for case 1.

    # state.aux[0, :, :] = 0.0
    # state.aux[1, :, :] = np.sin(2.*np.pi*X)+2
    # state.aux[1, :, :] = (0.5 - 0.5 * Y) + 0.1

    # Bottom-left corners
    Xp0 = rnodes[:mx,:my]
    Yp0 = theta_nodes[:mx,:my]

    # Top-left corners
    Xp1 = rnodes[:mx,1:]
    Yp1 = theta_nodes[:mx,1:]

    # Top-right corners
    Xp2 = rnodes[1:,1:]
    Yp2 = theta_nodes[1:,1:]

    # Top-left corners
    Xp3 = rnodes[1:,:my]
    Yp3 = theta_nodes[1:,:my]

    # Compute area of the physical element
    area = 1./2.*( (Yp0+Yp1)*(Xp1-Xp0) +
                   (Yp1+Yp2)*(Xp2-Xp1) +
                   (Yp2+Yp3)*(Xp3-Xp2) +
                   (Yp3+Yp0)*(Xp0-Xp3) )
    
    state.aux[0, :, :] = -u0 * X / np.sqrt(X**2 + Y**2)
    state.aux[1, :, :] = -u0 * Y / np.sqrt(X**2 + Y**2)
    state.aux[2, :, :] = area / (dx * dy)




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

    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap

    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    solver.aux_bc_lower[1] = pyclaw.BC.periodic
    solver.aux_bc_upper[1] = pyclaw.BC.periodic

    # solver.bc_lower[1] = pyclaw.BC.periodic
    # solver.bc_upper[1] = pyclaw.BC.periodic

    # solver.bc_lower[1] = pyclaw.BC.extrap
    # solver.bc_upper[1] = pyclaw.BC.extrap

    # 2d var velocity
    solver.dt_initial = 0.1

    # solver.cfl_max = 1.0
    # solver.cfl_desired = 0.8 # how to set this?

    solver.cfl_max = 0.5
    solver.cfl_desired = 0.4

    # r_lower = 0.2
    # r_upper = 1.0
    # m_r = 40

    # theta_lower = 0.0
    # theta_upper = np.pi*2.0
    # m_theta = 120

    r_lower = 3.903302085636277
    r_upper = 23.465031329617336
    m_r = 126

    theta_lower = 0.0
    theta_upper = np.pi*2.0
    m_theta = 512

    r = pyclaw.Dimension(r_lower, r_upper, m_r, name='x')
    theta = pyclaw.Dimension(theta_lower, theta_upper, m_theta, name='y')
    domain = pyclaw.Domain([r, theta])
    domain.grid.mapc2p = mapc2p_annulus
    domain.grid.num_ghost = solver.num_ghost

    # xlower=0.0; xupper=1.0; mx=100
    # x    = pyclaw.Dimension(xlower,xupper,mx,name='x')
    # domain = pyclaw.Domain(x)

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
    num_aux = 3

    state = pyclaw.State(domain, num_eqn, num_aux)

    qinit(state)
    auxinit(state)

    state.index_capa = 2

    # num_aux = 2
    # state = pyclaw.State(domain, num_eqn, num_aux)

    # state.problem_data['u'] = 0.0
    # state.problem_data['v'] = 1.0
    # state.problem_data['v'] = 0.5

    claw = pyclaw.Controller()
    claw.tfinal = 2.0
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
    from clawpack.pyclaw.examples.advection_2d_annulus.mapc2p import mapc2p
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
    # plotdata.plotdir = '_plots_2d_var_cartesian2'
    # plotdata.plotdir = '_plots_2d_var_cartesian3'
    # plotdata.plotdir = '_plots_2d_var_annular1'
    # plotdata.plotdir = '_plots_2d_var_annular2'
    plotdata.plotdir = '_plots_2d_var_whitelight2'



    # plotdata.mapc2p = mapc2p_annulus
    plotdata.mapc2p = mapc2p 

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