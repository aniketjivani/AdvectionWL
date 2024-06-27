Use Clawpack (https://www.clawpack.org) - routines for FV methods for hyperbolic systems - to implement 2D advection in cylindrical coordinates with given initial condition for C3 coronagraph white light images.

Repo contains tutorials from Clawpack and preprocessing scripts for appropriate data conversion of the white light simulation outputs.

Need to basically implement boundary conditions correctly frm this: https://www.clawpack.org/gallery/_static/amrclaw/examples/advection_2d_inflow/_plots/allframes_fig0.html

and this: https://www.clawpack.org/gallery/_static/amrclaw/examples/advection_2d_inflow/README.html - modify to remove the AMR stuff. One problematic part is that this example still has constant velocity. However, the inflow condition can be potentially reused.

Take help from this: https://www.clawpack.org/gallery/_static/amrclaw/examples/advection_2d_annulus/README.html


```fortran
!snippets from bc1
c     # Standard boundary condition choices for claw2
c
c     # At each boundary  k = 1 (left),  2 (right):
c     #   mthbc(k) =  0  for user-supplied BC's (must be inserted!)
c     #            =  1  for zero-order extrapolation
c     #            =  2  for periodic boundary coniditions
c     #            =  3  for solid walls, assuming this can be implemented
c     #                  by reflecting the data about the boundary and then
c     #                  negating the 2'nd component of q.
c     ------------------------------------------------
c
c     # Extend the data from the computational region
c     #      i = 1, 2, ..., mx2
c     # to the virtual cells outside the region, with
c     #      i = 1-ibc  and   i = mx+ibc   for ibc=1,...,mbc
c
      implicit double precision (a-h,o-z)
      dimension q(meqn,1-mbc:mx+mbc)
      dimension aux(maux,1-mbc:mx+mbc)

      dimension mthbc(2)

      common /cparam/ rho,bulk,cc,zz
      common /combc/ omega
   

c
c
c-------------------------------------------------------
c     # left boundary:
c-------------------------------------------------------
      go to (100,110,120,130) mthbc(1)+1
c
  100 continue
c     # incoming sine wave

c     # strength of 1-wave (extrapolate the outgoing wave):
      w1 = (-q(1,1) + zz*q(2,1)) / (2.d0*zz)

c     # strength of 2-wave (specify the incoming wave):
      if (omega*t .le. 8.d0*datan(1.d0)) then
           w2 = 0.5d0 * dsin(omega*t) 
	else
	   w2 = 0.d0
	endif

      do 105 ibc=1,mbc
         q(1,1-ibc) = -w1*zz + w2*zz
         q(2,1-ibc) = w1 + w2
  105    continue
      go to 199

```

```fortran
!Snippets from amrclaw 2d inflow:

!  ____________________________________________________
! 
!                _____________________ (xupper,yupper)
!               |                     |  
!           ____|____ (xhi_patch,yhi_patch)   
!           |   |    |                |
!           |   |    |                |
!           |   |    |                |
!           |___|____|                |
!  (xlo_patch,ylo_patch) |            |
!               |                     |
!               |_____________________|   
!    (xlower,ylower)
!  ____________________________________________________

!! \param val data array for solution \f$q \f$ (cover the whole grid **msrc**)
!! \param aux data array for auxiliary variables 
!! \param nrow number of cells in *i* direction on this grid
!! \param ncol number of cells in *j* direction on this grid
!! \param meqn number of equations for the system
!! \param naux number of auxiliary variables
!! \param hx spacing (mesh size) in *i* direction
!! \param hy spacing (mesh size) in *j* direction
!! \param level AMR level of this grid
!! \param time setting ghost cell values at time **time**
!! \param xlo_patch left bound of the input grid
!! \param xhi_patch right bound of the input grid 
!! \param ylo_patch lower bound of the input grid 
!! \param yhi_patch upper bound of the input grid 

!-------------------------------------------------------
! Left boundary:
!-------------------------------------------------------
if (xlo_patch < xlower-hxmarg) then
    ! number of grid cells from this patch lying outside physical domain:
    nxl = int((xlower + hxmarg - xlo_patch) / hx)

    select case(mthbc(1))
        case(0) ! User defined boundary condition
            ! Inflow boundary condition
            if (ubar < 0.1d0 * vbar) then
                stop "Inflow BCs at left boundary assume ubar >= vbar / 10"
            end if

            do j = 1, ncol
                y = ylo_patch + (j - 0.5d0) * hy
                if (nxl >= 1) then
                    ! First ghost cell
                    tau = hx / (2.d0 * ubar)
                    val(1,nxl,j) = qtrue(0.d0, y + vbar * tau, time + tau)
                end if
                if (nxl == 2) then
                    ! second ghost cell:
                    tau = 3.d0 * hx / (2.d0 * ubar)
                    val(1,1,j) = qtrue(0.d0, y + vbar * tau, time + tau)
                end if
            end do

!-------------------------------------------------------
! Bottom boundary:
!-------------------------------------------------------
if (ylo_patch < ylower - hymarg) then

    ! number of grid cells lying outside physical domain:
    nyb = int((ylower + hymarg - ylo_patch) / hy)

    select case(mthbc(3))
        case(0) ! User defined boundary condition
            ! Inflow boundary condition
            if (vbar < 0.1d0 * ubar) then
                stop "Inflow BCs at bottom boundary assume vbar >= ubar / 10"
            end if

            do i = 1, nrow
                x = xlo_patch + (i - 0.5d0) * hx
                if (nyb >= 1) then
                    ! First ghost cell
                    tau = hy / (2.d0 * vbar)
                    val(1,i,nyb) = qtrue(x + vbar * tau, 0.d0, time + tau)
                end if
                if (nyb == 2) then
                    ! second ghost cell:
                    tau = 3.d0 * hy / (2.d0 * vbar)
                    val(1,i,1) = qtrue(x + ubar * tau, 0.d0, time + tau)
                end if
            end do



```

By default, for all our problems:
`clawSolution.output_style=1`
Evenly spaced output, between the initial and final simulation times. The number of outputs is the value of claw.num_output_times.

From https://www.clawpack.org/pyclaw/problem.html:

```
Setting auxiliary variables
If the problem involves coefficients that vary in space or a mapped grid, the required fields are stored in state.aux. In order to use such fields, you must pass the num_aux argument to the State initialization

state = pyclaw.State(domain,solver.num_eqn,num_aux)
The number of fields in state.aux (i.e., the length of its first dimension) is set equal to num_aux. The values of state.aux are set in the same way as those of state.q.
```


```
Setting boundary conditions
The boundary conditions are specified through solver.bc_lower and solver.bc_upper, each of which is a list of length solver.num_dim. The ordering of the boundary conditions in each list is the same as the ordering of the Dimensions in the Grid; typically (x,y)
. Thus solver.bc_lower[0] specifies the boundary condition at the left boundary and solver.bc_upper[0] specifies the condition at the right boundary. Similarly, solver.bc_lower[1] and solver.bc_upper[1] specify the boundary conditions at the top and bottom of the domain.

If the state.aux array is used, boundary conditions must be set for it in a similar way, using solver.aux_bc_lower and solver.aux_bc_upper. Note that although state is passed to the BC routines, they should NEVER modify state. Rather, they should modify qbc/auxbc.


```

writing patch to aux file:

```
print(clawSolution.solution.state.patch)
dims = clawSolution.solution.state.patch.dimensions
dims[0] # 124 (r)
dims[1] # 512 (theta)
# q is state.aux of shape (3, 124, 512)
if patch.num_dim == 2:
    for j in range(dims[1].num_cells):
        for k in range(dims[0].num_cells):
            for m in range(q.shape[0]):
                f.write("%18.8e" % q[m,k,j])
            f.write('\n')    
        f.write('\n')

# loop first over r for a single theta, then over theta??
```
whitelight7 - both components positive
whitelight9 - both components negative
whitelight8, 10 - one positive one negative
whitelight11 - increase u0 to 60
whitelight12 - bring down u0 and change defn to np.abs(X) or np.abs(Y) when figuring out the velocity field.
whitelight13 - same discontinuous experiment as annular_var11

Interpolations saved to file, looks like nothing much going wrong with the indexing there. Perhaps its a problem with the velocity field.

annular_var6 and 7 - using two blobs with signs for u and v flipped. We see that the blobs don't traverse radially outward but in one direction only!!!!!
annular_var8 - change blob orientation so each is at 60 degrees to the horizontal / vertical. (i.e. theta is no longer 0!) - this is to check that the velocity field we gave is at the very least not horizontal and has correct radial outflow at least on one side.
annular_var9 - use u0 * np.abs(X) / .... and so on.
annular_var10 - use proper conditions i.e. extrap for radial direction instead of periodic. and use theta at an angle.
annular_var11 - discontinuous defn for radial flow. check if this ensures correct sign is followed.

okay, so np.abs(X) is not perfectly correct either. We need to use 