from dolfin import *
import numpy as np

# FIXME: Make mesh ghosted
parameters["ghost_mode"] = "shared_facet"
import argparse

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False

parser = argparse.ArgumentParser()


parser.add_argument('-l', dest='level', help='Number of refinements', default=0, type=int)
parser.add_argument('-sigma',dest='sigma', help='penalty parameter for IP', default=4.0, type=float)
parser.add_argument('-theta_f',dest='theta_f', help='parameter for symmetry', default=0.0, type=float)
parser.add_argument('-theta_c',dest='theta_c', help='parameter for symmetry', default=0.0, type=float)
parser.add_argument('-prefix_dir',dest='prefix_dir', help='folder to save files', default='results/', type=str)
parser.add_argument('-dt',dest='dt', help='time step size', default=0.01, type=float)
parser.add_argument('-T',dest='T', help='end time', default=1.2, type=float)
args = parser.parse_args()



# Define class marking Dirichlet boundary (x = 0 or x = 1)
class InflowBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[0], 0)

class OutflowBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(1 - x[0], 0)


# Define class marking Neumann boundary (y = 0 or y = 1)
class NeumanBoundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary and near(x[1]*(1 - x[1]), 0)

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
for i in range(args.level):
   info("refining mesh......")
   mesh = refine(mesh)
   info("number of vertices in fluid mesh: {}".format(
     MPI.sum(MPI.comm_world, mesh.num_vertices())))

V = FunctionSpace(mesh, 'DG', 1)

# Define test and trial functions
p = TrialFunction(V)
w = TestFunction(V)

# Define normal vector and mesh size
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

# Define the source term f, Dirichlet term u0 and Neumann term g
"""
# test convergence
f = Expression('2*cos(x[0]-x[1])', degree=2)
u1 = Expression('cos(x[0]-x[1])', degree=4)
u2 = Expression('cos(x[0]-x[1])', degree=4)
g = Expression('sin(x[0]-x[1])*(x[1]-.5)*2', degree=2)
kappa0 = Constant(1.0)
"""
f = Constant(0.0)
p1 = Constant(1.0)
p2 = Constant(0.0)
g = Constant(0.0)
kappa0 = Expression('((x[0]>=3./8) && (x[0]<=5./8) && (x[1]>=1./4) && (x[1]<=3./4))? 1.e-3 : 1.', degree=2)
DG0 = FunctionSpace(mesh, 'DG', 0)
kappa = Function(DG0, name='kappa')
kappa.interpolate(kappa0)
kappa.vector().set_local(np.random.rand(kappa.vector().local_size())+0.001)


# Mark facets of the mesh
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
InflowBoundary().mark(boundaries, 1)
OutflowBoundary().mark(boundaries,2)
NeumanBoundary().mark(boundaries, 3)

# Define outer surface measure aware of Dirichlet and Neumann boundaries
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define parameters
sigma = Constant(args.sigma)
theta_f = Constant(args.theta_f)

# Define variational problem
a = dot(grad(w), kappa*grad(p))*dx \
   - dot(jump(w, n), avg(kappa*grad(p)))*dS \
   - dot(w*n, kappa*grad(p))*ds(1) \
   - dot(w*n, kappa*grad(p))*ds(2) \
   + theta_f*dot(avg(kappa*grad(w)), jump(p, n))*dS \
   + theta_f*dot(kappa*grad(w), p*n)*ds(1) \
   + theta_f*dot(kappa*grad(w), p*n)*ds(2) \
   + sigma/h_avg*dot(jump(w, n), jump(p, n))*dS \
   + (sigma/h)*w*p*ds(1)\
   + (sigma/h)*w*p*ds(2)
L = w*f*dx - g*w*ds(3) + theta_f*p1*dot(kappa*grad(w), n)*ds(1) + theta_f*p2*dot(kappa*grad(w), n)*ds(2)+ (sigma/h)*p1*w*ds(1) + (sigma/h)*p2*w*ds(2) 

# Compute solution
p = Function(V, name="pressure")
solve(a == L, p)


p_file = XDMFFile(args.prefix_dir + '/pressure.xdmf')
p_file.parameters["flush_output"] = True
p_file.parameters["functions_share_mesh"] = True
p_file.parameters["rewrite_function_mesh"] = False
p_file.write(p,0.0)
p_file.write(kappa,0.0)

#print("Solution vector norm (0): {!r}".format(u.vector().norm("L2")))
print("Solution L2 norm : {!r}".format(norm(p,"L2")))
#print("Solution error norm (L2): {!r}".format(errornorm(u1,u,"L2")))
#print("Solution error norm (H1): {!r}".format(errornorm(u1,u,"H1")))
#
def u_T(p):
    return -kappa*grad(p)
def u_o(p):
    return -avg(kappa*grad(p)) + sigma/h_avg*jump(p,n)
def u_ndotn(p):
    return g
def u_ddotn1(p):
    return -kappa*dot(grad(p),n) + sigma/h*(p-p1)
def u_ddotn2(p):
    return -kappa*dot(grad(p),n) + sigma/h*(p-p2)

# compute Rloc 
DG0 = FunctionSpace(mesh, 'DG', 0)
rl = TrialFunction(DG0)
tl = TestFunction(DG0)
F = rl*tl*dx -  inner(u_o(p),jump(tl,n))*dS - u_ddotn1(p)*tl*ds(1) - u_ddotn2(p)*tl*ds(2) - u_ndotn(p)*tl*ds(3)
rl = Function(DG0, name='Rloc')
a = lhs(F)
L = rhs(F)
A = assemble(a)
b = assemble(L)
solve(A, rl.vector(), b, 'mumps')
p_file.write(rl,0.0)

"""
CG1 = FunctionSpace(mesh, 'CG', 2)
p = TrialFunction(CG1)
q = TestFunction(CG1)
f = Constant(0.0)
F = inner(kappa*grad(p), grad(q))*dx -f*q*dx
a = lhs(F)
L = rhs(F)
bc1 = DirichletBC(CG1, p1, boundaries, 1)
bc2 = DirichletBC(CG1, p2, boundaries, 2)
p = Function(CG1,name = "pressure")
solve(a == L, p, [bc1, bc2])
p_file.write(p,1.0)

DG0 = FunctionSpace(mesh, 'DG', 0)
rl = TrialFunction(DG0)
tl = TestFunction(DG0)
F = rl*tl*dx -  div(kappa*grad(p))*tl*dx
a = lhs(F)
L = rhs(F)
A = assemble(a)
b = assemble(L)
rl = Function(DG0, name='Rloc')
solve(A, rl.vector(), b, 'mumps')
p_file.write(rl,1.0)

exit()
"""

VC = FunctionSpace(mesh, 'DG', 0)
# Create functions
c0 = Function(VC)
c1 = Function(VC, name="consentration")
c = TrialFunction(VC)
w = TestFunction(VC)

dt = args.dt
# Define coefficients
k = Constant(dt)
D = Constant(0.0)
#cI = Expression('cos(2*3.141592653*x[1])*sin(2*3.141592653*t)',t=0, degree=2)
cI = Constant(1.0)
theta_c = Constant(args.theta_c)
beta = Constant((1.0,0.0))

F = (1/k)*inner(c - c0, w)*dx - inner(u_T(p)*c - D*grad(c), grad(w))*dx\
        + inner((u_o(p)*avg(c) \
                  + abs(inner(u_o(p), n('-')))/2 * jump(c,n)\
                  - avg(D*grad(c))\
                ), jump(w,n)\
               )*dS\
        +theta_c * inner(avg(D*grad(w)), jump(c,n))*dS\
        + cI* u_ddotn1(p)* w*ds(1)\
        + c* u_ddotn2(p)* w*ds(2)\
        + c* u_ndotn(p)* w*ds(3)\
        #+ sigma/h_avg*inner(jump(c,n),jump(w,n))*dS
"""
F = (1/k)*inner(c - c0, w)*dx - inner(beta*c - D*grad(c), grad(w))*dx\
        + inner( avg(beta*c), jump(w,n) ) *dS\
        + inner( beta*c, w*n ) *ds(2)\
        + inner( beta*c, w*n ) *ds(3)\
        + inner( beta*cI, w*n ) *ds(1)\
        + Constant(1.0)*abs(inner(beta, n('-')))/2 * inner( jump(c,n) , jump(w,n))*dS
"""

a = lhs(F)
L = rhs(F)

# Assemble matrices
A = assemble(a)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
c_file = XDMFFile(args.prefix_dir + '/concentration.xdmf')

# Time-stepping
t = dt
T = args.T
while t < T + DOLFIN_EPS:
    #cI.t = t

    info("time stepping t/T = {}/{}".format(t, T))
    # Compute tentative velocity step
    b = assemble(L)
    #[bc.apply(A1, b1) for bc in bcu]
    solve(A, c1.vector(), b, "mumps")

    # Save to file
    c_file.write(c1, t)

    # Move to next time step
    c0.assign(c1)
    t += dt


