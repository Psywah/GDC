from dolfin import *

# Sub domain for no-slip (mark whole boundary, inflow and outflow will overwrite)
class NoFlux(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Sub domain for inflow (right)
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and x[1] <= 0.6+DOLFIN_EPS and x[1] >0.4-DOLFIN_EPS  and on_boundary

# Sub domain for outflow (left)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and x[1] < 0.6+DOLFIN_EPS  and x[1] >0.4-DOLFIN_EPS  and on_boundary

# Sub domain for interface
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.6) or  near(x[1], 0.4)

# Sub domain for interface
class Bacteria(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]> 0.6 - DOLFIN_EPS  or  x[1]< 0.4 + DOLFIN_EPS 

# Sub domain for interface
class Water(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]< 0.6 + DOLFIN_EPS  and   x[1]>  0.4 - DOLFIN_EPS 


nx = 50
ny = 50
# Read mesh
mesh = UnitSquareMesh(nx, ny, diagonal='right')

# Create mesh functions over the cell facets
sub_bd = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# Mark all facets as sub domain 3
sub_bd.set_all(4)

# Mark no-slip facets as sub domain 0, 0.0
noflux = NoFlux()
noflux.mark(sub_bd, 0)

# Mark inflow as sub domain 1, 01
inflow = Inflow()
inflow.mark(sub_bd, 1)

# Mark outflow as sub domain 2, 0.2, True
outflow = Outflow()
outflow.mark(sub_bd, 2)

interface = Interface()
interface.mark(sub_bd, 3)


# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim())

# Mark all facets as sub domain 3
sub_domains.set_all(3)

# Mark no-slip facets as sub domain 0, 0.0
bacteria = Bacteria()
bacteria.mark(sub_domains, 1)
water = Water()
water.mark(sub_domains, 2)

# Define function spaces
P2 = VectorElement('CG',mesh.ufl_cell(),2)
P1 = FiniteElement('CG',mesh.ufl_cell(),1)
W = FunctionSpace(mesh, MixedElement([P2,P1]))

# No-slip boundary condition for velocity 
# x1 = 0, x1 = 1 and around the dolphin
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_bd, 3)

# Inflow boundary condition for velocity
# x0 = 1
inflow = Expression(("-100*(x[1]-0.4)*(x[1]-0.6)", "0.0"), degree = 2)
bc1 = DirichletBC(W.sub(0), inflow, sub_bd, 1)

# Boundary condition for pressure at outflow
# x0 = 0
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, sub_bd, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
dxx = dx(subdomain_data = sub_domains)
a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dxx(2)
L = inner(f, v)*dxx(2)

A = assemble(a, keep_diagonal= True)
A.ident_zeros()
b = assemble(L)
[bc.apply(A, b) for bc in bcs]

# Compute solution
w = Function(W)
solve(A, w.vector(), b, 'lu')

#solve(a == L, w, bcs)

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)

#print "Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2")
#print "Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2")

# # Split the mixed solution using a shallow copy
(u, p) = w.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
