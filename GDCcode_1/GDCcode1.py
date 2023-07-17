from dolfin import *
import numpy as np

# Sub domain for noflux (mark whole boundary, inflow and outflow will overwrite)
class NoFlux(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Sub boundary for inflow (left)
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and x[1] <= 0.6+DOLFIN_EPS and x[1] >0.4-DOLFIN_EPS  and on_boundary

# Sub boundary for outflow (right)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and x[1] < 0.6+DOLFIN_EPS  and x[1] >0.4-DOLFIN_EPS  and on_boundary

# Sub boundary for interface
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
    
# Read mesh
nx = 50
ny = 50
mesh = UnitSquareMesh(nx, ny, diagonal='crossed')
    
# Create mesh functions over the cell facets
sub_bd = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# Mark no-slip facets as sub domain 0, 0.0
landedge = Bacteria()
landedge.mark(sub_bd,4)

wateredge = Water()
wateredge.mark(sub_bd,5)

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
sub_domains.set_all(8)

# Mark no-slip facets as sub domain 0, 0.0
bacteria = Bacteria()
bacteria.mark(sub_domains, 6)
water = Water()
water.mark(sub_domains, 7)

#Define measure
dx = Measure("dx", domain=mesh, subdomain_data=sub_domains)
ds = Measure("ds", domain=mesh, subdomain_data=sub_bd)
dS = Measure("dS", domain=mesh, subdomain_data=sub_bd)

#Define function space
V = FunctionSpace(mesh,'DG',0)

# Define test and trial functions
B = TrialFunction(V)
S = TrialFunction(V)
I = TrialFunction(V)
R = TrialFunction(V)
v = TestFunction(V)

#Create functions
B0 = Function(V)
S0 = Function(V)
I0 = Function(V)
R0 = Function(V)

#Define coeffiecients
dt = 0.01
k = Constant(dt)
T = 10

Total_population = 12347
K = Constant(100000)
N = Constant(Total_population)
deathrate = 7/30
delta = Constant(deathrate)
natural_deathrate = 7/(43.5*365)
Lamda = Constant(Total_population*natural_deathrate)
d = Constant(natural_deathrate)
c_g = Constant(10.0)
eta = Constant(1.0)
g_B = Constant(0.3333)
mortality = 0.02*7/(43.5*365)
m = Constant(mortality)
recovery_rate = 1.4
gamma = Constant(recovery_rate)
D_1 = Constant(0.1)
D_2 = Constant(0.01)
D_3 = Constant(0.1)
beta_H = Constant(0.00011)
beta_E = Constant(0.075)
class Diffusion_coefficient(UserExpression):
    def eval(self, value, x):
        if x[1] < 0.6 and x[1] > 0.4:
            value[0] = 1.0
        else:
            value[0] = 0.1
D_4 = interpolate(Diffusion_coefficient(),V)
# D_4 = Constant(0.0)
class Shedding_rate(UserExpression):
    def eval(self, value, x):
        if x[1] < 0.6 and x[1] > 0.4:
            value[0] = 0.0
        else:
            value[0] = 10.0
xi = interpolate(Shedding_rate(),V)

class Maximal_capacity(UserExpression):
    def eval(self, value, x):
        if x[1] < 0.6 and x[1] > 0.4:
            value[0] = 100000
        else:
            value[0] = 200000
K_B = interpolate(Maximal_capacity(),V)
B_0 = Constant(100000.0)
ZERO  = Constant(0.0)
B0.assign(ZERO)
S0.assign(N)
I0.assign(ZERO)
R0.assign(ZERO)
f_B = xi*I0

# Get velocity field
P2 = VectorElement('CG',mesh.ufl_cell(),2)
P1 = FiniteElement('CG',mesh.ufl_cell(),1)
W = FunctionSpace(mesh, MixedElement([P2,P1]))
# No-slip boundary condition for velocity 
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, sub_bd, 3)

# Inflow boundary condition for velocity
inflow = Expression(("-100*(x[1]-0.4)*(x[1]-0.6)", "0.0"), degree = 2)
bc1 = DirichletBC(W.sub(0), inflow, sub_bd, 1)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, sub_bd, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]
(u, p) = TrialFunctions(W)
(phi, q) = TestFunctions(W)
f = Constant((0, 0))
a = (inner(grad(u), grad(phi)) - div(phi)*p + q*div(u))*dx(7)
L = inner(f, phi)*dx(7)

A = assemble(a, keep_diagonal= True)
A.ident_zeros()
b = assemble(L)
[bc.apply(A, b) for bc in bcs]

# Compute solution
w = Function(W)
solve(A, w.vector(), b, 'lu')

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = w.split(True)


# # Split the mixed solution using a shallow copy
(u, p) = w.split()

# Define normal vector and mesh size
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2
un = (dot(u, n) + abs(dot(u, n)))/2.0

F_B = inner((1/k)*(B-B0)-g_B*(1-(B0/K_B))*B+delta*B,v)*dx - dot(B*u-D_4*grad(B),grad(v))*dx\
    + (c_g/h_avg)*dot(jump(D_4*B,n),jump(v,n))*dS(4) + (c_g/h_avg)*dot(jump(D_4*B,n),jump(v,n))*dS(5)\
    + c_g*(D_4/h)*dot(B*n,v*n)*ds(1)\
    + dot(jump(v), un('+')*B('+') - un('-')*B('-'))*dS(5)\
    - dot(avg(D_4*grad(B)),jump(v,n))*dS(4) - dot(avg(D_4*grad(B)),jump(v,n))*dS(5) - dot(D_4*grad(B),v*n)*ds(1)\
    + eta*dot(jump(B,n),jump(v,n))*dS(3) + dot(u,n)*B*v*ds(2)\
    - f_B*v*dx - c_g*(D_4/h)*B_0*v*ds(1) + dot(u,n)*B_0*v*ds(1)
    #velocity does not work it seems

B1 = Function(V, name='Bacteria')
a_B = lhs(F_B)
L_B = rhs(F_B)
b_B = assemble(L_B)
A_B = assemble(a_B)
solve(A_B, B1.vector(), b_B, 'lu')
# solve(a_B == L_B, p)
B0.assign(B1)
F_S = inner((1/k)*(S-S0),v)*dx(6) - Lamda*v*dx(6) + beta_H*S0*I0*v*dx(6) + beta_E*S*B0/(B0+K)*v*dx(6)\
    + d*S*v*dx(6) + dot(D_1*grad(S),grad(v))*dx(6) + c_g*(D_1/h_avg)*dot(jump(S,n),jump(v,n))*dS(4)\
    + dot(avg(D_1*grad(S)),jump(v,n))*dS(4) + S*v*dx(7)
S1 = Function(V, name='Susceptible')
a_S = lhs(F_S)
L_S = rhs(F_S)
b_S = assemble(L_S)
A_S = assemble(a_S)
solve(A_S, S1.vector(), b_S, 'lu')



F_I = inner((1/k)*(I-I0),v)*dx(6)  - beta_H*S0*I0*v*dx(6) - beta_E*S1*B0/(B0+K)*v*dx(6)\
    + (d+m+gamma)*I*v*dx(6) + dot(D_2*grad(I),grad(v))*dx(6) + c_g*(D_2/h_avg)*dot(jump(I,n),jump(v,n))*dS(4)\
    + dot(avg(D_2*grad(I)),jump(v,n))*dS(4) + I*v*dx(7)
I1 = Function(V, name='Infectious')
a_I = lhs(F_I)
L_I = rhs(F_I)
b_I = assemble(L_I)
A_I = assemble(a_I)
solve(A_I, I1.vector(), b_I, 'lu')



F_R = inner((1/k)*(R-R0),v)*dx(6) + d*R*v*dx(6) \
    - gamma*I1*v*dx(6) + dot(D_3*grad(R),grad(v))*dx(6) + c_g*(D_3/h_avg)*dot(jump(R,n),jump(v,n))*dS(4)\
    + dot(avg(D_3*grad(R)),jump(v,n))*dS(4) + R*v*dx(7)

R1 = Function(V, name='Recovered')
a_R = lhs(F_R)
L_R = rhs(F_R)
b_R = assemble(L_R)
A_R = assemble(a_R)
solve(A_R, R1.vector(), b_R, 'lu')


S0.assign(S1)
I0.assign(I1)
R0.assign(R1)

p_file = XDMFFile('./results/bacteria.xdmf')

t=dt
p_file.write(B1, t)
# p_file.close()
s_file = XDMFFile('./results/susceptible.xdmf')

s_file.write(S1, t)
# s_file.close()

i_file = XDMFFile('./results/infectious.xdmf')

i_file.write(I1, t)
# i_file.close()

r_file = XDMFFile('./results/recovered.xdmf')

r_file.write(R1, t)
# r_file.close()

while t < T + DOLFIN_EPS:
    #cI.t = t
    t += dt
    info("time stepping t/T = {}/{}".format(t, T))
    # Compute tentative velocity step
    a_B = lhs(F_B)
    L_B = rhs(F_B)
    b_B = assemble(L_B)
    A_B = assemble(a_B)
    solve(A_B, B1.vector(), b_B, 'lu')
    B0.assign(B1)   

    a_S = lhs(F_S)
    L_S = rhs(F_S)
    b_S = assemble(L_S)
    A_S = assemble(a_S)
    solve(A_S, S1.vector(), b_S, 'lu')
    a_I = lhs(F_I)
    L_I = rhs(F_I)
    b_I = assemble(L_I)
    A_I = assemble(a_I)
    solve(A_I, I1.vector(), b_I, 'lu')
    a_R = lhs(F_R)
    L_R = rhs(F_R)
    b_R = assemble(L_R)
    A_R = assemble(a_R)
    solve(A_R, R1.vector(), b_R, 'lu')
    S0.assign(S1)
    I0.assign(I1)
    R0.assign(R1)
    # Save to file
    p_file.write(B1, t)

    s_file.write(S1, t)

    i_file.write(I1, t)
    
    r_file.write(R1, t)
    

    # Move to next time step
i_file.close()
r_file.close()
s_file.close()
p_file.close()
    