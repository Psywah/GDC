from dolfin import *

set_log_level(1)

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
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# Mark all facets as sub domain 3
sub_domains.set_all(4)

# Mark no-slip facets as sub domain 0, 0.0
noflux = NoFlux()
noflux.mark(sub_domains, 0)

# Mark inflow as sub domain 1, 01
inflow = Inflow()
inflow.mark(sub_domains, 1)

# Mark outflow as sub domain 2, 0.2, True
outflow = Outflow()
outflow.mark(sub_domains, 2)

interface = Interface()
interface.mark(sub_domains, 3)

# Save sub domains to file
file = File("boundaries.xml")
file << sub_domains

# Save sub domains to VTK files
file = File("boundaries.pvd")
file << sub_domains


# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim())

# Mark all facets as sub domain 3
sub_domains.set_all(3)

# Mark no-slip facets as sub domain 0, 0.0
bacteria = Bacteria()
bacteria.mark(sub_domains, 1)
water = Water()
water.mark(sub_domains, 2)


# Save sub domains to file
file = File("subdomains.xml")
file << sub_domains

# Save sub domains to VTK files
file = File("subdomains.pvd")
file << sub_domains


