module AutodiffSubDomainTests

#using Pkg
#Pkg.activate("/home/user/Documents/GitHub/TestRepo/envSD")

using Test
using Gridap
import Gridap: ∇
using LinearAlgebra: tr, ⋅
using Gridap
import Gridap: ∇
using Test
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.MultiField
using Gridap.CellData
using Gridap.Fields
using Gridap.Arrays
using Gridap.MultiField
using Gridap.Geometry

# Background model
n = 5
mesh = (n,n)
domain = (0,1,0,1) .- 1
order = 1
model = CartesianDiscreteModel(domain, mesh)
Ω = Triangulation(model)

# Extract a portion of the background mesh
R = 0.7
function is_in(coords)
  n = length(coords)
  x = (1/n)*sum(coords)
  d = x[1]^2 + x[2]^2 - R^2
  d < 0
end

cell_to_coords = get_cell_coordinates(Ω)
cell_to_is_solid = lazy_map(is_in,cell_to_coords)
cell_to_is_fluid = lazy_map(!,cell_to_is_solid)

Ωs = Triangulation(model,cell_to_is_solid)
Ωf = Triangulation(model,cell_to_is_fluid)

degree=2
dΩ = Measure(Ω,degree)
dΩs = Measure(Ωs,degree)
dΩf = Measure(Ωf,degree)

# FE Spaces
reffe_u = ReferenceFE(lagrangian,Float64,order)

V1 = TestFESpace(Ωs,reffe_u,conformity=:H1)
V2 = TestFESpace(Ωf,reffe_u,conformity=:H1)

V = MultiFieldFESpace([V2,V1])
uh = FEFunction(V,ones(num_free_dofs(V)))

j((u1,u2)) = ∫(u1)dΩs + ∫(u2)dΩf

js2 = ∇(j)(uh)
assemble_vector(js2,V)

end
