
module SurfaceCouplingTests

using Test
using Gridap
import Gridap: ∇
using LinearAlgebra: tr, ⋅
using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.Geometry: get_cell_coordinates
using Gridap.Geometry: get_node_coordinates
using Gridap.FESpaces: get_algebraic_operator
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using FillArrays

# Analytical functions

u(x) = x[1]
p(x) = x[1] 

s(x) = -Δ(u)(x)
f(x) = u(x) - p(x)

# Geometry + Integration

n = 10
mesh = (n,n)
D = length(mesh)
domain = 2 .* (0,1,0,1) .- 1
order = 1
model = CartesianDiscreteModel(domain, mesh)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,5])
add_tag_from_tags!(labels,"neumann",[6,7,8])

Ω = Triangulation(model)

const R = 0.4

function is_in(coords)
  n = length(coords)
  x = (1/n)*sum(coords)
  d = x[1]^2 + x[2]^2 - R^2
  d < 0
end

cell_to_coods = get_cell_coordinates(Ω)
cell_to_is_solid = lazy_map(is_in,cell_to_coods)
cell_to_is_fluid = lazy_map(!,cell_to_is_solid)

model_solid = DiscreteModel(model,cell_to_is_solid)
model_fluid = DiscreteModel(model,cell_to_is_fluid)

# #=
model_lowerdim = DiscreteModel(Polytope{D-1},model)
bmodel = DiscreteModel(model_lowerdim,tags="boundary")


Ωs = Triangulation(model_solid)
Ωf = Triangulation(model_fluid)
ΩΓ = Triangulation(model_lowerdim)

#n_Λ = get_normal_vector(Λ)
#n_Γ = get_normal_vector(Γ)

order = 1
degree = 2*order

dΩ = Measure(Ω,degree)
dΩs = Measure(Ωs,degree)
dΩf = Measure(Ωf,degree)
#dΛ = Measure(Λ,degree)
#dΓ = Measure(Γ,degree)
dΩΓ = Measure(ΩΓ,degree)

# FE Spaces

reffe_u = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_u,conformity=:H1,labels=labels,dirichlet_tags="boundary")

reffe_p = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(model_fluid,reffe_p,conformity=:L2)

 #=
reffeg  = ReferenceFE(lagrangian,Float64,order)
Q = FESpace(
  model_lowerdim,
  reffeg,
  conformity=:H1,
  constraint=:zeromean
  )
 =#

U = TrialFESpace(V,u)
P = TrialFESpace(Q)


#QΓ =  CellQuadrature(Γ,degree)
#@show num_free_dofs(P)
#@show length(get_node_coordinates(Γi))

get_cell_shapefuns_trial(Q)
get_cell_shapefuns(Q)


#get_data(QΓ)

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

#uh, ph = FEFunction(X,rand(num_free_dofs(X)))
#vh, qh = FEFunction(Y,rand(num_free_dofs(Y)))
#writevtk(Ω,"trian",cellfields=["uh"=>uh,"ph"=>ph,"vh"=>vh,"qh"=>qh])

# Weak form
a((u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) )*dΩ +
  ∫( q*u -q*p )*dΩf 

l((v,q)) =
  ∫( v⋅s )*dΩ +
  ∫(  q⋅f )*dΩf 


# FE problem

op = AffineFEOperator(a,l,X,Y)
uh, ph = solve(op)

# Visualization

eu = u - uh
ep = p - ph

#writevtk(Ω,"trian",cellfields=["uh"=>uh,"ph"=>ph,"eu"=>eu,"ep"=>ep])
#writevtk(Ωf,"trian_fluid",cellfields=["uh"=>uh,"ph"=>ph,"eu"=>eu,"ep"=>ep])
 # Errors 
 eu_l2 = sqrt(sum(∫( eu⋅eu )*dΩ)) 
 eu_h1 = sqrt(sum(∫( eu⋅eu + ∇(eu)⊙∇(eu) )*dΩ)) 
 ep_l2 = sqrt(sum(∫( ep*ep )*dΩf)) 
 tol = 1.0e-9
@test eu_l2 < tol
@test eu_h1 < tol
@test ep_l2 < tol

end # module
