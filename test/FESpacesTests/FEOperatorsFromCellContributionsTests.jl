module FEOperatorsFromCellContributionsTests

using Test
using Gridap.Helpers
using Gridap.Arrays
using Gridap.Algebra
using Gridap.TensorValues
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Integration
using Gridap.Fields
using Gridap.FESpaces
using Gridap.CellData

u(x) = x[1] + x[2]
f(x) = -Δ(u)(x)

domain = (0,1,0,1)
partition = (4,4)
model = CartesianDiscreteModel(domain,partition)

order = 2
V = TestFESpace(model=model,reffe=:Lagrangian,order=order,valuetype=Float64,conformity=:H1)
U = TrialFESpace(V)

trian_Ω = Triangulation(model)
trian_Γ = BoundaryTriangulation(model)
n_Γ = get_normal_vector(trian_Γ)

degree = 2*order
dΩ = LebesgueMeasure(trian_Ω,degree)
dΓ = LebesgueMeasure(trian_Γ,degree)

γ = 10
h = 1/4

# @form only needed if you want to write a(U,V) == l(V)
@form a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ + ∫( (γ/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )*dΓ
@form l(v) = ∫( v*f )*dΩ + ∫( (γ/h)*v*u - (n_Γ⋅∇(v))*u  )*dΓ

op = AffineFEOperator(U,V,a,l)
uh = solve(op)
e = u - uh

el2 = sqrt( sum( ∫( e*e )*dΩ ) )
eh1 = sqrt( sum( ∫( ∇(e)⋅∇(e) + e*e )*dΩ ) )

@test el2 < 1.e-8
@test eh1 < 1.e-7

# Idem with syntactic sugar

uh = solve( a(U,V) == l(V) )
# previous line idem to:
# uh = solve( operate(a,U,V) == operate(l,V) )
# thanks to the @form macro
# which in turn it is equivalent to
# op = AffineFEOperator(U,V,a,l)
e = u - uh

el2 = sqrt( sum( ∫( e*e )*dΩ ) )
eh1 = sqrt( sum( ∫( ∇(e)⋅∇(e) + e*e )*dΩ ) )

@test el2 < 1.e-8
@test eh1 < 1.e-7

end # mdoule
