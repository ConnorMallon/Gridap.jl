
module SUPGSteadyLDC

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
#using GridapODEs.ODETools
#using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
#using GridapEmbedded
import Gridap: ∇
#import GridapODEs.TransientFETools: ∂t
using LineSearches: BackTracking
using Gridap.Algebra: NewtonRaphsonSolver
using Gridap.CellData

conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u) #Changing to using the linear solver

Re = 5000 # 3500

# Physical constants
#u_max = 150 # 150 #150# 150#  150 #cm/s
L = 1 #cm
ρ = 1# 1.06e-3 #kg/cm^3 
μ = 1/Re# 3.50e-5 #kg/cm.s
ν = μ/ρ 
#Δt =  0.05 / 2 #0.1/10  # 0.046 / (u_max/2) # / (u_max) #/ 1000 # 0.046  #s \\

n=50
h=L/n

u_max=1
@show  Re = ρ*u_max*L/μ
#@show C_t = u_max*Δt/h

#n_t= 5#10000 # 5 #300
#t0 = 0.0
#tF = Δt * n_t
#dt = Δt

f(x) = VectorValue(0.0,0.0)

order = 1#2

# Select geometry
n = n
partition = (n,n)
D=length(partition)

# Setup background model
domain = (0,L,0,L)
bgmodel = simplexify(CartesianDiscreteModel(domain,partition))
#const h = L/n

labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

#Non-embedded geometry 
model=bgmodel

Ω = Triangulation(model)
degree = order
dΩ = Measure(Ω,degree)

Γ = BoundaryTriangulation(model)
dΓ = Measure(Γ,degree)
n_Γ = get_normal_vector(Γ)

reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
#Spaces
V0 = FESpace(
  model,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags=["diri0","diri1"]
  )

reffeₚ = ReferenceFE(lagrangian,Float64,order)

Q = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  constraint=:zeromean)

uD0 = VectorValue(0,0)
#uD1 = VectorValue(1,0)
uD1(x) = VectorValue(4*x[1]*(1-x[1]),0)
U = TrialFESpace(V0,[uD0,uD1])
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

#NITSCHE
#α_γ = 100
#γ(u) =  α_γ * ( ν / h  )#+  ρ * normInf(u) / 6 ) # Nitsche Penalty parameter ( γ / h ) 

#@show VD = ν / h
#@show CD = ρ * u_max / 6 
#@show TD = h*ρ / (12*θ*Δt)

#STABILISATION
α_τ = 0.0001 #1#0.2#0.03 #Tunable coefficiant (0,1)
h_SUPG(u)=h

#τ_SUPG(u) = α_τ * inv( sqrt(   ( 2 * normInf(u) / h_SUPG(u) )*( 2 * normInf(u) / h_SUPG(u) ) + 9 * ( 4*ν / (h_SUPG(u)*h_SUPG(u)) )*( 4*ν / (h_SUPG(u)*h_SUPG(u)) ) )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )
τ_SUPG(u) = α_τ * inv( sqrt(  ( 2 * (inner(u,u)) / h_SUPG(u) )*( 2  / h_SUPG(u) ) + 9 * ( 4*ν / (h_SUPG(u)*h_SUPG(u)) )*( 4*ν / (h_SUPG(u)*h_SUPG(u)) ) )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )

τ_PSPG(u) = τ_SUPG(u) # PSPG stabilisation - inf-sup stab  ( ρ^-1 * τ_PSPG(u) )


## Weak form terms
#Interior terms
m_Ω(ut,v) = ρ * ut⊙v
a_Ω(u,v) = μ * ∇(u)⊙∇(v) 
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#PSPG 
sp_Ω(w,p,q)    = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ ∇(p)
st_Ω(w,ut,q)   = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ ut
sc_Ω(w,u,q)    = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ conv(u, ∇(u))
dsc_Ω(w,u,du,q)= (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ dconv(du, ∇(du), u, ∇(u))
ϕ_Ω(w,q)     = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ f

#SUPG
sp_sΩ(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
st_sΩ(w,ut,v)   = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ ut
sc_sΩ(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩ(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩ(w,v)     = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f

res((u,p),(v,q)) = 
∫(   a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q)  + c_Ω(u,v) )dΩ + # + ρ * 0.5 * (∇⋅u) * u ⊙ v  
∫(- sp_Ω(u,p,q)    + ϕ_Ω(u,q)     - sc_Ω(u,u,q)  )dΩ +
∫(- sp_sΩ(u,p,v)   + ϕ_sΩ(u,v)    - sc_sΩ(u,u,v) )dΩ #+


jac((u,p),(du,dp),(v,q)) =
∫(  a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q)  + dc_Ω(u, du, v) )dΩ + # + ρ * 0.5 * (∇⋅u) * du ⊙ v 
∫( - sp_Ω(u,dp,q)  - dsc_Ω(u,u,du,q) )dΩ + 
∫(- sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) )dΩ #+ 

# #=
nls = NLSolver(
    show_trace = true,
    method = :newton,
    linesearch = BackTracking(),
)

#op = TransientFEOperator(res,jac,jac_t,X,Y)
op = FEOperator(res,jac,X,Y)
# Solve Navier-Stokes
nls = NLSolver(show_trace = true, method = :newton, linesearch = BackTracking())
solver = FESolver(nls)
xh = solve(solver, op)

# Adjoint operator
# adj_op = AdjointFEOperator(op,xh)

# Postprocess
filePath = "./test/GridapTests/ExtraTests/runs/solution_ConvStab_steady"
writevtk(Ω,filePath, cellfields = ["u"=>xh[1], "p"=>xh[2], "h" => h])

end
