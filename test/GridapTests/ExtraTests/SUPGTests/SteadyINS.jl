module inscut

#formulation taken from: https://arxiv.org/pdf/1710.08898.pdf


#############
#WARNING: Steady solutions for high Re do in general not exist & maybe unstable.
#############
#############
#############


using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using Gridap.FESpaces: get_algebraic_operator
import Gridap: ∇
using LineSearches: BackTracking
using Gridap.Algebra: NewtonRaphsonSolver
using Gridap.CellData
using LineSearches: BackTracking

conv(u, ∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) #+ conv(du, ∇u) 

Re = ( 1.5 * 10^-4 )^-1 * 0.01
# Re = ( 15 )^-1 Diffusion dominated

# Physical constants
u_max = 1
L = 1 
ρ =  1
μ =  1/Re
ν = μ/ρ 

@show Re = ρ*u_max*L/(μ)

u1(x) = (1/10)*(4*sin(π*x[1]/2)+4*sin(π*x[2])+7*sin(π*x[1]*x[2]/5)+5)
u2(x) = (1/10)*(6*sin(4*π*x[1]/5)+3*sin(3*π*x[2]/10)+2*sin(3*π*x[1]*x[2]/10)+3)
u(x) = VectorValue(u1(x),u2(x))

p(x) = (1/2)*(sin(π*x[1]/2)+2*sin(3*π*x[2]/10)+sin(π*x[1]*x[2]/5)+1)

f(x) = - μ * Δ(u)(x) + ∇(p)(x) + ρ * conv( u(x) , ∇(u)(x) ) 
g(x) = (∇⋅u)(x)

order = 1

function run_test(n)

# Select geometry
n = n
partition = (n,n)
D=length(partition)
h=L/n

# Setup background model
domain = (0,L,0,L)
model = (CartesianDiscreteModel(domain,partition))
Ω = Triangulation(model)
degree = order*2
dΩ = Measure(Ω,degree)

#Spaces
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = FESpace(
  model,
  reffeᵤ,
  conformity=:H1,
  dirichlet_tags="boundary"
  )

reffeₚ = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(
  model,
  reffeₚ,
  conformity=:H1,
  dirichlet_tags="boundary"
  )

U = TrialFESpace(V,u)
P = TrialFESpace(Q,p)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

#STABILISATION
α_τ = 0.01

τ_SUPG(u) = α_τ * inv(sqrt(  ( 2 * normInf(u) / h )*( 2 * normInf(u) / h ) + 9 * ( 4*ν / h^2 )^2 )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )
#τ_SUPG(u) = α_τ * inv(sqrt( ( 2 * sqrt(u⋅u) / h )*( 2 * sqrt(u⋅u) / h )  + 9 * ( 4*ν / h^2 )^2  )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )

τ_PSPG(u) = τ_SUPG(u) # PSPG stabilisation - inf-sup stab  ( ρ^-1 * τ_PSPG(u) )

## Weak form terms
#Interior terms
a_Ω(u,v) = μ * ∇(u)⊙∇(v) 
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(u, v) = ρ *  v ⊙ conv(u, ∇(u))
dc_Ω(u, du, v) = ρ * v ⊙ dconv(du, ∇(du), u, ∇(u))

#PSPG 
sp_Ω(w,p,q)    = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ ∇(p)
sc_Ω(w,u,q)    = (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ conv(u, ∇(u))
dsc_Ω(w,u,du,q)= (ρ^(-1) * τ_PSPG(w)) * ρ *  ∇(q) ⋅ dconv(du, ∇(du), u, ∇(u))
ϕ_Ω(w,q)       = (ρ^(-1) * τ_PSPG(w))     *  ∇(q) ⋅ f

#SUPG
sp_sΩ(w,p,v)    = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ ∇(p)
sc_sΩ(w,u,v)    = τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ conv(u, ∇(u)) 
dsc_sΩ(w,u,du,v)= τ_SUPG(w) * ρ *  conv(w,∇(v)) ⋅ dconv(du, ∇(du), u, ∇(u)) 
ϕ_sΩ(w,v)       = τ_SUPG(w)     *  conv(w,∇(v)) ⋅ f 

res((u,p),(v,q)) = 
∫(  a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) - v⋅f  + q*g + c_Ω(u,v) )dΩ  + 
∫(- sp_Ω(u,p,q)     + ϕ_Ω(u,q)     - sc_Ω(u,u,q)  )dΩ +
∫(- sp_sΩ(u,p,v)    + ϕ_sΩ(u,v)    - sc_sΩ(u,u,v) )dΩ

jac((u,p),(du,dp),(v,q)) = 
∫( a_Ω(du,v) + b_Ω(v,dp) + b_Ω(du,q)  + dc_Ω(u, du, v) )dΩ + 
∫( - sp_Ω(u,dp,q)  - dsc_Ω(u,u,du,q) )dΩ + 
∫(- sp_sΩ(u,dp,v) - dsc_sΩ(u,u,du,v) )dΩ 

#op = FEOperator(res,X,Y)
op = FEOperator(res,jac,X,Y)

 #=
nls = NLSolver(
  show_trace=true, 
  method=:newton, 
  linesearch=BackTracking())
 =#

ls=LUSolver()
nls = NewtonRaphsonSolver(ls,1e-4,30)

solver = FESolver(nls)

uh, ph = solve(solver,op)

#writevtk(Ω,"results_steady_ins",cellfields=["uh"=>uh,"ph"=>ph])

eu = u - uh
ep = p - ph

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

eul2 = l2(eu)
euh1 = h1(eu)
epl2 = l2(ep)

@show (eul2,euh1,epl2,h)

(eul2,euh1,epl2,h)

end 

function conv_test(ns)

    eul2s = Float64[]
    euh1s = Float64[]
    epl2s = Float64[]
    hs = Float64[]
  
    for n in ns
  
      eul2, euh1,  epl2, h = run_test(n)
  
      push!(eul2s,eul2)
      push!(euh1s,euh1)
      push!(epl2s,epl2)
      push!(hs,h)
  
    end
  
    (eul2s, euh1s, epl2s, hs)
  
end
  
const ns = [8,16,32,64,128,256]

eul2s, euh1s, epl2s, hs = conv_test(ns);

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

@show mL2U = round(slope(hs,eul2s),digits=1)
@show mH1U = round(slope(hs,euh1s),digits=1)
@show mL2P = round(slope(hs,epl2s),digits=1)

using Plots
plot(hs,[eul2s,euh1s,epl2s],
    xaxis=:log, yaxis=:log,
    label=["L2U:$(mL2U)" "H1U:$(mH1U)" "L2P:$(mL2P)"],
    shape=:auto,
    xlabel="h",ylabel="L2 error norm",
    title = "steady_ins_convergence")
savefig("steady_ins_convergence")

end #module