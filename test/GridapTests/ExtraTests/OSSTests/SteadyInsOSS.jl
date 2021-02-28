module inscut

#using Pkg
#Pkg.activate(".")

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
using Gridap
using LineSearches: BackTracking
using WriteVTK

conv(u,∇u) = (∇u') ⋅ u
dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u) 

Re= (1.5 * 10^-4)^-1  

# Physical constants
u_max = 1#150 # 150 #150# 150#  150 #cm/s
L = 1 #cm
ρ =  1# 1.06e-3 #kg/cm^3 
μ =  1/Re# 3.50e-5 #kg/cm.s
ν = μ/ρ 
#Δt =  0.046 / (u_max/2) # / (u_max) #/ 1000 # 0.046  #s \\

@show Re = ρ*u_max*L/(μ)

#u1(x) = (1/10)*(x[2])
#u2(x) = (1/10)*(-x[1])
u1(x) = (1/10)*(4*sin(π*x[1]/2)+4*sin(π*x[2])+7*sin(π*x[1]*x[2]/5)+5)
u2(x) = (1/10)*(6*sin(4*π*x[1]/5)+3*sin(3*π*x[2]/10)+2*sin(3*π*x[1]*x[2]/10)+3)
u_sol(x) = VectorValue(u1(x),u2(x))
u(x) = u_sol(x)

#p(x) = x[1]-x[2]
p(x) = (1/2)*(sin(π*x[1]/2)+2*sin(3*π*x[2]/10)+sin(π*x[1]*x[2]/5)+1)

f(x) = - μ * Δ(u)(x) + ∇(p)(x) + ρ * conv( u(x) , ∇(u)(x) )  
g(x) = (∇⋅u)(x)

function run_test(n)

order = 2

# Select geometry
n = n
partition = (n,n)
D=length(partition)
h=L/n
@show Re_e = ρ*u_max*h/(μ)

# Setup background model
domain = (0,L,0,L)
bgmodel = (CartesianDiscreteModel(domain,partition))
#const h = L/n

#Non-embedded geometry 
model=bgmodel

Ω = Triangulation(model)
degree = order*2
dΩ = Measure(Ω,degree)

order = order
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags="boundary")
Q = TestFESpace(model,reffeₚ,conformity=:H1,constraint=:zeromean)
#Q = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags="boundary")
W = TestFESpace(model,reffeᵤ,conformity=:H1)
U = TrialFESpace(V,u)
P = TrialFESpace(Q,p)
S = TrialFESpace(W)
X = MultiFieldFESpace([U,P,S])
Y = MultiFieldFESpace([V,Q,W])
X₀ = MultiFieldFESpace([U,P])
Y₀ = MultiFieldFESpace([V,Q])

# Stabilization Parameters
c₁ = 12.0; c₂ = 2.0; c₃ = 1.0
h² = get_cell_measure(Ω)
h = evaluate(x->.√(x),h²)
u_abs(u) = √(u⋅u)
τₘ(u_abs) = 1.0 / (c₁*ν/(h²) + c₂*u_abs/h)
τc(u_abs) = c₃ * (ν + c₂/c₁*h*u_abs)

# Navier-Stokes weak form
res((u,p,η),(v,q,κ)) = ∫( 
(conv∘(u,∇(u)))⋅v + μ * ∇(u)⊙∇(v) - (∇⋅v)*p + q*(∇⋅u) - f⋅v - g⋅q +
τₘ(u_abs∘u)*(conv∘(u,∇(u)))⋅(conv∘(u,∇(v))) -
τₘ(u_abs∘u)*η⋅(conv∘(u,∇(v))) +
τₘ(u_abs∘u)*(conv∘(u,∇(u)))⋅κ - 
τₘ(u_abs∘u)*η⋅κ + 
τc(u_abs∘u)*(∇⋅u)*(∇⋅v) 
)dΩ

jac((u,p,η),(du,dp,dη),(v,q,κ)) = 
∫( 
dconv(du, ∇(du), u, ∇(u) )⋅v + μ * ∇(du)⊙∇(v) - (∇⋅v)*dp + q*(∇⋅du) +
τₘ(u_abs∘u)*(conv∘(u,∇(du)))⋅(conv∘(u,∇(v))) - 
τₘ(u_abs∘u)*dη⋅(conv∘(u,∇(v))) +
τₘ(u_abs∘u)*(conv∘(u,∇(du)))⋅κ - 
τₘ(u_abs∘u)*dη⋅κ + 
τc(u_abs∘u)*(∇⋅du)*(∇⋅v) 
)dΩ

op = FEOperator(res,jac,X,Y)

nls = NLSolver(
  show_trace=true, 
  method=:newton, 
  linesearch=BackTracking()
  )

solver = FESolver(nls)

@show sqrt(sum( ∫( g )*dΩ ))

uh, ph = solve(solver,op)

e = u - uh

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

@show el2 = l2(e)
@show eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

(el2,eh1,h)

i=1
writevtk(Ω,"results_steadyins_$(i)",cellfields=["uh"=>uh,"ph"=>ph])

(el2,eh1,h)

end #function run_test

function conv_test(ns)

    eul2s = Float64[]
    euh1s=Float64[]

    hs = Float64[]
  
    for n in ns
  
      eul2, euh1,  h = run_test(n)
  
      push!(eul2s,eul2)
      push!(euh1s,euh1)
      push!(hs,h[1])
  
    end
  
    (eul2s, euh1s, hs)
  
end
  
  const ID = 3
  const ns = [2,4,8,16,32,64]#,128,256]
  
  #global ID = ID+1
 
  @show hs=ns.^-1
  @show ms = hs


  eul2s, euh1s, hs = conv_test(ns);
  using Plots
  plot(hs,[eul2s,euh1s],
      xaxis=:log, yaxis=:log,
      label=["L2U" "H1U"],
      shape=:auto,
      xlabel="h",ylabel="L2 error norm",
      title = "steadyins_convergence_orth,ID=$(ID)")
  savefig("steadins_convergence_oss_$(ID)")
  
function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

@show slope(hs,eul2s)
@show slope(hs,euh1s)

end #module