module SUPG2D

using Gridap
using Test
using Gridap.CellData
using Plots


β = VectorValue(1,1)
u(x) = (1/10)*(4*sin(π*x[1]/2)+4*sin(x[1]*x[2])+7*sin(π*x[1]*x[2]/5)+5)
f(x) = β⋅∇(u)(x)

function run_test(n)

L=1
domain = (0,L,0,L)
cells = (n,n)
h=L/n
model = CartesianDiscreteModel(domain,cells)

order = 1
V = FESpace(model, ReferenceFE(lagrangian,Float64,order),conformity=:H1,dirichlet_tags="boundary")
U = TrialFESpace(V,u)

Ω = Triangulation(model)
#Γ = BoundaryTriangulation(model,tags=1)
#n_Γ = get_normal_vector(Γ)

degree = 2*order
dΩ = Measure(Ω,degree)
#dΓ = Measure(Γ,degree)

α_τ=1
#τ_SUPG(u) = α_τ * inv(sqrt( ( 2 * normInf(β) / h )*( 2 * normInf(β) / h )  )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )
τ_SUPG(u) = α_τ * inv(sqrt( ( 2 * sqrt(β⋅β) / h )*( 2 * sqrt(β⋅β) / h )  )) # SUPG Stabilisation - convection stab ( τ_SUPG(u )

conv(u, ∇u) = (∇u') ⋅ u
ac(u,v) = v * conv( β , ∇(u) )
sc_sΩ(u,v)   = τ_SUPG(β)   *  conv(β,∇(v)) ⋅ conv(β, ∇(u)) 
ϕ_sΩ(v)      = τ_SUPG(β)   *  conv(β,∇(v)) ⋅ f

a(u,v) = ∫( ac(u,v) + sc_sΩ(u,v) )*dΩ
l(v) = ∫( v*f + ϕ_sΩ(v) )*dΩ #+ ∫( v*(n_Γ⋅∇u) )*dΓ 

op = AffineFEOperator(a,l,U,V)
uh = solve(op)

e = u - uh

l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))


el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

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
      push!(hs,h)
  
    end
  
    (eul2s, euh1s, hs)
  
end
  
  const ID = 3
  const ns = [2,4,8,16,32]
  
  #global ID = ID+1
 
  @show hs=ns.^-1
  @show ms = hs


  eul2s, euh1s, hs = conv_test(ns);
  using Plots
  plot(hs,[eul2s,euh1s],
      xaxis=:log, yaxis=:log,
      label=["L2U" "m"],
      shape=:auto,
      xlabel="h",ylabel="L2 error norm",
      title = "ScalarAdvection2D_convergence,ID=$(ID)")
  savefig("ScalarAdvection2D_convergence$(ID)")
  
  
function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

@show slope(hs,eul2s)
@show slope(hs,euh1s)


end


