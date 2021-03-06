module CompressedCellArraysTests

using Test
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields

u(x) = x[1]^2 + x[2]

domain= (0,1,0,1)
cells = (3,3)
model = CartesianDiscreteModel(domain,cells)

Γ = BoundaryTriangulation(model)

cell_mat = [ones(3,3) for cell in 1:num_cells(Γ)]
cell_vec = [ones(3) for cell in 1:num_cells(Γ)]
cell_matvec = pair_arrays(cell_mat,cell_vec)

cell_to_bgcell = get_cell_to_bgcell(Γ)
ccell_to_first_cell = compress_ids(cell_to_bgcell)
@test ccell_to_first_cell == [1, 3, 4, 6, 7, 8, 10, 11, 13]

ccell_to_mat= compress_contributions(cell_mat,cell_to_bgcell,ccell_to_first_cell)
@test length(ccell_to_mat) == 8
@test ccell_to_mat[1] == 2*ones(3,3)
@test ccell_to_mat[1] == 2*ones(3,3)
@test ccell_to_mat[2] == ones(3,3)
@test ccell_to_mat[3] == 2*ones(3,3)
@test ccell_to_mat[3] == 2*ones(3,3)
test_array(ccell_to_mat,collect(ccell_to_mat))

cache = array_cache(ccell_to_mat)
@test getindex!(cache,ccell_to_mat,1) == 2*ones(3,3)
@test getindex!(cache,ccell_to_mat,1) == 2*ones(3,3)
@test getindex!(cache,ccell_to_mat,2) == ones(3,3)

ccell_to_vec = compress_contributions(cell_vec,cell_to_bgcell,ccell_to_first_cell)
@test ccell_to_vec[1] == 2*ones(3)
@test ccell_to_vec[1] == 2*ones(3)
@test ccell_to_vec[2] == ones(3)
@test ccell_to_vec[3] == 2*ones(3)
@test ccell_to_vec[3] == 2*ones(3)
test_array(ccell_to_vec,collect(ccell_to_vec))

cache = array_cache(ccell_to_vec)
@test getindex!(cache,ccell_to_vec,1) == 2*ones(3)
@test getindex!(cache,ccell_to_vec,1) == 2*ones(3)
@test getindex!(cache,ccell_to_vec,2) == ones(3)

ccell_to_matvec = compress_contributions(cell_matvec,cell_to_bgcell,ccell_to_first_cell)
@test ccell_to_matvec[1] == (2*ones(3,3), 2*ones(3))
@test ccell_to_matvec[1] == (2*ones(3,3), 2*ones(3))
@test ccell_to_matvec[2] == (ones(3,3)  , ones(3)  )
@test ccell_to_matvec[3] == (2*ones(3,3), 2*ones(3))
@test ccell_to_matvec[3] == (2*ones(3,3), 2*ones(3))
test_array(ccell_to_matvec,collect(ccell_to_matvec))

cache = array_cache(ccell_to_matvec)
@test getindex!(cache,ccell_to_matvec,1) == (2*ones(3,3), 2*ones(3))
@test getindex!(cache,ccell_to_matvec,1) == (2*ones(3,3), 2*ones(3))
@test getindex!(cache,ccell_to_matvec,2) == (ones(3,3)  , ones(3)  )

# now for blocks

mat = ArrayBlock(Matrix{Matrix{Float64}}(undef,2,2),[true false; true true])
mat[1,1] = ones(3,3)
mat[2,1] = ones(4,3)
mat[2,2] = ones(4,4)

vec = ArrayBlock(Vector{Vector{Float64}}(undef,2),[false, true])
vec[2] = ones(4)

cell_mat = [copy(mat) for cell in 1:num_cells(Γ)]
cell_vec = [copy(vec) for cell in 1:num_cells(Γ)]
cell_matvec = pair_arrays(cell_mat,cell_vec)
ccell_to_mat= compress_contributions(cell_mat,cell_to_bgcell,ccell_to_first_cell)
ccell_to_vec= compress_contributions(cell_vec,cell_to_bgcell,ccell_to_first_cell)
ccell_to_matvec= compress_contributions(cell_matvec,cell_to_bgcell,ccell_to_first_cell)
test_array(ccell_to_mat,collect(ccell_to_mat))
test_array(ccell_to_vec,collect(ccell_to_vec))
test_array(ccell_to_matvec,collect(ccell_to_matvec))

# now for blocks of blocks

mat11 = copy(mat)
mat = ArrayBlock(Matrix{MatrixBlock{Matrix{Float64}}}(undef,2,2),[true false; false false])
mat[1,1] = mat11

vec2 = copy(vec)
vec = ArrayBlock(Vector{VectorBlock{Vector{Float64}}}(undef,2),[false, true])
vec[2] = vec2

cell_mat = [copy(mat) for cell in 1:num_cells(Γ)]
cell_vec = [copy(vec) for cell in 1:num_cells(Γ)]
cell_matvec = pair_arrays(cell_mat,cell_vec)
ccell_to_mat= compress_contributions(cell_mat,cell_to_bgcell,ccell_to_first_cell)
ccell_to_vec= compress_contributions(cell_vec,cell_to_bgcell,ccell_to_first_cell)
ccell_to_matvec= compress_contributions(cell_matvec,cell_to_bgcell,ccell_to_first_cell)
test_array(ccell_to_mat,collect(ccell_to_mat))
test_array(ccell_to_vec,collect(ccell_to_vec))
test_array(ccell_to_matvec,collect(ccell_to_matvec))


end # module
