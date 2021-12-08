
# TO IMPROVE .... Perhaps via a Map?
# Do we have already this function in Gridap?
# What if there are no cells? I am assuming that there is at least one cell
# What if the number of dofs per field per cell is different among cells?
function _get_cell_dofs_field_offsets(uh::MultiFieldFEFunction,trian)
  U = get_fe_space(uh)
  uh_dofs = get_cell_dof_values(uh,trian)[1]
  nfields = length(U.spaces)
  dofs_field_offsets=Vector{Int}(undef,nfields+1)
  dofs_field_offsets[1]=1
  for i in 1:nfields
    dofs_field_offsets[i+1]=dofs_field_offsets[i]+length(uh_dofs.array[i])
  end
  dofs_field_offsets
end

function _compute_cell_ids(f::MultiFieldFEFunction,trian)
  uhs = f.single_fe_functions
  blockmask = [ is_change_possible(get_triangulation(uh),trian) for uh in uhs ]
  active_block_ids = findall(blockmask)
  active_block_data = Any[ FESpaces._compute_cell_ids(uhs[i],trian) for i in active_block_ids ]
  @show active_block_data
  nonzero_ids = findfirst(!iszero,active_block_data)
  abd = active_block_data[nonzero_ids]
end

function FESpaces._gradient(f,uh::MultiFieldFEFunction,fuh::DomainContribution)
  terms = DomainContribution()
  U = get_fe_space(uh)
  for trian in get_domains(fuh)
    g = FESpaces._change_argument(gradient,f,trian,uh)
    cell_u = lazy_map(DensifyInnerMostBlockLevelMap(),get_cell_dof_values(uh,trian))
    cell_id = _compute_cell_ids(uh,trian)
    cell_grad = autodiff_array_gradient(g,cell_u,cell_id)
    monolithic_result=cell_grad
    blocks = [] # TO-DO type unstable. How can I infer the type of its entries?
    nfields = length(U.spaces)
    cell_dofs_field_offsets=_get_cell_dofs_field_offsets(uh,trian)
    for i in 1:nfields
      view_range=cell_dofs_field_offsets[i]:cell_dofs_field_offsets[i+1]-1
      block=lazy_map(x->view(x,view_range),monolithic_result)
      append!(blocks,[block])
    end
    cell_grad=lazy_map(BlockMap(nfields,collect(1:nfields)),blocks...)
    add_contribution!(terms,trian,cell_grad)
  end
  terms
end

function FESpaces._jacobian(f,uh::MultiFieldFEFunction,fuh::DomainContribution)
  terms = DomainContribution()
  U = get_fe_space(uh)
  for trian in get_domains(fuh)
    g = FESpaces._change_argument(jacobian,f,trian,uh)
    cell_u = lazy_map(DensifyInnerMostBlockLevelMap(),get_cell_dof_values(uh))
    cell_id = FESpaces._compute_cell_ids(uh,trian)
    cell_grad = autodiff_array_jacobian(g,cell_u,cell_id)
    monolithic_result=cell_grad
    blocks        = [] # TO-DO type unstable. How can I infer the type of its entries?
    blocks_coords = Tuple{Int,Int}[]
    nfields = length(U.spaces)
    cell_dofs_field_offsets=_get_cell_dofs_field_offsets(uh)
    for j=1:nfields
      view_range_j=cell_dofs_field_offsets[j]:cell_dofs_field_offsets[j+1]-1
      for i=1:nfields
        view_range_i=cell_dofs_field_offsets[i]:cell_dofs_field_offsets[i+1]-1
        # TO-DO: depending on the residual being differentiated, we may end with
        #        blocks [i,j] full of zeros. I guess that it might desirable to early detect
        #        these zero blocks and use a touch[i,j]==false block in ArrayBlock.
        #        How can we detect that we have a zero block?
        block=lazy_map(x->view(x,view_range_i,view_range_j),monolithic_result)
        append!(blocks,[block])
        append!(blocks_coords,[(i,j)])
      end
    end
    cell_grad=lazy_map(BlockMap((nfields,nfields),blocks_coords),blocks...)
    add_contribution!(terms,trian,cell_grad)
  end
  terms
end

function FESpaces._change_argument(
  op::typeof(jacobian),f,trian,uh::MultiFieldFEFunction)

  U = get_fe_space(uh)
  function g(cell_u)
    single_fields = GenericCellField[]
    nfields = length(U.spaces)
    cell_dofs_field_offsets=_get_cell_dofs_field_offsets(uh)
    for i in 1:nfields
      view_range=cell_dofs_field_offsets[i]:cell_dofs_field_offsets[i+1]-1
      cell_values_field = lazy_map(a->view(a,view_range),cell_u)
      cf = CellField(U.spaces[i],cell_values_field)
      push!(single_fields,cf)
    end
    xh = MultiFieldCellField(single_fields)
    cell_grad = f(xh)
    cell_grad_cont_block=get_contribution(cell_grad,trian)
    bs = [cell_dofs_field_offsets[i+1]-cell_dofs_field_offsets[i] for i=1:nfields]
    lazy_map(DensifyInnerMostBlockLevelMap(),
             Fill(bs,length(cell_grad_cont_block)),
             cell_grad_cont_block)
  end
  g
end

function FESpaces._change_argument(
  op::typeof(gradient),f,trian,uh::MultiFieldFEFunction)
  U = get_fe_space(uh)
  function g(cell_u)
    single_fields = GenericCellField[]
    nfields = length(U.spaces)
    cell_dofs_field_offsets=_get_cell_dofs_field_offsets(uh,trian)
    for i in 1:nfields
      view_range=cell_dofs_field_offsets[i]:cell_dofs_field_offsets[i+1]-1
      cell_values_field = lazy_map(a->view(a,view_range),cell_u)
      cf = CellField(U.spaces[i],cell_values_field)
      push!(single_fields,cf)
    end
    xh = MultiFieldCellField(single_fields)
    cell_grad = f(xh)
    get_contribution(cell_grad,trian)
  end
  g
end

function Algebra.hessian(f::Function,uh::MultiFieldFEFunction)
  @notimplemented
end

#function FESpaces._change_argument(
#  op::typeof(hessian),f,trian,uh::MultiFieldFEFunction)
#
#  U = get_fe_space(uh)
#  function g(cell_u)
#    single_fields = GenericCellField[]
#    nfields = length(U.spaces)
#    for i in 1:nfields
#      cell_values_field = lazy_map(a->a.array[i],cell_u)
#      cf = CellField(U.spaces[i],cell_values_field)
#      cell_data = lazy_map(BlockMap((nfields,nfields),i),get_data(cf))
#      uhi = GenericCellField(cell_data,get_triangulation(cf),DomainStyle(cf))
#      push!(single_fields,uhi)
#    end
#    xh = MultiFieldCellField(single_fields)
#    cell_grad = f(xh)
#    get_contribution(cell_grad,trian)
#  end
#  g
#end
