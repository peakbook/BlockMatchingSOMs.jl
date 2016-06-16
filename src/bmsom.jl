export BMSOM
export Online, Batch
export train!, labeling!, predict

abstract UpdateMode

type Online <: UpdateMode
end
type Batch <: UpdateMode
end

type Block
    pos :: Tuple{Int,Int}
    indices :: Tuple
    blksize :: Int
    distance :: Number 
    function Block(mapsize::Tuple, pos::Tuple{Int,Int}, blksize::Int)
        indices = (:, vec(Int[sub2ind(mapsize, i,j)
            for i=pos[1]:pos[1]+blksize-1,
                j=pos[2]:pos[2]+blksize-1]))
        new(pos, indices, blksize)
    end
end

type BMSOM{T<:AbstractFloat}
    mapsize :: Int
    refvecs :: Array{T,3}
    labels :: Array{Integer,2}
    metric :: Distances.PreMetric
    labeldict :: Array{Any}
    labelhist :: Array{T,3}

    function BMSOM(mapsize::Int, dim::Int;
                   init::Initializer=UniformInit(), metric::Distances.PreMetric=Euclidean())

        refvecs = initialize(T, (dim, mapsize, mapsize), init)
        labels = Array{Integer,2}(mapsize, mapsize)

        new(mapsize, refvecs, labels, metric)
    end
end


function Base.getindex(som::BMSOM, index...)
    som.refvecs[index...]
end

function Base.ndims(som::BMSOM)
    size(som.refvecs,1)
end

function Base.size(som::BMSOM)
    (som.mapsize, som.mapsize)
end

function calcdist!(blk::Block, iv::AbstractArray, refvecs::AbstractArray,  metric::Distances.PreMetric)
    mv = mean(sub(refvecs, blk.indices), 2)
    blk.distance = evaluate(metric, vec(mv), iv)
end

function update_loop!(refvecs::AbstractArray, drefvecs::AbstractArray, tv::AbstractArray, blk::Block, umode::Batch)
    blkrvs = sub(refvecs, blk.indices)
    dblkrvs = sub(drefvecs, blk.indices)
    d = (tv*ones(1, size(blkrvs,2)) - blkrvs)
    dblkrvs[:,:] += d/blk.blksize^2
end

function update_loop!(refvecs::AbstractArray, drefvecs::AbstractArray, tv::AbstractArray, blk::Block, umode::Online)
    blkrvs = sub(refvecs, blk.indices)
    d = (tv*ones(1, size(blkrvs,2)) - blkrvs)
    blkrvs[:,:] += d/blk.blksize^2
end

function update_after!{T}(refvecs::AbstractArray{T,3}, drefvecs::AbstractArray{T,3}, umode::Batch)
    refvecs[:,:,:] += drefvecs
    drefvecs[:] = zero(T)
end

function update_after!(refvecs::AbstractArray, drefvecs::AbstractArray, umode::Online)
    # do nothing
end

function findwinner(som::BMSOM, tv::AbstractArray, minblksize::Int=2)
    x,y = 1,1
    winblk = Block(size(som), (x,y), som.mapsize)
    calcdist!(winblk, tv, som.refvecs, som.metric)
    for blksize = som.mapsize-1:-1:minblksize
        wcands = Block[Block(size(som), (x+i,y+j), blksize) for i=0:1, j=0:1] 
        ds = map(blk->calcdist!(blk, tv, som.refvecs, som.metric), wcands) 
        val,idx = findmin(ds)
        if val < winblk.distance
            winblk = wcands[idx]
        end
        x,y = winblk.pos
    end

    return winblk
end

function train!{T<:AbstractFloat}(som::BMSOM{T}, tvs::AbstractArray{T,2}, epoch::Int;
                           minblksize::Int=2, umode::UpdateMode=Batch(), control::Bool=true)
    d_refvecs = zeros(T, size(som.refvecs))
    blksize = minblksize
    for i=1:epoch
        if control 
            blksize = Int(floor(som.mapsize-(som.mapsize-(minblksize))/epoch*i))
        end
        for di=1:size(tvs,2)
            winblk = findwinner(som, tvs[:,di], blksize)
            update_loop!(som.refvecs, d_refvecs, tvs[:,di], winblk, umode)
        end
        update_after!(som.refvecs, d_refvecs, umode)
    end
end

function labeling!{T<:AbstractFloat, S<:Any}(som::BMSOM{T}, tvs::AbstractArray{T,2}, tvls::AbstractArray{S,1})
    @assert(size(tvs,2) == size(tvls,1), "training data size error.")

    som.labeldict = unique(tvls)
    nlabels = length(som.labeldict)

    lbl2idx = Dict{S, Integer}()
    for i=1:nlabels
        lbl2idx[som.labeldict[i]] = i
    end

    som.labelhist = zeros(T, (nlabels, som.mapsize, som.mapsize))

    for j=1:som.mapsize, i=1:som.mapsize
        for di=1:size(tvs,2)
            som.labelhist[lbl2idx[tvls[di]],i,j] += evaluate(som.metric, som.refvecs[:,i,j], tvs[:,di])
        end
    end

    for i=1:som.mapsize,j=1:som.mapsize
        val, idx = findmin(som.labelhist[:,i,j])
        som.labels[i,j] = idx
    end
end

function predict{T<:AbstractFloat}(som::BMSOM{T}, iv::AbstractArray{T})
    dist = zeros(T, size(som))
    for j=1:som.mapsize, i=1:som.mapsize
        dist[i,j] = evaluate(som.metric, vec(som.refvecs[:,i,j]), iv)
    end
    val, idx = findmin(dist)

    som.labeldict[som.labels[idx]], val
end
