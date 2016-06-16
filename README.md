# BlockMatchingSOMs
A naive implementation of Block Maching based Self-Organizing Maps in Julia.

## Example

```julia
using BlockMatchingSOMs
using Distances
using Compose
using Colors

labels = ["red", "green", "blue", "yellow", "pink",
          "magenta", "skyblue", "darkblue", "orange"]
data = reduce(hcat,map(labels) do x
    c = parse(Colorant, x)
    Float64[c.r, c.g, c.b]
end)

epoch = 2000
mapsize = 20
dim = size(data,1)
som = BMSOM{Float64}(mapsize, dim, metric=Euclidean())
train!(som, data, epoch, umode=Online())
labeling!(som, data, labels)

w=1.0/mapsize
c = [RGBA(som[:,i,j]...) for i=1:mapsize, j=1:mapsize]
img = SVG("map.svg", 6cm, 6cm)
draw(img, compose(context(),
[(context(i/mapsize, j/mapsize, w, w),
rectangle(), fill(c[i, j])) for i=1:mapsize, j=1:mapsize]...))

```

![SOM_MAP](http://peakbook.github.io/images/map.svg)

