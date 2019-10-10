#=
word -> length of word
* easy to generate data
* (should be) easy to train
* (should be) easy to test
=#

using Flux

data = [[1, 2, 3], [2, 3, 4], [1, 2, 3]]
labels = [3.0, 3.0, 2.0]

simple_net = Chain(
  Dense(3, 10, σ),
  Dense(10, 1))

loss(x, y) = Flux.mse(simple_net(x), y)

ps = Flux.params(simple_net)

# just use gradient descent
opt = Flux.Descent(0.01)

Flux.train!(loss, ps, zip(data, labels), opt)

# save the weights at the end
weights = Tracker.data.(params(simple_net));
using BSON: @save
@save "mymodel.bson" weights


# loading the model weights
# @load "mymodel.bson" weights
# Flux.loadparams!(simple_net, weights)