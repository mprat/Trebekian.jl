#=
word -> length of word
* easy to generate data
* (should be) easy to train
* (should be) easy to test
=#

using Flux

train_data = (v -> Float32.(v)).([[1, 2, 3], [2, 3, 4], [1, 2]])
train_labels = [3.0, 3.0, 2.0]

test_data = 2 .* train_data
test_labels = train_labels

simple_rnn = Chain(Flux.RNN(1, 10), Dense(10, 1))

function loss(x, y)
  l = Flux.mse(simple_rnn.(x)[end], y)
  Flux.reset!(simple_rnn)
  return l
end

ps = Flux.params(simple_rnn)

# just use gradient descent
opt = Flux.Descent(0.01)

println(loss.(train_data, train_labels))
println(loss.(test_data, test_labels))

Flux.train!(loss, ps, zip(train_data, train_labels), opt)

# after training, evaluate the loss
println(loss.(test_data, test_labels))