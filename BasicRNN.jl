#=
list -> sum of list
* easy to generate data
* (should be) easy to train
* (should be) easy to test
=#

using Flux: @epochs
using NNlib

num_samples = 1000
num_epochs = 50

function generate_data(num_samples)
	train_data = (v -> Float32.(v)).([rand(1.0:10.0, 3) for i in 1:num_samples])
	train_labels = (v -> sum(v)).(train_data)

	test_data = 2 .* train_data
	test_labels = (v -> sum(v)).(test_data)

	train_data, train_labels, test_data, test_labels
end

train_data, train_labels, test_data, test_labels = generate_data(num_samples)
# we use RELU activations because tanh limits to the range -1,1 which is NOT
# what you want out of a sum
# simple_rnn = Chain(Flux.RNN(3, 3, NNlib.relu), Flux.RNN(3, 1, NNlib.relu))
simple_rnn = Flux.RNN(1, 1, NNlib.leakyrelu)

function eval_model(x)
	simple_rnn.(x)[end]
end

function loss(x, y)
  l = abs(sum((eval_model(x) .- y)))
  Flux.reset!(simple_rnn)
  return l
end

ps = Flux.params(simple_rnn)

opt = Flux.ADAM()

println("Training loss before = ", sum(loss.(train_data, train_labels)))
println("Test loss before = ", sum(loss.(test_data, test_labels)))

# callback function during training
evalcb() = @show(sum(loss.(test_data, test_labels)))

@epochs num_epochs Flux.train!(loss, ps, zip(train_data, train_labels), opt, cb = Flux.throttle(evalcb, 1))

# after training, evaluate the loss
println("Test loss after = ", sum(loss.(test_data, test_labels)))
