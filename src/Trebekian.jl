module Trebekian

using Flux
using Random
Random.seed!(1234);

simple_rnn = Chain(Flux.RNN(1, 10), Dense(10, 1))

function loss(x, y)
  l = Flux.mse(simple_rnn.(x)[end], y)
  Flux.reset!(simple_rnn)
  return l
end

# generate some data for training
# from the shared dict on your computer
# and return a training and testing batch
# TODO: specify how much data to use for both training and testing
function generate_data()
	words = readlines("/usr/share/dict/words")
	lens = length.(words)
	words = shuffle!((v -> Float32.(v)).(collect.(words)))

	# split into train and test batches - just use the last 10 as
	# my test
	num_in_test_set = Int(round(length(words) / 10))
	num_in_training_set = length(words) - num_in_test_set

	start_index = 1
	break_index = start_index + num_in_training_set
	end_index = break_index + num_in_test_set

	words[start_index:break_index - 1], lens[start_index:break_index - 1], words[break_index:end_index - 1], lens[break_index:end_index - 1]
end

function train_simplernn(train_data, train_labels, test_data, test_labels)
	simple_rnn = Chain(Flux.RNN(1, 10), Dense(10, 1))

	ps = Flux.params(simple_rnn)

	# just use gradient descent
	opt = Flux.Descent(0.01)

	# eval callback function
	evalcb() = @show(sum(loss.(test_data, test_labels)))
	evalcb()

	Flux.train!(loss, ps, zip(train_data, train_labels), opt, cb = Flux.throttle(evalcb, 1))

	# print the training loss at the end
	evalcb()
end

end # module
