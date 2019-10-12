#=
This snippet is a simple get-to-learn-RNNs implementation that trains a single-layer RNN
to take a list of numbers of arbitrary length and output it's sum.

This was chosen because:
* It is easy to generate data
* It is easy to test
* It (should be) easy to train
* It is easy to evaluate visually
=#

using Flux: @epochs

num_samples = 1000
num_epochs = 50

function generate_data(num_samples)
	# we just generate data of variable length from 2 to 7 elements with each element being
	# a float between 1 and 10, to keep it simple!
	train_data = [rand(1.0:10.0, rand(2:7)) for i in 1:num_samples]
	train_labels = (v -> sum(v)).(train_data)

	# why bother generating new data when you can just multiply your
	# test data! No really, in real models you never want to do this
	# because that means you're evaluating on your training data which
	# is a big no-no. For learning purposes, it works great!
	test_data = 2 .* train_data
	test_labels = 2 .* train_labels

	train_data, train_labels, test_data, test_labels
end

train_data, train_labels, test_data, test_labels = generate_data(num_samples)
# we use no activation because tanh limits to the range -1,1 which is NOT
# what you want out of a summation function
simple_rnn = Flux.RNN(1, 1, (x -> x))

function eval_model(x)
	out = simple_rnn.(x)[end]
	Flux.reset!(simple_rnn)
	out
end

loss(x, y) = abs(sum((eval_model(x) .- y)))

ps = Flux.params(simple_rnn)

opt = Flux.ADAM()

println("Training loss before = ", sum(loss.(train_data, train_labels)))
println("Test loss before = ", sum(loss.(test_data, test_labels)))

# callback function during training
evalcb() = @show(sum(loss.(test_data, test_labels)))

@epochs num_epochs Flux.train!(loss, ps, zip(train_data, train_labels), opt, cb = Flux.throttle(evalcb, 1))

# after training, evaluate the loss
println("Test loss after = ", sum(loss.(test_data, test_labels)))
