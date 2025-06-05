#include <cmath>

#include <neural_net.h>

Neural_net::Neural_net(uint32_t num_inputs, uint32_t num_hidden, uint32_t num_outputs) :
	num_inputs(num_inputs), num_hidden(num_hidden), num_outputs(num_outputs) {
	if (num_hidden > 0) {
		vector_hidden.resize(num_hidden, 0.0f);
	}

	if (num_outputs > 0) {
		vector_outputs.resize(num_outputs, 0.0f);
	}

	num_expected_weights = (num_inputs * num_hidden) + (num_hidden * num_outputs);
}

bool Neural_net::forward(const std::vector<float>& vector_inputs, const std::vector<float>& vector_weights, uint32_t& idx_best_output) {
	if (vector_inputs.size() != num_inputs) {
		return false;
	}

	if (vector_weights.size() != num_expected_weights) {
		return false;
	}

	auto hidden = vector_hidden.data();
	auto outputs = vector_outputs.data();
	auto inputs = vector_inputs.data();
	auto weights = vector_weights.data();

	for (uint32_t idx_hidden = 0; idx_hidden < num_hidden; ++idx_hidden) {
		hidden[idx_hidden] = 0.0f;
		for (uint32_t idx_input = 0; idx_input < num_inputs; ++idx_input) {
			hidden[idx_hidden] += inputs[idx_input] * weights[idx_input * num_hidden + idx_hidden];
		}

		hidden[idx_hidden] = std::tanh(hidden[idx_hidden]);
	}

	auto offset = num_inputs * num_hidden;

	for (uint32_t idx_output = 0; idx_output < num_outputs; ++idx_output) {
		outputs[idx_output] = 0.0f;
		for (uint32_t idx_hidden = 0; idx_hidden < num_hidden; ++idx_hidden) {
			outputs[idx_output] += hidden[idx_hidden] * weights[offset + idx_hidden * num_outputs + idx_output];
		}
	}

	auto best_output_val = std::max_element(vector_outputs.begin(), vector_outputs.end());
	idx_best_output = static_cast<uint32_t>(std::distance(vector_outputs.begin(), best_output_val));

	return true;
}