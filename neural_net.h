#pragma once

#include <vector>

class Neural_net {
public:
	Neural_net(uint32_t num_inputs, uint32_t num_hidden, uint32_t num_outputs);
	bool forward(const std::vector<float>& vector_inputs, const std::vector<float>& vector_weights, uint32_t& idx_best_output);
private:
	uint32_t num_inputs = {};
	uint32_t num_hidden = {};
	uint32_t num_outputs = {};
	uint32_t num_expected_weights = {};
	std::vector<float> vector_hidden = {};
	std::vector<float> vector_outputs = {};
};