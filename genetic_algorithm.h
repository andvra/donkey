#pragma once

#include <vector>

constexpr float mutation_rate = 0.1f;
constexpr float mutation_stddev = 0.2f;
constexpr float elites_rate = 0.05f;

class Genetic_algorithm {
public:
	struct Genome {
		Genome(uint32_t num_weights);

		std::vector<float> weights = {};
		float fitness = 0.0f;
	};

	Genetic_algorithm(uint32_t num_genomes, uint32_t num_weights_per_genome);
	bool crossover(const Genome& parent_a, const Genome& parent_b, Genome& child);
	void mutate(Genome& g);
	bool new_generation();

	std::vector<Genome> population = {};
};