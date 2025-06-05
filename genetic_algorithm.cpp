#include <algorithm>

#include <genetic_algorithm.h>

Genetic_algorithm::Genome::Genome(uint32_t num_weights) {
	weights.resize(num_weights);

	for (auto& weight : weights) {
		weight = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
	}
}

Genetic_algorithm::Genetic_algorithm(uint32_t num_genomes, uint32_t num_weights_per_genome) {
	population.reserve(num_genomes);

	for (uint32_t idx_genome = 0; idx_genome < num_genomes; idx_genome++) {
		population.push_back(Genome(num_weights_per_genome));
	}
}

bool Genetic_algorithm::crossover(const Genome& parent_a, const Genome& parent_b, Genome& child) {
	if ((parent_a.weights.size() != parent_b.weights.size())
		|| (parent_a.weights.size() != child.weights.size())) {
		return false;
	}

	auto num_weights = parent_a.weights.size();

	for (int idx_weight = 0; idx_weight < num_weights; ++idx_weight) {
		child.weights[idx_weight] = (rand() % 2 == 0) ? parent_a.weights[idx_weight] : parent_b.weights[idx_weight];
	}

	return true;
}

void Genetic_algorithm::mutate(Genome& g) {
	for (float& w : g.weights) {
		if ((rand() / (float)RAND_MAX) < mutation_rate) {
			w += ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * mutation_stddev; // small tweak
		}
	}
}

bool Genetic_algorithm::new_generation() {
	std::sort(population.begin(), population.end(), [](const Genome& genome1, const Genome& genome2) {return genome1.fitness > genome2.fitness; });
	auto num_genomes = population.size();
	auto new_population = std::vector<Genome>();
	auto num_elites = static_cast<uint32_t>(num_genomes * elites_rate);

	new_population.insert(new_population.end(), population.begin(), population.begin() + num_elites);

	while (new_population.size() < num_genomes) {
		auto& parent_a = population[rand() % num_elites];
		auto& parent_b = population[rand() % num_elites];
		auto num_weights = static_cast<uint32_t>(parent_a.weights.size());
		auto child = Genome(num_weights);

		if (!crossover(parent_a, parent_b, child)) {
			return false;
		}

		mutate(child);
		new_population.push_back(child);
	}

	population = new_population;

	return true;
}