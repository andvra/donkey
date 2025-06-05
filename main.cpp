#include <algorithm>
#include <iostream>
#include <numbers>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

const int INPUT_SIZE = 9;
const int HIDDEN_SIZE = 2 * INPUT_SIZE;
const int OUTPUT_SIZE = 3;
const int WEIGHT_COUNT = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE); // No biases for simplicity
const int POP_SIZE = 500;
const int GENERATIONS = 100;
const float MUTATION_RATE = 0.1f;
const float MUTATION_STDDEV = 0.2f;
const int SIM_STEPS = 20;

enum Action { LEFT = 0, RIGHT = 1, JUMP = 2 };

class Neural_net {
public:
	uint32_t num_inputs = {};
	uint32_t num_hidden = {};
	uint32_t num_outputs = {};
	uint32_t num_expected_weights = {};
	std::vector<float> vector_hidden = {};
	std::vector<float> vector_outputs = {};

	Neural_net(uint32_t num_inputs, uint32_t num_hidden, uint32_t num_outputs) :
		num_inputs(num_inputs), num_hidden(num_hidden), num_outputs(num_outputs) {
		if (num_hidden > 0) {
			vector_hidden.resize(num_hidden, 0.0f);
		}

		if (num_outputs > 0) {
			vector_outputs.resize(num_outputs, 0.0f);
		}

		num_expected_weights = (num_inputs * num_hidden) + (num_hidden * num_outputs);
	}

	bool forward(const std::vector<float>& vector_inputs, const std::vector<float>& vector_weights, uint32_t& idx_best_output) {
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
};

struct Genome {
	std::vector<float> weights;
	float fitness;

	Genome() : weights(WEIGHT_COUNT), fitness(0.0f) {
		for (auto& w : weights) w = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f); // [-1, 1]
	}
};

Genome crossover(const Genome& a, const Genome& b) {
	Genome child;
	for (int i = 0; i < WEIGHT_COUNT; ++i) {
		child.weights[i] = (rand() % 2 == 0) ? a.weights[i] : b.weights[i];
	}
	return child;
}

void mutate(Genome& g) {
	for (float& w : g.weights) {
		if ((rand() / (float)RAND_MAX) < MUTATION_RATE)
			w += ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * MUTATION_STDDEV; // small tweak
	}
}

const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    uniform vec2 offset;
    uniform mat4 uProjection;

    void main() {
        gl_Position = uProjection * vec4(aPos + offset, 0.0, 1.0);
    }
)glsl";

const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;
    uniform vec4 uColor;

    void main() {
        FragColor = uColor;
    }
)glsl";

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

struct Entity {
	int offset_x = {};
	int offset_y = {};
	int width = {};
	int height = {};
	int v_x = {};
	int v_y = {};
	bool is_on_ground = false;
	int level = 0;
};

struct Player : public Entity {
	int score = 0;
	bool alive = false;
	int dead_at_step = 0;
};

struct Line_segment {
	int x_start = {};
	int y_start = {};
	int x_end = {};
	int y_end = {};
};

struct Shader_locations {
	int offset = {};
	int color = {};
	int projection = {};
};

struct Buffer_info {
	GLuint vao = {};
	GLuint vbo = {};
	GLuint ebo = {};
};

void spawn_barrel(std::vector<Entity>& barrels) {
	auto barrel = Entity();

	barrel.is_on_ground = false;
	barrel.height = 8;
	barrel.width = 8;
	barrel.offset_y = 100;
	barrel.offset_x = 0;
	barrel.v_x = 2 * (2 * (std::rand() % 2) - 1);

	barrels.push_back(barrel);
}

void physics(int num_physics_steps, std::vector<Line_segment>& line_segments, std::vector<Player>& players, std::vector<Entity>& barrels) {
	auto max_x = 14 * 8;

	auto get_collision_line_segment = [&line_segments](Entity& entity, int y_before, int y_after) {
		auto x_left = entity.offset_x - entity.width / 2;
		auto x_right = entity.offset_x + entity.width / 2;
		Line_segment* cur_line_segment = nullptr;

		for (auto& line_segment : line_segments) {
			auto line_x_min = std::min(line_segment.x_start, line_segment.x_end);
			auto line_x_max = std::max(line_segment.x_start, line_segment.x_end);
			auto has_overlap_x_left = (std::clamp(x_left, line_x_min, line_x_max) == x_left);
			auto has_overlap_x_right = (std::clamp(x_right, line_x_min, line_x_max) == x_right);
			auto has_overlap_x = has_overlap_x_left || has_overlap_x_right;

			if (!has_overlap_x) {
				continue;
			}

			// Assume start/end y is same value
			auto line_y = line_segment.y_start;

			if ((y_before > line_y) && (y_after <= line_y)) {
				if (cur_line_segment == nullptr) {
					cur_line_segment = &line_segment;
				}
				else if (cur_line_segment->y_start < line_y) {
					cur_line_segment = &line_segment;
				}
			}
		}

		return cur_line_segment;
		};

	auto apply_gravity = [&get_collision_line_segment](Entity& entity) {
		Line_segment* line_segment_collision = nullptr;

		if (!entity.is_on_ground) {
			if (entity.v_y < 0) {
				line_segment_collision = get_collision_line_segment(entity, entity.offset_y - entity.height / 2 + 2, entity.offset_y - entity.height / 2 + entity.v_y);
			}
		}
		else {
			line_segment_collision = get_collision_line_segment(entity, entity.offset_y - entity.height / 2 + 2, entity.offset_y - entity.height / 2 - 1);
		}

		if (line_segment_collision) {
			entity.offset_y = line_segment_collision->y_start + entity.height / 2;
			entity.is_on_ground = true;
			entity.v_y = 0;
			entity.level = 0;

			if (entity.offset_y >= -92) {
				entity.level = 1;
			}
			if (entity.offset_y >= -57) {
				entity.level = 2;
			}
			if (entity.offset_y >= -26) {
				entity.level = 3;
			}
			if (entity.offset_y >= 6) {
				entity.level = 4;
			}
			if (entity.offset_y >= 39) {
				entity.level = 5;
			}
		}
		else {
			if (entity.is_on_ground) {
				entity.v_y = -1;
				entity.is_on_ground = false;
			}
		}

		if (!entity.is_on_ground) {
			entity.offset_y += entity.v_y;
			entity.v_y -= 1;
		}
		};

	struct Hit_info {
		bool hit_wall = false;
	};

	auto apply_movement = [&max_x](Entity& entity) {
		auto offset_x_before = entity.offset_x;
		entity.offset_x = std::clamp(entity.offset_x + entity.v_x, -max_x, max_x);
		auto hit_wall = (entity.offset_x == offset_x_before) && (entity.v_x != 0);

		return Hit_info{ hit_wall };
		};

	for (auto& player : players) {
		if (!player.alive) {
			continue;
		}
		apply_gravity(player);
		apply_movement(player);
	}

	for (auto& barrel : barrels) {
		apply_gravity(barrel);
		auto hit_info = apply_movement(barrel);
		if (hit_info.hit_wall) {
			barrel.v_x = -barrel.v_x;
		}
	}

	for (auto& player : players) {
		if (!player.alive) {
			continue;
		}
		if (!player.is_on_ground) {
			// TODO: Think we should always be alive if we are in the air?
			continue;
		}
		for (auto& barrel : barrels) {
			auto ok1 = player.offset_x <= (barrel.offset_x + barrel.width / 2);
			auto ok2 = player.offset_x >= (barrel.offset_x - barrel.width / 2);
			auto ok3 = player.offset_y <= (barrel.offset_y + barrel.height / 2);
			auto ok4 = player.offset_y >= (barrel.offset_y - barrel.height / 2);
			if (ok1 && ok2 && ok3 && ok4) {
				player.alive = false;
				player.dead_at_step = num_physics_steps;
				// TODO: We assume the last line segment is the one at the bottom of the board.
				//	Should be an alright assumption
				player.score = player.offset_y - line_segments.back().y_end;
				break;
			}
		}
	}
}

void jump(Player& player) {
	if (!player.is_on_ground) {
		return;
	}

	player.v_y = 6;
	player.is_on_ground = false;
}

void move_left(Player& player) {
	player.v_x = -1;
}

void move_right(Player& player) {
	player.v_x = 1;
}

void brain_run_human(GLFWwindow* window, Player& player) {
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		move_left(player);
	}

	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		move_right(player);
	}

	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		jump(player);
	}
}

std::vector<Genome> population(POP_SIZE);
auto neural_net = Neural_net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

void brain_run_machine(std::vector<Line_segment>& line_segments, std::vector<Player>& players, std::vector<Entity>& barrels) {
	for (auto idx_player = 0; idx_player < players.size(); idx_player++) {
		auto& player = players[idx_player];
		auto& genome = population[idx_player];

		auto distance_ceiling = 100.0f;
		auto level = (float)player.level;

		for (auto& line_segment : line_segments) {
			if (line_segment.y_start < player.offset_y) {
				continue;
			}
			auto ok1 = (line_segment.x_start <= (player.offset_x + player.width / 2));
			auto ok2 = (line_segment.x_end >= (player.offset_x - player.width / 2));
			if (ok1 && ok2) {
				auto cur_distance_ceiling = (float)(line_segment.y_start - player.offset_y);
				if (cur_distance_ceiling < distance_ceiling) {
					distance_ceiling = cur_distance_ceiling;
				}
			}
		}

		struct Barrel_distance {
			Entity* barrel = nullptr;
			float angle = 0.0f;
			float distance = 0.0f;
		};

		auto barrel_distances = std::vector<Barrel_distance>(2);

		for (auto& barrel_distance : barrel_distances) {
			barrel_distance.distance = 100.0f;
		}

		auto idx_cur_worst = 0;

		for (auto& barrel : barrels) {
			auto distance = std::hypotf((float)barrel.offset_x - player.offset_x, (float)barrel.offset_y - player.offset_y);

			if (distance < barrel_distances[idx_cur_worst].distance) {
				auto angle = std::atan2f((float)barrel.offset_y - player.offset_y, (float)barrel.offset_x - player.offset_x);
				auto new_barrel_distance = Barrel_distance();
				new_barrel_distance.angle = angle;
				new_barrel_distance.distance = distance;
				new_barrel_distance.barrel = &barrel;
				barrel_distances[idx_cur_worst] = new_barrel_distance;
			}
		}

		auto is_on_ground = (player.is_on_ground ? 1.0f : 0.0f);

		// Normalizing
		auto player_offset_x = player.offset_x / 100.0f;
		auto player_offset_y = player.offset_y / 100.0f;
		level /= 5;

		for (auto& barrel_distance : barrel_distances) {
			barrel_distance.angle /= std::numbers::pi_v<float>;
			barrel_distance.distance /= 100.0f;
		}

		distance_ceiling /= 100.0f;

		auto inputs = std::vector<float>{
			is_on_ground,
			player_offset_x,
			player_offset_y,
			level,
			barrel_distances[0].distance,
			barrel_distances[0].angle,
			barrel_distances[1].distance,
			barrel_distances[1].angle,
			distance_ceiling
		};

		auto idx_best_output = uint32_t{};

		if (!neural_net.forward(inputs, genome.weights, idx_best_output)) {
			std::cout << "Could not feed-forward\n";
		}

		auto action = static_cast<Action>(idx_best_output);

		switch (action) {
		case Action::JUMP: jump(player); break;
		case Action::LEFT: move_left(player); break;
		case Action::RIGHT: move_right(player); break;
		}
	}
}

void brain_run(GLFWwindow* window, std::vector<Line_segment>& line_segments, std::vector<Player>& players, std::vector<Entity>& barrels, bool is_human) {
	if (players.empty()) {
		return;
	}

	for (auto& player : players) {
		player.v_x = 0;
	}

	if (is_human) {
		brain_run_human(window, players[0]);
	}
	else {
		brain_run_machine(line_segments, players, barrels);
	}
}

void brain_update(const std::vector<Player>& players) {
	static auto best_score_overall = 0.0f;
	static auto best_level_overall = 0;

	auto best_level = 0;

	for (auto idx_agent = 0; idx_agent < POP_SIZE; idx_agent++) {
		population[idx_agent].fitness = (float)players[idx_agent].score;
		best_level = std::max(best_level, players[idx_agent].level);
	}

	best_level_overall = std::max(best_level_overall, best_level);

	std::sort(population.begin(), population.end(), [](const Genome& genome1, const Genome& genome2) {return genome1.fitness > genome2.fitness; });

	best_score_overall = std::max(best_score_overall, population.front().fitness);
	std::cout << "Best score in generation (best total): " << population.front().fitness << " (" << best_score_overall << ")\n";
	std::cout << "Best level in generation (best total): " << best_level << " (" << best_level_overall << ")\n";

	std::vector<Genome> new_pop;
	int elites = POP_SIZE / 20;
	new_pop.insert(new_pop.end(), population.begin(), population.begin() + elites);

	while (new_pop.size() < POP_SIZE) {
		const Genome& p1 = population[rand() % elites];
		const Genome& p2 = population[rand() % elites];
		Genome child = crossover(p1, p2);
		mutate(child);
		new_pop.push_back(child);
	}

	population = new_pop;
}

void game_logics(int num_physics_steps, std::vector<Entity>& barrels) {
	if (num_physics_steps % 100 == 0) {
		spawn_barrel(barrels);
	}
}

void init_players(std::vector<Player>& players, bool is_human, int player_width, int player_height) {
	auto num_agents = POP_SIZE;
	auto num_players = is_human ? 1 : num_agents;

	players.resize(num_agents);

	for (auto idx_player = 0; idx_player < num_players; idx_player++) {
		auto player = Player();
		player.offset_x = -50;
		player.offset_y = -100;
		player.width = player_width;
		player.height = player_height;
		player.alive = true;
		players[idx_player] = player;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int)
{
	auto window_width = 800 * 2;
	auto window_height = 600 * 2;
	auto square_size_pixels = 8;
	auto num_squares_x = 28;
	auto num_squares_y = 32;
	auto board_width = num_squares_x * square_size_pixels;
	auto board_height = num_squares_y * square_size_pixels;
	auto scale = 4.0f;

	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		xpos -= window_width / 2.0;
		ypos -= window_height / 2.0;
		auto ok_x = std::abs(xpos) <= scale * board_width / 2.0;
		auto ok_y = std::abs(ypos) <= scale * board_height / 2.0;
		if (ok_x && ok_y) {
			xpos /= scale;
			ypos = (-ypos) / scale;
			std::cout << "x: " << (int)xpos << " y: " << (int)ypos << std::endl;
		}
	}
}

void render(int num_physics_steps, const std::vector<Player>& players, const std::vector<Entity>& barrels, const std::vector<Line_segment>& line_segments, const Shader_locations& shader_locations,
	const Buffer_info& buffer_info_background, const Buffer_info& buffer_info_player, const Buffer_info& buffer_info_barrel, const Buffer_info& buffer_info_lines) {
	// Background
	glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Game background
	glUniform2f(shader_locations.offset, 0.0f, 0.0f);
	glUniform4f(shader_locations.color, 0.0f, 0.0f, 0.0f, 1.0f);
	glBindVertexArray(buffer_info_background.vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	// Lines
	glUniform4f(shader_locations.color, 240.0f / 255, 82.0f / 255, 156.0f / 255, 1.0f);
	glBindVertexArray(buffer_info_lines.vao);
	glDrawArrays(GL_LINES, 0, (GLsizei)line_segments.size() * sizeof(line_segments[0]));

	// Barrels
	for (auto& barrel : barrels) {
		glUniform2f(shader_locations.offset, (float)barrel.offset_x, (float)barrel.offset_y);
		glUniform4f(shader_locations.color, 0.7f, 0.4f, 0.4f, 1.0f);
		glBindVertexArray(buffer_info_barrel.vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	}

	// Players
	auto static colors = std::vector<glm::vec3>{
		{0.2f, 0.4f, 1.0f},
		{0.3f, 0.5f, 0.8f},
		{0.5f, 0.2f, 0.7f},
		{0.7f, 0.8f, 0.7f},
		{0.9f, 0.2f, 0.5f},
		{0.5f, 0.7f, 0.2f},
	};

	for (auto& player : players) {
		if (!player.alive && ((num_physics_steps - player.dead_at_step) > 500)) {
			continue;
		}

		glUniform2f(shader_locations.offset, (float)player.offset_x, (float)player.offset_y);

		auto color = colors[player.level];

		if (!player.alive) {
			color *= 0.5;
		}

		glUniform4f(shader_locations.color, color.r, color.g, color.b, 1.0f);
		glBindVertexArray(buffer_info_player.vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	}
}

int main() {
	auto window_width = 800 * 2;
	auto window_height = 600 * 2;

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Donkey", nullptr, nullptr);

	glfwSetMouseButtonCallback(window, mouse_button_callback);

	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	// Build and compile shaders
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);

	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	auto is_human = false;
	auto players = std::vector<Player>();
	auto player_width = 8;
	auto player_height = 8;

	init_players(players, is_human, player_width, player_height);

	auto vertices_entity_8x8 = std::vector<float>{
		-player_width / 2.0f, -player_height / 2.0f,
		-player_width / 2.0f, player_height / 2.0f,
		player_width / 2.0f, player_height / 2.0f,
		player_width / 2.0f, -player_height / 2.0f
	};

	auto indices_entity_8x8 = std::vector<int>{
		0,1,2,
		2,3,0
	};

	auto buffer_info_player = Buffer_info{};

	glGenVertexArrays(1, &buffer_info_player.vao);
	glGenBuffers(1, &buffer_info_player.vbo);
	glGenBuffers(1, &buffer_info_player.ebo);
	glBindVertexArray(buffer_info_player.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_info_player.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_entity_8x8), vertices_entity_8x8.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_info_player.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_entity_8x8), indices_entity_8x8.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	auto buffer_info_barrel = Buffer_info{};

	glGenVertexArrays(1, &buffer_info_barrel.vao);
	glGenBuffers(1, &buffer_info_barrel.vbo);
	glGenBuffers(1, &buffer_info_barrel.ebo);
	glBindVertexArray(buffer_info_barrel.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_info_barrel.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_entity_8x8), vertices_entity_8x8.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_info_barrel.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_entity_8x8), indices_entity_8x8.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	auto square_size_pixels = 8;
	auto num_squares_x = 28;
	auto num_squares_y = 32;
	auto board_width = num_squares_x * square_size_pixels;
	auto board_height = num_squares_y * square_size_pixels;

	float bgVertices[] = {
		-board_width / 2.0f, -board_height / 2.0f,
		 board_width / 2.0f, -board_height / 2.0f,
		 board_width / 2.0f,  board_height / 2.0f,
		-board_width / 2.0f,  board_height / 2.0f
	};

	unsigned int bgIndices[] = {
		0, 1, 2,
		2, 3, 0
	};

	auto buffer_info_background = Buffer_info();

	glGenVertexArrays(1, &buffer_info_background.vao);
	glGenBuffers(1, &buffer_info_background.vbo);
	glGenBuffers(1, &buffer_info_background.ebo);
	glBindVertexArray(buffer_info_background.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_info_background.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(bgVertices), bgVertices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_info_background.ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bgIndices), bgIndices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	auto buffer_info_lines = Buffer_info();
	auto line_segments = std::vector<Line_segment>();

	line_segments.push_back({
		-3 * 8, 9 * 8,
		3 * 8, 9 * 8 });

	line_segments.push_back({
		-(num_squares_x / 2) * 8, 5 * 8 + 4,
		4 * 8, 5 * 8 + 4 });

	auto generate_line_vertices = [&line_segments](uint32_t num_blocks, int x_offset, int y_offset, int x_factor) {
		for (auto idx_block = 0; idx_block < (int)num_blocks; idx_block++) {
			int pix_x_start = x_offset + 8 * (x_factor * 2 * idx_block);
			int pix_x_end = x_offset + 8 * (x_factor * 2 * (idx_block + 1));
			int pix_y_start = y_offset - idx_block;
			int pix_y_end = pix_y_start;
			line_segments.push_back({
				pix_x_start, pix_y_start,
				pix_x_end, pix_y_end });
		}
		};

	generate_line_vertices(4, 4 * 8, 5 * 8 + 3, 1);
	generate_line_vertices(13, 8 * num_squares_x / 2, 8 * 2 + 3, -1);
	generate_line_vertices(13, -8 * num_squares_x / 2, -8 * 1 - 6, 1);
	generate_line_vertices(13, 8 * num_squares_x / 2, -8 * 5 - 6, -1);
	generate_line_vertices(13, -8 * num_squares_x / 2, -8 * 10, 1);
	generate_line_vertices(7, 8 * num_squares_x / 2, -8 * 14 - 2, -1);

	line_segments.push_back({
		(-num_squares_x / 2) * 8, -8 * 15,
		0, -8 * 15 });

	glGenVertexArrays(1, &buffer_info_lines.vao);
	glGenBuffers(1, &buffer_info_lines.vbo);
	glBindVertexArray(buffer_info_lines.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_info_lines.vbo);
	glBufferData(GL_ARRAY_BUFFER, line_segments.size() * sizeof(line_segments[0]), line_segments.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_INT, GL_FALSE, 2 * sizeof(int), (void*)0);
	glEnableVertexAttribArray(0);

	glUseProgram(shaderProgram);

	auto shader_locations = Shader_locations();

	shader_locations.offset = glGetUniformLocation(shaderProgram, "offset");
	shader_locations.color = glGetUniformLocation(shaderProgram, "uColor");
	shader_locations.projection = glGetUniformLocation(shaderProgram, "uProjection");

	auto scale = 4.0f;

	auto projection = (glm::mat4)glm::ortho(
		(1.0f / scale) * -window_width / 2.0f,
		(1.0f / scale) * window_width / 2.0f,
		(1.0f / scale) * -window_height / 2.0f,
		(1.0f / scale) * window_height / 2.0f
	);

	// Turns our coordinate system into pixel coordinats with window center as origin
	glUniformMatrix4fv(shader_locations.projection, 1, GL_FALSE, glm::value_ptr(projection));

	auto barrels = std::vector<Entity>();
	auto physics_update_rate_s = 1 / 240.0;// 120.0;// 1.0 / 30;
	auto time_last_physics = glfwGetTime();
	auto num_physics_steps = 0;
	auto time_last_fps = glfwGetTime();
	auto num_frames_since_last_update = 0;
	auto pos_previous_x = std::vector<int>(POP_SIZE);
	auto pos_previous_y = std::vector<int>(POP_SIZE);
	auto last_clear_physics_step = 0;

	for (auto idx_player = 0; idx_player < players.size(); idx_player++) {
		auto& player = players[idx_player];
		pos_previous_x[idx_player] = player.offset_x;
		pos_previous_y[idx_player] = player.offset_y;
	}

	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		auto cur_time = glfwGetTime();

		while ((cur_time - time_last_physics) > physics_update_rate_s) {
			game_logics(num_physics_steps, barrels);
			brain_run(window, line_segments, players, barrels, is_human);
			physics(num_physics_steps, line_segments, players, barrels);
			time_last_physics += physics_update_rate_s;
			num_physics_steps++;
		}

		// Kill of long-running agents that does not move
		if (!is_human && (num_physics_steps - last_clear_physics_step > 2000)) {
			auto min_level = num_physics_steps / 2000;
			std::cout << "Killing of agents below level " << min_level << std::endl;

			for (auto& player : players) {
				if (player.level < min_level) {
					player.alive = false;
				}
			}

			last_clear_physics_step = min_level * 2000;
		}

		// Kill players that do not move. Similar to above, more aggressive
		if (!is_human && (num_physics_steps > 0) && (num_physics_steps % 200 == 0)) {
			for (auto idx_player = 0; idx_player < players.size(); idx_player++) {
				auto& player = players[idx_player];
				auto prev_x = pos_previous_x[idx_player];
				auto prev_y = pos_previous_y[idx_player];

				if (std::hypotf((float)player.offset_x - prev_x, (float)player.offset_y - prev_y) < 20.0f) {
					player.alive = false;
				}

				pos_previous_x[idx_player] = player.offset_x;
				pos_previous_y[idx_player] = player.offset_y;
			}
		}

		auto num_alive = 0;

		for (auto& player : players) {
			if (player.alive) {
				num_alive++;
			}
		}

		// Reset
		if (num_alive == 0) {
			brain_update(players);
			init_players(players, is_human, player_width, player_height);
			num_physics_steps = 0;
			barrels.clear();
		}

		// FPS
		auto time_since_last_fps = cur_time - time_last_fps;
		num_frames_since_last_update++;

		if (time_since_last_fps > 1.0) {
			std::cout << "FPS / frame rate: " << num_frames_since_last_update / time_since_last_fps << " / " << 100 * time_since_last_fps / num_frames_since_last_update << "ms" << std::endl;
			time_last_fps = cur_time;
			num_frames_since_last_update = 0;
		}

		render(num_physics_steps, players, barrels, line_segments, shader_locations, buffer_info_background, buffer_info_player, buffer_info_barrel, buffer_info_lines);
		glfwSwapBuffers(window);

		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}