#include <algorithm>
#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
};

struct Player : public Entity {
	int score = 0;
};

struct Line_segment {
	int x_start = {};
	int y_start = {};
	int x_end = {};
	int y_end = {};
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

void physics(std::vector<Line_segment>& line_segments, std::vector<Player>& players, std::vector<Entity>& barrels) {
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

void brain_human(GLFWwindow* window, Player& player) {
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

void brain_machine(std::vector<Line_segment>& line_segments, std::vector<Player>& players, std::vector<Entity>& barrels) {
	if (barrels.empty()) {
		return;
	}

	for (auto& player : players) {
		Entity* barrel_closest_above = nullptr;
		auto diff_y_closest_above = std::numeric_limits<int>::max();

		for (auto& barrel : barrels) {
			auto diff_y_current = barrel.offset_y - player.offset_y;

			if ((diff_y_current >= 0) && (diff_y_current < diff_y_closest_above)) {
				diff_y_closest_above = diff_y_current;
				barrel_closest_above = &barrel;
			}
		}

		if (barrel_closest_above) {
			auto is_close_x = (std::abs(barrel_closest_above->offset_x - player.offset_x) < 12);
			auto is_close_y = (std::abs(barrel_closest_above->offset_y - player.offset_y) < 12);

			if (is_close_x && is_close_y) {
				jump(player);
			}
		}
		line_segments;
		//Line_segment* line_segment_closest_better = nullptr;

		//for (auto& line_segment : line_segments) {
		//}
	}
}

void brain(GLFWwindow* window, std::vector<Line_segment>& line_segments, std::vector<Player>& players, std::vector<Entity>& barrels, bool is_human) {
	if (players.empty()) {
		return;
	}

	for (auto& player : players) {
		player.v_x = 0;
	}

	if (is_human) {
		brain_human(window, players[0]);
	}
	else {
		brain_machine(line_segments, players, barrels);
	}
}

void game_logics(int num_physics_steps, std::vector<Entity>& barrels) {
	if (num_physics_steps % 100 == 0) {
		spawn_barrel(barrels);
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
	auto num_agents = 100;
	auto player_width = 8;
	auto player_height = 8;

	auto players = std::vector<Player>();

	auto num_players = is_human ? 1 : num_agents;
	for (auto idx_player = 0; idx_player < num_players; idx_player++) {
		auto player = Player();
		player.offset_x = rand() % 200 - 100;
		player.offset_y = rand() % 100 - 50;
		player.width = player_width;
		player.height = player_height;
		players.push_back(player);
	}

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

	struct Buffer_info {
		GLuint vao;
		GLuint vbo;
		GLuint ebo;
	};

	auto buffer_info_player = Buffer_info();

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

	auto buffer_info_barrel = Buffer_info();

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

	GLuint bgVAO, bgVBO, bgEBO;
	glGenVertexArrays(1, &bgVAO);
	glGenBuffers(1, &bgVBO);
	glGenBuffers(1, &bgEBO);

	glBindVertexArray(bgVAO);
	glBindBuffer(GL_ARRAY_BUFFER, bgVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(bgVertices), bgVertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bgEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(bgIndices), bgIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	auto buffer_info_lines = Buffer_info();
	std::vector<Line_segment> line_segments = {};

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
	int offsetLoc = glGetUniformLocation(shaderProgram, "offset");
	int colorLoc = glGetUniformLocation(shaderProgram, "uColor");
	int projectionLoc = glGetUniformLocation(shaderProgram, "uProjection");

	auto scale = 4.0f;

	auto projection = (glm::mat4)glm::ortho(
		(1.0f / scale) * -window_width / 2.0f,
		(1.0f / scale) * window_width / 2.0f,
		(1.0f / scale) * -window_height / 2.0f,
		(1.0f / scale) * window_height / 2.0f
	);

	// Turns our coordinate system into pixel coordinats with window center as origin
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

	auto barrels = std::vector<Entity>();
	auto physics_update_rate_s = 1 / 120.0;// 1.0 / 30;
	auto time_last_physics = glfwGetTime();
	auto num_physics_steps = 0;
	auto time_last_fps = glfwGetTime();
	auto num_frames_since_last_update = 0;

	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		auto cur_time = glfwGetTime();

		while ((cur_time - time_last_physics) > physics_update_rate_s) {
			game_logics(num_physics_steps, barrels);
			brain(window, line_segments, players, barrels, is_human);
			physics(line_segments, players, barrels);
			time_last_physics += physics_update_rate_s;
			num_physics_steps++;
		}

		// Background
		glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Game background
		glUniform2f(offsetLoc, 0.0f, 0.0f);
		glUniform4f(colorLoc, 0.0f, 0.0f, 0.0f, 1.0f);
		glBindVertexArray(bgVAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// Lines
		glUniform4f(colorLoc, 240.0f / 255, 82.0f / 255, 156.0f / 255, 1.0f);
		glBindVertexArray(buffer_info_lines.vao);
		glDrawArrays(GL_LINES, 0, (GLsizei)line_segments.size() * sizeof(line_segments[0]));

		// Barrels
		for (auto& barrel : barrels) {
			glUniform2f(offsetLoc, (float)barrel.offset_x, (float)barrel.offset_y);
			glUniform4f(colorLoc, 0.7f, 0.4f, 0.4f, 1.0f);
			glBindVertexArray(buffer_info_barrel.vao);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		}

		// Players
		for (auto& player : players) {
			glUniform2f(offsetLoc, (float)player.offset_x, (float)player.offset_y);
			glUniform4f(colorLoc, 0.2f, 0.4f, 1.0f, 1.0f);
			glBindVertexArray(buffer_info_player.vao);
			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		}

		// FPS
		auto time_since_last_fps = cur_time - time_last_fps;
		num_frames_since_last_update++;

		if (time_since_last_fps > 1.0) {
			std::cout << num_frames_since_last_update / time_since_last_fps << std::endl;
			time_last_fps = cur_time;
			num_frames_since_last_update = 0;
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}