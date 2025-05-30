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
	//if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) offsetX -= 1.0f;
	//if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) offsetX += 1.0f;
	//if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) offsetY += 1.0f;
	//if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) offsetY -= 1.0f;
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

struct Entity {
	int offset_x;
	int offset_y;
	int width;
	int height;
	int acc_y;
	bool is_on_ground = false;
};

struct Line_segment {
	float x_start;
	float y_start;
	float x_end;
	float y_end;
};

void physics(GLFWwindow* window, std::vector<Line_segment>& line_segments, Entity& player, std::vector<Entity>& barrels) {
	auto max_x = 14 * 8;
	auto max_y = 16 * 8;

	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		player.offset_x = std::clamp(player.offset_x - 1, -max_x, max_x);
	}
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		player.offset_x = std::clamp(player.offset_x + 1, -max_x, max_x);
	}
	//if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
	//	player.offset_y = std::clamp(player.offset_y + 1, -max_y, max_y);
	//}
	//if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
	//	player.offset_y = std::clamp(player.offset_y - 1, -max_y, max_y);
	//}

	if (!player.is_on_ground) {
		auto offset_y_before = player.offset_y - player.height / 2;
		auto offset_y_after = offset_y_before + player.acc_y;
		auto x_left = (float)(player.offset_x - player.width / 2);
		auto x_right = (float)(player.offset_x + player.width / 2);

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

			if ((offset_y_before > line_y) && (offset_y_after <= line_y)) {
				player.is_on_ground = true;
				player.offset_y = line_y + player.height / 2;
			}
		}
		if (!player.is_on_ground) {
			player.offset_y += player.acc_y;
			player.acc_y -= 1;
		}

	}

	if (player.is_on_ground && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		player.acc_y = 7;
		player.is_on_ground = false;
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

	// Vertex setup
	GLuint VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	auto player = Entity();

	player.offset_x = 0;
	player.offset_y = -50;
	player.width = 8;
	player.height = 8;

	std::vector<float> vertices_player = {
		-player.width/2.0f, -player.height / 2.0f,
		-player.width / 2.0f, player.height / 2.0f,
		player.width / 2.0f, player.height / 2.0f,
		player.width / 2.0f, -player.height / 2.0f
	};

	std::vector<int> indices_player = {
		0,1,2,
		2,3,0
	};

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_player), vertices_player.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices_player), indices_player.data(), GL_STATIC_DRAW);
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


	struct Buffer_info {
		GLuint vao;
		GLuint vbo;
		GLuint ebo;
	};

	auto buffer_info_lines = Buffer_info();
	std::vector<Line_segment> line_segments = {};

	line_segments.push_back({
		-3 * 8, 9 * 8,
		3 * 8, 9 * 8 });

	line_segments.push_back({
		-(num_squares_x / 2) * 8.0f, 5 * 8 + 4,
		4 * 8, 5 * 8 + 4 });

	auto generate_line_vertices = [&line_segments](uint32_t num_blocks, int x_offset, int y_offset, int x_factor) {
		for (auto idx_block = 0; idx_block < num_blocks; idx_block++) {
			float pix_x_start = x_offset + 8 * (x_factor * 2 * idx_block);
			float pix_x_end = x_offset + 8 * (x_factor * 2 * (idx_block + 1));
			float pix_y_start = y_offset - idx_block;
			float pix_y_end = pix_y_start;
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
		(-num_squares_x / 2) * 8.0f, -8 * 15,
		0, -8 * 15 });

	glGenVertexArrays(1, &buffer_info_lines.vao);
	glGenBuffers(1, &buffer_info_lines.vbo);
	glBindVertexArray(buffer_info_lines.vao);
	glBindBuffer(GL_ARRAY_BUFFER, buffer_info_lines.vbo);
	glBufferData(GL_ARRAY_BUFFER, line_segments.size() * sizeof(line_segments[0]), line_segments.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glUseProgram(shaderProgram);
	int offsetLoc = glGetUniformLocation(shaderProgram, "offset");
	int colorLoc = glGetUniformLocation(shaderProgram, "uColor");
	int projectionLoc = glGetUniformLocation(shaderProgram, "uProjection");

	auto scale = 4.0f;
	glm::mat4 projection = glm::ortho(
		(1.0f / scale) * -window_width / 2.0f,
		(1.0f / scale) * window_width / 2.0f,
		(1.0f / scale) * -window_height / 2.0f,
		(1.0f / scale) * window_height / 2.0f
	);
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

	auto barrels = std::vector<Entity>();

	auto physics_update_rate_s = 1.0f / 30;
	auto time_last_physics = glfwGetTime();

	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		auto cur_time = glfwGetTime();

		if ((cur_time - time_last_physics) > physics_update_rate_s) {
			physics(window, line_segments, player, barrels);
			time_last_physics = cur_time;
		}

		glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUniform2f(offsetLoc, 0.0f, 0.0f);
		glUniform4f(colorLoc, 0.0f, 0.0f, 0.0f, 1.0f);
		glBindVertexArray(bgVAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glUniform4f(colorLoc, 240.0f / 255, 82.0f / 255, 156.0f / 255, 1.0f);
		glBindVertexArray(buffer_info_lines.vao);
		glDrawArrays(GL_LINES, 0, line_segments.size() * sizeof(line_segments[0]));

		glUniform2f(offsetLoc, player.offset_x, player.offset_y);
		glUniform4f(colorLoc, 0.2f, 0.4f, 1.0f, 1.0f);
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}