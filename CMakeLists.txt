cmake_minimum_required(VERSION 3.16)
project(donkey)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(donkey main.cpp)

target_include_directories(donkey PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(donkey PRIVATE
    glad::glad    
    glfw
    glm::glm
)