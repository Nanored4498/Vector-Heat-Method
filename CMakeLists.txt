project(vector_heat)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)
set(BASE_CXX_FLAGS "-Wall -Wextra -Werror")

add_subdirectory("/home/yoann/libigl" "/home/yoann/libigl/build" EXCLUDE_FROM_ALL)

set(SRC
	src/main.cpp
	src/heat_vector.cpp
	src/my_draw_buffer.cpp
)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME}
	igl::opengl_glfw
	igl::png
)