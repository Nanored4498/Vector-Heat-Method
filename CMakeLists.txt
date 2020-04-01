project(vector_heat)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_BUILD_TYPE Release)
set(BASE_CXX_FLAGS "-Wall -Wextra -Werror")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
option(LIBIGL_USE_STATIC_LIBRARY     "Use libIGL as static librarie" ON)
option(LIBIGL_WITH_COMISO            "Use CoMiso"                   OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    OFF)
option(LIBIGL_WITH_PNG               "Use PNG"                      ON)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 OFF)
option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         OFF)
option(LIBIGL_WITH_XML               "Use XML"                      OFF)
find_package(LIBIGL REQUIRED)

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