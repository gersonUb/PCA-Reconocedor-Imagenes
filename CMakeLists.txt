cmake_minimum_required(VERSION 3.12)
project(my_module)

set(CMAKE_CXX_STANDARD 17)

# Buscar pybind11
find_package(pybind11 REQUIRED)

# Incluir encabezados estándar de C++
include_directories("/usr/include/pybind11")
include_directories(${pybind11_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories("/usr/include/eigen3/Eigen/")

# Definir el módulo de Python
pybind11_add_module(my_module test.cpp)

target_link_libraries(my_module PRIVATE)
