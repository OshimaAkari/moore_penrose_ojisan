cmake_minimum_required(VERSION 3.0.0)
project(moore_penrose_ojisan VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV 4)
find_package(YAML-CPP)
find_package(dlib)
find_package(BLAS)
find_package(LAPACK)



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS}
                    ${YAML_CPP_INCLUDE_DIRS}
                    ${Dlib_INCLUDE_DIRS}
                    ${BLAS_INCLUDE_DIRS}
                    ${LAPACK_INCLUDE_DIRS})

file(GLOB CPP_SOURCES src/*.cpp)

add_library(${PROJECT_NAME} ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES}
                        ${YAML_CPP_LIBRARIES}
                        ${dlib_LIBRARIES}
                        ${BLAS_LIBRARIES}
                        ${LAPACK_LIBRARIES})


add_executable(real_time_ojisan script/main.cpp)
target_link_libraries(real_time_ojisan ${PROJECT_NAME})

