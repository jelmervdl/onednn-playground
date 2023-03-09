set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(moe moe.cpp)

target_link_libraries(moe oneDNN::lib cnpy)

find_package(OpenMP)
target_link_libraries(moe OpenMP::OpenMP_CXX)

