cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

project(open_spiel_remote_sanity_check VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(main main.cpp)

include_directories(/repo/)
include_directories(/repo/open_spiel/abseil-cpp/)

list(APPEND CMAKE_PREFIX_PATH "/repo/build")
find_library(LIB_OPEN_SPIEL open_spiel)
target_link_libraries(main "${LIB_OPEN_SPIEL}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)