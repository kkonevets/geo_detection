project (sparse)
set(CMAKE_C_STANDARD 11)

find_package(Threads REQUIRED)

add_executable (sparse src/matrix.c src/C-Thread-Pool/thpool.c)
target_link_libraries(sparse ${CMAKE_THREAD_LIBS_INIT})
