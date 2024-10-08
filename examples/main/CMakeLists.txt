set(TARGET llama-cli)
add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)



set(TARGET main)
add_library(${TARGET} SHARED main.cpp)

# 安装库文件而非可执行文件
install(TARGETS ${TARGET} LIBRARY DESTINATION lib)

# find_library(LOG_LIB log)

# 链接库
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_CXX_FLAGS "-fPIC")

# 编译特性
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# set(TARGET main)
# # 将可执行文件修改为共享库
# add_library(${TARGET} SHARED main.cpp)

# # 找到 Android log 库
# find_library(LOG_LIB log)

# # 链接库
# target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} ${LOG_LIB})

# # 编译特性
# target_compile_features(${TARGET} PRIVATE cxx_std_11)

# # 安装库文件而非可执行文件
# install(TARGETS ${TARGET} LIBRARY DESTINATION lib)