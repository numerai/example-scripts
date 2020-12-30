cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(numerai)

set(CMAKE_CXX_STANDARD 17)

# Torch.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(numerai example.cc)
# target_link_libraries(numerai "${TORCH_LIBRARIES}" csv)
target_link_libraries(numerai "${TORCH_LIBRARIES}")
set_property(TARGET numerai PROPERTY CXX_STANDARD 17)

# Copy DLLs to avoid memory errors on Windows. https://github.com/pytorch/pytorch/issues/25457
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
