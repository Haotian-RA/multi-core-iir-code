# Build
cmake_minimum_required(VERSION 3.12)

project(recursive_filter VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Test
include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(vcl
    GIT_REPOSITORY https://github.com/vectorclass/version2.git
    PREFIX external
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

include_directories(${EXTERNAL_INSTALL_LOCATION}/src/vcl
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -I/usr/local/include -march=native -mavx2 -mfma -O3")
add_executable(cascaded_sos test/cascaded_sos.cpp)
add_executable(cascaded_sos_unlimited test/cascaded_sos_unlimited.cpp)
add_executable(filter_test test/filter_test.cpp)
add_executable(single_sos_unlimited test/single_sos_unlimited.cpp)
add_executable(varying_inter_block test/varying_inter_block.cpp)
add_executable(filter example/filter.cpp)

add_test(NAME cascaded_sos COMMAND cascaded_sos)
add_test(NAME cascaded_sos_unlimited COMMAND cascaded_sos_unlimited)
add_test(NAME filter_test COMMAND filter_test)
add_test(NAME single_sos_unlimited COMMAND single_sos_unlimited)
add_test(NAME varying_inter_block COMMAND varying_inter_block)

enable_testing()

# Install
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

if(NOT INCLUDE_INSTALL_DIR)
  set(INCLUDE_INSTALL_DIR include/recursive_filter)
endif()

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  PATH_VARS INCLUDE_INSTALL_DIR)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
