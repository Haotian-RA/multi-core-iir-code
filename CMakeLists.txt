cmake_minimum_required(VERSION 3.12)

project(recursive_filter VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(ExternalProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
set(TBB_INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}/oneTBB/src/oneTBB)
set(TBB_LIBRARY_RELEASE ${TBB_INSTALL_DIR}/build/linux_intel64_gcc_cc9_libc2.31_kernel5.4.0_release)

# Add vcl external project
ExternalProject_Add(vcl
    GIT_REPOSITORY https://github.com/vectorclass/version2.git
    PREFIX ${EXTERNAL_INSTALL_LOCATION}/vcl
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
)

# Add an old release of oneTBB
ExternalProject_Add(oneTBB
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    PREFIX ${EXTERNAL_INSTALL_LOCATION}/oneTBB
    GIT_TAG 2019  # Tag for the 2019 release
    CONFIGURE_COMMAND cd ${TBB_INSTALL_DIR} && cmake .
    BUILD_COMMAND cd ${TBB_INSTALL_DIR} && make
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
)

# Ensure that external projects are built before using them
ExternalProject_Get_Property(oneTBB install_dir)
add_library(TBB INTERFACE IMPORTED)
set_target_properties(TBB PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TBB_INSTALL_DIR}/include"
    INTERFACE_LINK_LIBRARIES "-Wl,-rpath,${TBB_LIBRARY_RELEASE} -L${TBB_LIBRARY_RELEASE} -ltbb"
)

ExternalProject_Get_Property(vcl install_dir)
add_library(VCL INTERFACE IMPORTED)
set_target_properties(VCL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EXTERNAL_INSTALL_LOCATION}/vcl/src"
)

# Include directories for the projects
include_directories(${EXTERNAL_INSTALL_LOCATION}/vcl/src
    ${TBB_INSTALL_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -I/usr/local/include -march=native -mavx2 -mfma -O3")

# Add executables and link libraries
add_executable(cascaded_sos test/cascaded_sos.cpp)
add_dependencies(cascaded_sos oneTBB vcl)
target_link_libraries(cascaded_sos TBB VCL)

add_executable(cascaded_sos_unlimited test/cascaded_sos_unlimited.cpp)
add_dependencies(cascaded_sos_unlimited oneTBB vcl)
target_link_libraries(cascaded_sos_unlimited TBB VCL)

add_executable(filter_test test/filter_test.cpp)
add_dependencies(filter_test oneTBB vcl)
target_link_libraries(filter_test TBB VCL)

add_executable(single_sos_unlimited test/single_sos_unlimited.cpp)
add_dependencies(single_sos_unlimited oneTBB vcl)
target_link_libraries(single_sos_unlimited TBB VCL)

add_executable(varying_inter_block test/varying_inter_block.cpp)
add_dependencies(varying_inter_block oneTBB vcl)
target_link_libraries(varying_inter_block TBB VCL)

add_executable(filter example/filter.cpp)
add_dependencies(filter oneTBB vcl)
target_link_libraries(filter TBB VCL)

# Enable testing
enable_testing()
add_test(NAME cascaded_sos COMMAND cascaded_sos)
add_test(NAME cascaded_sos_unlimited COMMAND cascaded_sos_unlimited)
add_test(NAME filter_test COMMAND filter_test)
add_test(NAME single_sos_unlimited COMMAND single_sos_unlimited)
add_test(NAME varying_inter_block COMMAND varying_inter_block)

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
