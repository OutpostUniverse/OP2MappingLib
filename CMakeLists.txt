cmake_minimum_required(VERSION 3.16)
project(op2maplib VERSION 0.3.4 LANGUAGES CXX)

# op2maplib - reusable map-editing algorithms for Outpost 2 map tools, with a
# plain C API (include/op2maplib/op2maplib.h) for use from any language.
#
# Targets:
#   op2maplib        the library (shared by default, see OP2MAPLIB_BUILD_SHARED)
#   op2maplib_tests  C++ tests, run by ctest with profiles/op2.json
#
# The project VERSION also feeds the Windows DLL version resource; keep it in
# step with OM_VERSION_STRING in src/api.cpp.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Executables and DLLs side by side, so the tests find the DLL on Windows.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Static builds define OP2MAPLIB_STATIC for consumers, so OM_API needs no dllimport.
option(OP2MAPLIB_BUILD_SHARED "Build a shared library (DLL/.so/.dylib)" ON)
option(OP2MAPLIB_BUILD_TESTS "Build the tests" ON)

# cliff.cpp is the cliff solver; the other files are described in their headers.
set(OP2MAPLIB_SOURCES
    src/api.cpp
    src/blend.cpp
    src/cliff.cpp
    src/profile.cpp
    src/regions.cpp
    src/json.cpp
    src/mapgen/api_mapgen.cpp
    src/mapgen/field.cpp
    src/mapgen/generate.cpp
    src/mapgen/mountains.cpp
    src/mapgen/net.cpp
    src/mapgen/learned_sketch.cpp
    src/mapgen/learned_tiles.cpp
    src/mapgen/learned_extra.cpp
    src/mapgen/api_learned.cpp
    src/mapgen/paths.cpp
    src/mapgen/relief.cpp
    src/mapgen/shape.cpp
)

if(OP2MAPLIB_BUILD_SHARED)
    add_library(op2maplib SHARED ${OP2MAPLIB_SOURCES})
    # OP2MAPLIB_BUILD makes OM_API export the C functions; everything else stays hidden.
    target_compile_definitions(op2maplib PRIVATE OP2MAPLIB_BUILD)
    set_target_properties(op2maplib PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    if(WIN32)
        # Version details (company, product, version) for the DLL's properties.
        target_sources(op2maplib PRIVATE src/op2maplib.rc)
        target_compile_definitions(op2maplib PRIVATE
            OM_VER_MAJOR=${PROJECT_VERSION_MAJOR}
            OM_VER_MINOR=${PROJECT_VERSION_MINOR}
            OM_VER_PATCH=${PROJECT_VERSION_PATCH})
    endif()
else()
    add_library(op2maplib STATIC ${OP2MAPLIB_SOURCES})
    target_compile_definitions(op2maplib PUBLIC OP2MAPLIB_STATIC)
endif()

target_include_directories(op2maplib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(MSVC)
    target_compile_options(op2maplib PRIVATE /W4 /permissive- /utf-8)
    # Static runtime, so the DLL works without the VC++ redistributable.
    set_property(TARGET op2maplib PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    # No fused multiply-add: it rounds differently, and every platform must give the
    # same maps (the learned generator sums millions of products).
    target_compile_options(op2maplib PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # 32-byte vector types in internal functions (src/mapgen/net.cpp); no ABI is exposed.
        target_compile_options(op2maplib PRIVATE -Wno-psabi)
    endif()
endif()

install(TARGETS op2maplib RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES profiles/op2.json DESTINATION share/op2maplib/profiles)

# The tests use only the public C API, like any other caller.
if(OP2MAPLIB_BUILD_TESTS)
    enable_testing()
    add_executable(op2maplib_tests tests/blend_tests.cpp)
    target_link_libraries(op2maplib_tests PRIVATE op2maplib)
    if(MSVC)
        target_compile_options(op2maplib_tests PRIVATE /W4 /utf-8)
        # Same runtime as the library.
        set_property(TARGET op2maplib_tests PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    add_test(NAME blend COMMAND op2maplib_tests ${CMAKE_CURRENT_SOURCE_DIR}/profiles/op2.json)
endif()
