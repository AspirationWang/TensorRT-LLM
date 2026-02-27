set(TRTLLM_NB_MODULE bindings)
set(TRTLLM_NB_MODULE
    ${TRTLLM_NB_MODULE}
    PARENT_SCOPE)

set(SRCS
    batch_manager/algorithms.cpp
    batch_manager/bindings.cpp
    batch_manager/cacheTransceiver.cpp
    batch_manager/kvCacheManager.cpp
    batch_manager/llmRequest.cpp
    executor/bindings.cpp
    executor/executor.cpp
    executor/executorConfig.cpp
    executor/request.cpp
    runtime/bindings.cpp
    testing/modelSpecBinding.cpp
    runtime/moeBindings.cpp
    userbuffers/bindings.cpp
    ../runtime/ipcNvlsMemory.cu
    bindings.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

nanobind_add_module(${TRTLLM_NB_MODULE} ${SRCS})

set_property(TARGET ${TRTLLM_NB_MODULE} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_directories(${TRTLLM_NB_MODULE} PUBLIC
                        "${TORCH_INSTALL_PREFIX}/lib")

if(ENABLE_NVSHMEM)
  target_link_libraries(${TRTLLM_NB_MODULE} PUBLIC nvshmem::nvshmem_host
                                                   nvshmem::nvshmem_device)
endif()

# ======================= 新增修改点1：查找datasystem依赖 =======================
# 查找Datasystem（依赖zmq/tbb/spdlog，由FindDatasystem.cmake处理）
find_package(DATASYSTEM REQUIRED)
# ==============================================================================

target_include_directories(
  ${TRTLLM_NB_MODULE}
  # =========================== 新增修改点2:添加Datasystem头文件路径 ===========================
  ${DATASYSTEM_INCLUDE_DIR}/../
  # ===========================================================================================
)

target_link_libraries(
  ${TRTLLM_NB_MODULE}
  PUBLIC ${SHARED_TARGET}
         ${UNDEFINED_FLAG}
         ${NO_AS_NEEDED_FLAG}
         ${Python3_LIBRARIES}
         ${TORCH_LIBRARIES}
         torch_python
         ${CUDA_NVML_LIB}
         DATASYSTEM::datasystem)
get_property(DATASYSTEM_LIB_PATH TARGET DATASYSTEM::datasystem PROPERTY IMPORTED_LOCATION)
# 2．打印头文件包含路径
get_property(DATASYSTEM_INC_PATH TARGET DATASYSTEM::datasystem PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# 3．打印它依赖的其他库（之前的libsecurec.so、tbb等）
get_property(DATASYSTEM_LINK_LIBS TARGET DATASYSTEM::datasystem PROPERTY INTERFACE_LINK_LIBRARIES)
target_compile_definitions(
  ${TRTLLM_NB_MODULE} PUBLIC TRTLLM_NB_MODULE=${TRTLLM_NB_MODULE}
                             PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(NOT WIN32)
  set_target_properties(
    ${TRTLLM_NB_MODULE}
    PROPERTIES
      LINK_FLAGS
      "-Wl,-rpath,'$ORIGIN/libs' -Wl,-rpath,'$ORIGIN/../nvidia/nccl/lib' -Wl,-rpath,'${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/stubs' ${AS_NEEDED_FLAG} ${UNDEFINED_FLAG}"
  )
endif()
