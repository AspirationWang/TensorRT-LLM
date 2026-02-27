# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: NVIDIA TensorRT
# Source Code License Agreement
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
set(EXECUTOR_TARGET_NAME tensorrt_llm_executor)
set(EXECUTOR_STATIC_TARGET ${EXECUTOR_TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# keep this list sorted alphabetically
set(SRCS
    cache_transmission/mpi_utils/connection.cpp
    cache_transmission/agent_utils/connection.cpp
    cache_transmission/transferAgent.cpp
    cache_transmission/cacheSplitConcat.cu
    contextPhaseParams.cpp
    debugConfig.cpp
    decodingConfig.cpp
    executor.cpp
    executorConfig.cpp
    executorImpl.cpp
    executorKVCacheEventManager.cpp
    extendedRuntimePerfKnobConfig.cpp
    guidedDecodingConfig.cpp
    guidedDecodingParams.cpp
    jsonSerialization.cpp
    kvCacheConfig.cpp
    kvCacheRetentionConfig.cpp
    logitsPostProcessorConfig.cpp
    loraConfig.cpp
    orchestratorConfig.cpp
    outputConfig.cpp
    parallelConfig.cpp
    peftCacheConfig.cpp
    promptTuningConfig.cpp
    mropeConfig.cpp
    multimodalInput.cpp
    request.cpp
    requestUtils.cpp
    requestWithId.cpp
    response.cpp
    samplingConfig.cpp
    dynamicBatchConfig.cpp
    dynamicBatchTuner.cpp
    schedulerConfig.cpp
    serialization.cpp
    speculativeDecodingConfig.cpp
    tensor.cpp
    types.cpp
    requestUtils.cpp
    contextPhaseParams.cpp
    disaggServerUtil.cpp
    cacheTransceiverConfig.cpp)

if(NOT WIN32)
  # additional warnings
  #
  # Ignore overloaded-virtual warning. We intentionally change parameters of
  # some methods in derived class.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  if(WARNING_IS_ERROR)
    message(STATUS "Treating warnings as errors in GCC compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
else() # Windows
  # warning level 4
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# ======================= 新增修改点1：查找datasystem依赖 =======================
# 查找Datasystem（依赖zmq/tbb/spdlog，由FindDatasystem.cmake处理）
find_package(DATASYSTEM REQUIRED)
# ==============================================================================

add_library(${EXECUTOR_STATIC_TARGET} STATIC ${SRCS})
target_include_directories(
  ${EXECUTOR_STATIC_TARGET}
  PUBLIC ${DATASYSTEM_INCLUDE_DIR}/../
)
set_target_properties(
  ${EXECUTOR_STATIC_TARGET}
  PROPERTIES CXX_STANDARD "17" CXX_STANDARD_REQUIRED "YES" CXX_EXTENSIONS "NO"
             POSITION_INDEPENDENT_CODE ON)

set_property(TARGET ${EXECUTOR_STATIC_TARGET}
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set(TOP_LEVEL_DIR "${PROJECT_SOURCE_DIR}/..")
target_compile_definitions(${EXECUTOR_STATIC_TARGET}
                           PUBLIC TOP_LEVEL_DIR="${TOP_LEVEL_DIR}")

add_subdirectory(cache_transmission/ucx_utils)
add_subdirectory(cache_transmission/nixl_utils)

# ========================== 新增修改点3：链接DataSystem/库 ==========================
# 链接DataSystem核心库（自动包含zmq/tbb/spdlog依赖）
target_link_libraries(${BATCH_MANAGER_STATIC_TARGET}
  PUBLIC
    DATASYSTEM::datasystem # DataSystem导入目标（包含所有依赖）
)
get_property(DATASYSTEM_LIB_PATH TARGET DATASYSTEM::datasystem PROPERTY IMPORTED_LOCATION)
# 2．打印头文件包含路径
get_property(DATASYSTEM_INC_PATH TARGET DATASYSTEM::datasystem PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# 3．打印它依赖的其他库（之前的libsecurec.so、tbb等）
get_property(DATASYSTEM_LINK_LIBS TARGET DATASYSTEM::datasystem PROPERTY INTERFACE_LINK_LIBRARIES)
# ===================================================================================