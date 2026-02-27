# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
set(SRCS executorWorker.cpp)

# ======================= 新增修改点1：查找datasystem依赖 =======================
# 查找Datasystem（依赖zmq/tbb/spdlog，由FindDatasystem.cmake处理）
find_package(DATASYSTEM REQUIRED)
# ==============================================================================

include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXECUTOR_WORKER_TARGET executorWorker)

add_executable(${EXECUTOR_WORKER_TARGET} ${SRCS})

target_link_libraries(${EXECUTOR_WORKER_TARGET}
                      PUBLIC ${SHARED_TARGET} nvinfer_plugin_tensorrt_llm DATASYSTEM::datasystem)
get_property(DATASYSTEM_LIB_PATH TARGET DATASYSTEM::datasystem PROPERTY IMPORTED_LOCATION)
# 2．打印头文件包含路径
get_property(DATASYSTEM_INC_PATH TARGET DATASYSTEM::datasystem PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# 3．打印它依赖的其他库（之前的libsecurec.so、tbb等）
get_property(DATASYSTEM_LINK_LIBS TARGET DATASYSTEM::datasystem PROPERTY INTERFACE_LINK_LIBRARIES)
target_compile_features(${EXECUTOR_WORKER_TARGET} PRIVATE cxx_std_17)
