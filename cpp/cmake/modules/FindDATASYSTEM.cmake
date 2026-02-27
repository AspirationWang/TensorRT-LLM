#
# SPDX - FileCopyrightText : Copyright ( c )1993-2022 NVIDIA &
# AFFILIATES . All rights reserved . SPDX - License - Identifier : Apache -2.0
# Licensed under the Apache License , Version 2.0( the " License "); you may not 
# use this file except in compliance with the License . You may obtain a copy of 
# the License at 
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing , software 
# & distributed under the License is distributed on an " AS IS " BASIS , WITHOUT 
# WARRANTIES OR CONDITIONS OF ANY KINO , either express or implied . See the 
# License for the specific language governing permissions and limitations under 
# the License .
#

# ================================= 第一步：查找 DataSystem 核心库和头文件 ================================= #
find_package(Python3 REQUIRED)
set(PYTHON_VERSION "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
set(DATASYSTEM_BASE_PATH "/usr/local/lib/python${PYTHON_VERSION}/dist-packages/yr/datasystem")
set(DATASYSTEM_INCLUDE_BASE "${DATASYSTEM_BASE_PATH}/include")
set(DATASYSTEM_LIB_BASE "${DATASYSTEM_BASE_PATH}/lib")

# 查找Datasystem动态库
find_library(DATASYSTEM_LIBRARY 
  NAMES datasystem 
  PATHS ${DATASYSTEM_LIB_BASE}
  NO_DEFAULT_PATH)

if(DATASYSTEM_LIBRARY)
  set(DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${DATASYSTEM_LIBRARY})
endif()

# 查找DataSystem静态库
find_library(DATASYSTEM_STATIC_LIBRARY 
  NAMES datasystem_static 
  PATHS ${DATASYSTEM_LIB_BASE} 
  NO_DEFAULT_PATH)
if(DATASYSTEM_STATIC_LIBRARY)
  set(DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${DATASYSTEM_STATIC_LIBRARY})
endif()

# 查找Datasystem头文件路径
find_path(DATASYSTEM_INCLUDE_DIR 
  NAMES datasystem.h 
  PATHS ${DATASYSTEM_INCLUDE_BASE} 
  PATH_SUFFIXES datasystem 
  NO_DEFAULT_PATH 
)

# ================================= 第二步：查找依赖库﹣ zeroMQ  ================================= #
# 查找ZMQ动态库
find_library(ZMQ_LIBRARY NAMES zmq)
if(ZMQ_LIBRARY)
  set (DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${ZMQ_LIBRARY})
endif()

# 查找ZMQ静态库（可选，根据实际场景调整）
find_library(ZMQ_STATIC_LIBRARY NAMES zmq_static)
if(ZMQ_STATIC_LIBRARY)
  set (DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${ZMQ_STATIC_LIBRARY})
endif()

# 查找ZMQ头文件路径
find_path(ZMQ_INCLUDE_DIR NAMES zmq.h)

# ================================= 第三步：查找依赖库﹣ TBB   ================================= #
# 查找TBB动态库
find_library(TBB_LIBRARY NAMES tbb)
if(TBB_LIBRARY)
  set (DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${TBB_LIBRARY})
endif()

# 查找TBB静态库（可选）
find_library(TBB_STATIC_LIBRARY NAMES tbb_static)
if(TBB_STATIC_LIBRARY)
  set (DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${TBB_STATIC_LIBRARY})
endif()

# 查找TBB头文件路径
find_path(TBB_INCLUDE_DIR NAMES tbb/tbb.h)

# ================================= 第四步：查找依赖库﹣ spdlog   ================================= #
# 查找spdlog动态库
find_library(SPDLOG_LIBRARY NAMES spdlog)
if(SPDLOG_LIBRARY)
set (DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${SPDLOG_LIBRARY})
endif()

# 查找spdlog静态库（可选）
find_library(SPDLOG_STATIC_LIBRARY NAMES spdlog_static)
if(SPDLOG_STATIC_LIBRARY)
  set (DATASYSTEM_LIBRARIES ${DATASYSTEM_LIBRARIES} ${SPDLOG_STATIC_LIBRARY})
endif()

# 查找spdlog 头文件路径
find_path(SPDLOG_INCLUDE_DIR NAMES spdlog/spdlog.h)

# ================================= 验证查找结果 ================================= #
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  DATASYSTEM 
  FOUND_VAR DATASYSTEM_FOUND 
  REQUIRED_VARS DATASYSTEM_LIBRARIES DATASYSTEM_INCLUDE_DIR ZMQ_INCLUDE_DIR TBB_INCLUDE_DIR SPDLOG_INCLUDE_DIR 
  VERSION_VAR DATASYSTEM_VERSION_STRING)
# ================================= 创建导入目标 ================================= #
# 1.DataSystem动态库导入目标
if(DATASYSTEM_LIBRARY)
  add_library (DATASYSTEM::datasystem SHARED IMPORTED)
  # 整合所有依赖头文件
  target_include_directories (DATASYSTEM::datasystem SYSTEM INTERFACE 
    "${DATASYSTEM_INCLUDE_DIR}"
    "${ZMQ_INCLUDE_DIR}"
    "${TBB_INCLUDE_DIR}"
    "${SPDLOG_INCLUDE_DIR}")
  set_property(TARGET DATASYSTEM::datasystem PROPERTY IMPORTED_LOCATION "${DATASYSTEM_LIBRARY}")
  # 链接依赖库
  target_link_libraries(DATASYSTEM::datasystem INTERFACE ${ZMQ_LIBRARY} ${TBB_LIBRARY} ${SPDLOG_LIBRARY})
endif ()

# 2.DataSystem静态库导入目标
if(DATASYSTEM_STATIC_LIBRARY)
  add_library(DATASYSTEM::datasystem_static STATIC IMPORTED)
  target_include_directories(DATASYSTEM::datasystem_static SYSTEM INTERFACE 
    "${DATASYSTEM_INCLUDE_DIR}"
    "${ZMQ_INCLUDE_DIR}"
    "${TBB_INCLUDE_DIR}"
    "${SPDLOG_INCLUDE_DIR}")
  set_property(TARGET DATASYSTEM::datasystem_static PROPERTY IMPORTED_LOCATION "${DATASYSTEM_STATIC_LIBRARY}")
  target_link_libraries(DATASYSTEM::datasystem_static INTERFACE ${ZMQ_STATIC_LIBRARY} ${TBB_STATIC_LIBRARY} ${SPDLOG_STATIC_LIBRARY})
endif()