/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>

#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"

#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/kvCachePartialCopy.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#ifdef ENABLE_CUFILE
#include <cufile.h>
#endif
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <vector>
#include <time.h>

namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

static bool gpuToFilePosix(tr::ITensor::SharedPtr const& srcPtr, std::string const& filename)
{
    int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY, 0664);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for writing (POSIX fallback)", filename.c_str());

    ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
    std::vector<uint8_t> hostBuffer(numBytes);

    cudaError_t cpyErr = cudaMemcpy(hostBuffer.data(), srcPtr->data(), numBytes, cudaMemcpyDeviceToHost);
    TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to host failed, error=%d", cpyErr);

    ssize_t written = ::write(fd, hostBuffer.data(), numBytes);
    TLLM_CHECK_WITH_INFO(written >= 0, "POSIX write error=%zd", written);

    TLLM_LOG_DEBUG("Wrote %zd bytes to %s (POSIX fallback)", written, filename.c_str());

    ::close(fd);
    return true;
}

static bool fileToGpuPosix(tr::ITensor::SharedPtr const& dstPtr, std::string const& filename)
{
    int fd = ::open(filename.c_str(), O_RDONLY);
    TLLM_CHECK_WITH_INFO(fd >= 0, "Failed to open '%s' for reading (POSIX fallback)", filename.c_str());

    ssize_t numBytes = static_cast<ssize_t>(dstPtr->getSizeInBytes());
    std::vector<uint8_t> hostBuffer(numBytes);

    ssize_t bytesRead = ::read(fd, hostBuffer.data(), numBytes);
    TLLM_CHECK_WITH_INFO(bytesRead >= 0, "POSIX read error=%zd", bytesRead);

    TLLM_LOG_DEBUG("Read %zd bytes from %s (POSIX fallback)", bytesRead, filename.c_str());

    cudaError_t cpyErr = cudaMemcpy(dstPtr->data(), hostBuffer.data(), numBytes, cudaMemcpyHostToDevice);
    TLLM_CHECK_WITH_INFO(cpyErr == cudaSuccess, "cudaMemcpy to device failed, error=%d", cpyErr);

    ::close(fd);
    return true;
}

KVCacheTransferManager::KVCacheTransferManager(tr::BufferManager const& bufferManager)
    : mBufferManager{bufferManager}
    , mOnboardManager(std::make_shared<tr::CudaStream>())
    , mOffloadManager(std::make_shared<tr::CudaStream>())
{
}

tr::ITensor::SharedPtr KVCacheTransferManager::computeBlockPointer(
    BlockPtr const& block, std::vector<KVCacheBlockPool> const& pools, size_t poolIdx)
{
    TLLM_CHECK_WITH_INFO(!pools.empty(), "Pool index %lu is out of bounds", poolIdx);
    auto const& pool = pools.at(poolIdx);
    auto ptr = block->isPrimary() ? pool.primaryPtr : pool.secondaryPtr;
    if (ptr == nullptr) {
        /* 开启datasystem后，获取到的指针可能为空指针，如果不返回，下方会访问空指针报错 */
        return nullptr;
    }
    auto const blockOffset = block->getMemoryPoolBlockIndex();
    tr::ITensor::SharedPtr blockTensor{tr::ITensor::slice(ptr, blockOffset, 1)};
    return blockTensor;
}

void KVCacheTransferManager::copyBlock(BlockPtr const& src, BlockPtr const& dst,
    std::vector<KVCacheBlockPool> const& pools, bool isOffload, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::optional<std::string> directory)
{
    TLLM_LOG_DEBUG("copyBlock entered: srcId=%d, dstId=%d, isOffload=%s, mode=%d", src->getBlockId(), dst->getBlockId(),
        (isOffload ? "true" : "false"), static_cast<int>(mode));

    if (mode == executor::KVCacheTransferMode::DRAM)
    {
        TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] mode = %d: pools.size() = %u, numTokensToCopy = %d.",
            pools.size(), numTokensToCopy);
        TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] Src Key = %s.", std::to_string(BlockKeyHasher::hash(src->getBlockKey())).c_str());
        TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] Dst Key = %s.", std::to_string(BlockKeyHasher::hash(dst->getBlockKey())).c_str());
        TLLM_LOG_DEBUG("Using DRAM-based copy (GPU <-> CPU) for this block.");
        // Iterate over all pools, partial-copy logic
        for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
        {
            auto srcPtr = computeBlockPointer(src, pools, poolIdx);
            auto dstPtr = computeBlockPointer(dst, pools, poolIdx);
            if (srcPtr == nullptr) {
                if (dstPtr == nullptr) {
                    TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] srcPtr is empty ptr, dstPtr is empty ptr");
                } else {
                    TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] srcPtr is empty ptr, dstPtr is not empty ptr");
                }
            } else {
                if (dstPtr == nullptr) {
                    TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] srcPtr is not empty ptr, dstPtr is empty ptr");
                } else {
                    TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] srcPtr is not empty ptr, dstPtr is not empty ptr");
                }
            }

            KvCacheManagerDataSystem& dataSystem = KvCacheManagerDataSystem::getInstance();
            if (!dataSystem.isKVClientInitialized())
            {
                TLLM_LOG_ERROR("[TensorRT-LLM][Datasystem] KvCache Client is not initialized");
                return;
            }
            std::shared_ptr<datasystem::KVClient> kvClient = dataSystem.getKVClient();

            KvCacheManagerDataSystemTmp& dataSystem1 = KvCacheManagerDataSystemTmp::getInstance();
            if (!dataSystem1.isKVClientInitialized())
            {
                TLLM_LOG_ERROR("[TensorRT-LLM][Datasystem] KvCache Client TMP is not initialized");
                return;
            }
            std::shared_ptr<datasystem::KVClient> kvClient1 = dataSystem1.getKVClient();

            // If no partial tokens or if the dataType is not supported for partial copy, copy entire block.
            if (isOffload) {
                /* 先create再set */
                std::shared_ptr<datasystem::Buffer> buffer;
                datasystem::SetParam para;
                para.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
                datasystem::Status createRet = kvClient->Create(std::to_string(BlockKeyHasher::hash(src->getBlockKey())), srcPtr->getSizeInBytes(), para, buffer);
                TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] Create Key = %s.", std::to_string(BlockKeyHasher::hash(src->getBlockKey())).c_str());
                if (createRet.IsError()) {
                    TLLM_LOG_ERROR("[TensorRT-LLM][Datasystem] Create KvCache failed, detail : %s", createRet.ToString().c_str());
                    return;
                }
            
                mOffloadManager.offloadCopy(*srcPtr, buffer->MutableData());
                /* set的地址为buffer */
                buffer->MLatch();
                datasystem::Status setRet = kvClient->Set(buffer);
                buffer->MUnlatch();
                if (setRet.IsError()) {
                    TLLM_LOG_ERROR("[TensorRT-LLM][Datasystem] Set KvCache failed, detail : %s", setRet.ToString().c_str());
                    return;
                }
                TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] Set Key = %s success.", std::to_string(BlockKeyHasher::hash(src->getBlockKey())).c_str());
            } else {
                /* 如果在HBM中，则直接使用，不需要查datasystem */
                if (src->isPrimary()) {
                    TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] Kvcache in HBM, Key = %s.", std::to_string(BlockKeyHasher::hash(src->getBlockKey())).c_str());
                    mOnboardManager.copy(*srcPtr, *dstPtr);
                } else {
                    /* 如果不在HBM中，前面已经判断了datasystem中有数据，直接get完成后通过cuda接口加载到HBM中 */
                    // struct timespec start, end;
                    // clock_gettime(CLOCK_MONOTONIC, &start);
                    datasystem::Optional<datasystem::Buffer> buffer;
                    datasystem::Status getRet = kvClient1->Get(std::to_string(BlockKeyHasher::hash(src->getBlockKey())), buffer, 0);
                    TLLM_LOG_INFO("[TensorRT-LLM][Datasystem] Get Key = %s.", std::to_string(BlockKeyHasher::hash(src->getBlockKey())).c_str());
                    if (getRet.IsError()) {
                        TLLM_LOG_ERROR("[TensorRT-LLM][Datasystem] Get KvCache failed, detail : %s", getRet.ToString().c_str());
                        return;
                    }
            
                    TLLM_LOG_DEBUG("[TensorRT-LLM][Datasystem] Get KvCache Success");
                    buffer->RLatch();
                    mOnboardManager.onBoardCopy(*dstPtr, buffer->MutableData(), buffer->GetSize());
                    buffer->RUnlatch();
                    // clock_gettime(CLOCK_MONOTONIC, &end);
                    // long long duration_ns = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
                    // double duration_ms = duration_ns / 1e6;
                    // TLLM_LOG_INFO("[TensorRT-LLM][Datasystem] Get Key time = %lf.", duration_ms);
                }
            }
        }
    }

    for (size_t poolIdx = 0; poolIdx < pools.size(); ++poolIdx)
    {
        auto srcPtr = computeBlockPointer(src, pools, poolIdx);
        auto dstPtr = computeBlockPointer(dst, pools, poolIdx);

        TLLM_CHECK_WITH_INFO(
            directory.has_value(), "Expected a directory path for KVCache offload, but none was provided.");

        int size = std::snprintf(
            nullptr, 0, "%s/block_%d_pool_%zu.bin", directory.value().c_str(), src->getBlockId(), poolIdx);

        std::string filename(size + 1, '\0');
        std::snprintf(filename.data(), filename.size(), "%s/block_%d_pool_%zu.bin", directory.value().c_str(),
            src->getBlockId(), poolIdx);

        if (mode == executor::KvCacheTransferMode::POSIX_DEBUG_FALLBACK)
        {
            TLLM_LOG_INFO("Forcing POSIX fallback for file: %s", filename.c_str());
            if (isOffload)
            {
                gpuToFilePosix(srcPtr, filename);
            }
            else
            {
                fileToGpuPosix(dstPtr, filename);
            }
            continue;
        }

        int openFlags = isOffload ? (O_CREAT | O_WRONLY) : O_RDONLY;
        int fd = ::open(filename.c_str(), openFlags, 0664);
        if (fd < 0)
        {
            TLLM_LOG_ERROR(
                "Failed to open '%s' for %s; fallback POSIX", filename.c_str(), (isOffload ? "writing" : "reading"));

            if (isOffload)
            {
                gpuToFilePosix(srcPtr, filename);
            }
            else
            {
                fileToGpuPosix(dstPtr, filename);
            }
            continue;
        }

#ifdef ENABLE_CUFILE
        CUfileDescr_t cufileDesc = {};
        cufileDesc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
        cufileDesc.handle.fd = fd;

        CUfileHandle_t cufileHandle;
        CUfileError_t status = cuFileHandleRegister(&cufileHandle, &cufileDesc);
        if (status.err != CU_FILE_SUCCESS)
        {
            // Fallback to POSIX
            TLLM_LOG_WARN(
                "cuFileHandleRegister failed (err=%d). Falling back to POSIX for '%s'", status.err, filename.c_str());
            ::close(fd);
            if (isOffload)
                gpuToFilePosix(srcPtr, filename);
            else
                fileToGpuPosix(dstPtr, filename);
            continue;
        }

        ssize_t numBytes = static_cast<ssize_t>(srcPtr->getSizeInBytes());
        if (isOffload)
        {
            ssize_t written = cuFileWrite(cufileHandle, srcPtr->data(), numBytes, 0, 0);
            if (written < 0)
            {
                TLLM_LOG_ERROR("cuFileWrite error=%zd. Fallback to POSIX", written);
                cuFileHandleDeregister(cufileHandle);
                ::close(fd);
                gpuToFilePosix(srcPtr, filename);
                continue;
            }
        }
        else
        {
            ssize_t readCount = cuFileRead(cufileHandle, dstPtr->data(), numBytes, 0, 0);
            if (readCount < 0)
            {
                TLLM_LOG_ERROR("cuFileRead error=%zd. Fallback to POSIX", readCount);
                cuFileHandleDeregister(cufileHandle);
                ::close(fd);
                fileToGpuPosix(dstPtr, filename);
                continue;
            }
        }

        cuFileHandleDeregister(cufileHandle);
        ::close(fd);
#else
        // If GDS isn't enabled, fallback to POSIX automatically
        TLLM_LOG_DEBUG("ENABLE_CUFILE=OFF, so fallback to POSIX for %s", filename.c_str());
        ::close(fd); // close the file opened for GDS
        if (isOffload)
        {
            gpuToFilePosix(srcPtr, filename);
        }
        else
        {
            fileToGpuPosix(dstPtr, filename);
        }
#endif
    }
}

void KVCacheTransferManager::onboard(BlockPtr const& offloadBlock, BlockPtr const& block,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::optional<std::string> directory)
{
    if (mode != executor::KvCacheTransferMode::DRAM
        && mPendingOffloads.find(offloadBlock->getBlockId()) == mPendingOffloads.end())
    {
        TLLM_LOG_DEBUG("Skipping onboard for block %d because it was never previously offloaded to disk",
            offloadBlock->getBlockId());
        return;
    }

    if (mPendingOffloads.find(offloadBlock->getBlockId()) != mPendingOffloads.end())
    {
        mOnboardManager.getStream().wait(mPendingOffloads[offloadBlock->getBlockId()]);
    }
    copyBlock(offloadBlock, block, pools, false, numTokensToCopy, mode, directory);
}

void KVCacheTransferManager::offload(BlockPtr const& block, BlockPtr const& offloadBlock,
    std::vector<KVCacheBlockPool> const& pools, int numTokensToCopy, executor::KvCacheTransferMode mode,
    std::optional<std::string> directory)
{
    mPendingOffloads[block->getBlockId()] = tr::CudaEvent();
    copyBlock(block, offloadBlock, pools, true, numTokensToCopy, mode, directory);
    mOffloadManager.getStream().record(mPendingOffloads[block->getBlockId()]);
}

void KVCacheTransferManager::syncTransfers()
{
    tr::CudaEvent offloadEvent;
    mOffloadManager.getStream().record(offloadEvent);

    tr::CudaEvent onboardEvent;
    mOnboardManager.getStream().record(onboardEvent);

    mBufferManager.getStream().wait(offloadEvent);
    mBufferManager.getStream().wait(onboardEvent);

    // Once we synchronize, clear our list of pending thransfers.
    mPendingOffloads.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
