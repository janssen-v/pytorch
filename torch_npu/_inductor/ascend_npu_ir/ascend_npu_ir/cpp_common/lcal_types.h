/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef LCAL_TYPES_H
#define LCAL_TYPES_H

#include <map>
#include <string>
#include "hccl_types.h"

namespace Lcal {
constexpr int LCAL_SUCCESS = 0;
constexpr int LCAL_ERROR_NOT_INITIALIZED = -1;
constexpr int LCAL_ERROR_MKIRT = -2;
constexpr int LCAL_ERROR_PARA_CHECK_FAIL = -3;
constexpr int LCAL_ERROR_INTERNAL = -4;
constexpr int LCAL_ERROR_TIMEOUT = -5;
constexpr int LCAL_ERROR_NOT_FOUND = -7;
constexpr int64_t LCAL_INVALID_VALUE = -1;

// shared buffer size，这里要和collectives.cce文件中的常量联动修改！！！
constexpr int LCAL_BUFF_BYTES = 204 * 1024 * 1024;

enum class ChipName {
    CHIP_310P3 = 0,
    CHIP_910B1,
    CHIP_910B2,
    CHIP_910B3,
    CHIP_910B4,
    CHIP_910B41,
    CHIP_910B2C,
    CHIP_910_9391,
    CHIP_910_9381,
    CHIP_910_9392,
    CHIP_910_9382,
    CHIP_910_9372,
    CHIP_910_9361,
    CHIP_910A5,
    RESERVED,
};

enum class PhysicalLink {
    HCCS = 0,
    PCIE = 1,
    RESERVED,
};

// 包含 物理链路、芯片名称 信息。
struct PhysicalInfo {
    ChipName chipName = ChipName::RESERVED;
    PhysicalLink physicalLink = PhysicalLink::RESERVED;
    uint32_t coreNum = 0;
};

enum class LcalType {
    ALL_REDUCE = 1,
    REDUCE_SCATTER = 2,
    ALL_GATHER = 3,
    BROADCAST = 4,
    ALL2ALL = 5,
    ALL2ALL_V_C = 6,
    GATHER = 7,
    LOCAL_REDUCE = 8,
    SEND = 9,
    RECV = 10,
    PURE_MATMUL = 101,
    MATMUL_ALL_REDUCE = 102,
    MATMUL_REDUCE_SCATTER = 103,
    ALL_GATHER_MATMUL = 104,
    ALL_GATHER_MATMUL_V2 = 105,
    ALL2ALL_MATMUL = 106,
    MATMUL_ALL2ALL = 107,
    MTE2_TEST = 108,
    ALL_GATHER_MATMUL_REDUCE_SCATTER = 111,
    LCAL_TYPE_MAX = 310,
    BANDWIDTH = 201,

    ALLTOALLV_ALLGATHER_MATMUL = 305,
    MATMUL_REDUCESCATTER_ALLTOALLV = 306,
    ALLTOALLVC_ALLGATHER_MATMUL = 307,
    MATMUL_REDUCESCATTER_ALLTOALLVC = 308
};

const std::map<LcalType, std::string> LCAL_TYPE2NAME = {
    { LcalType::ALL_REDUCE, "LcalAllReduce" },
    { LcalType::REDUCE_SCATTER, "LcalReduceScatter" },
    { LcalType::ALL_GATHER, "LcalAllGather" },
    { LcalType::BROADCAST, "LcalBroadcast" },
    { LcalType::PURE_MATMUL, "LcalPureMatmul" },
    { LcalType::MATMUL_ALL_REDUCE, "LcalMatmulAllReduce" },
    { LcalType::MATMUL_REDUCE_SCATTER, "LcalMatmulReduceScatter" },
    { LcalType::ALL_GATHER_MATMUL, "LcalAllGatherMatmul" },
    { LcalType::ALL_GATHER_MATMUL_V2, "LcalAllGatherMatmulV2" },
    { LcalType::ALL2ALL_MATMUL, "LcalAll2AllMatmul" },
    { LcalType::MATMUL_ALL2ALL, "LcalMatmulAll2All" },
    { LcalType::MTE2_TEST, "LcalMTE2Test" },
    { LcalType::ALL2ALL, "LcalAll2All" },
    { LcalType::ALL2ALL_V_C, "LcalAll2AllVC" },
    { LcalType::ALL_GATHER_MATMUL_REDUCE_SCATTER, "LcalAllGatherMatmulReduceScatter" },
    { LcalType::BANDWIDTH, "LcalBandwidthTest" },
    { LcalType::LOCAL_REDUCE, "LcalLocalReduce" },
    { LcalType::GATHER, "LcalGather" },
    { LcalType::SEND, "LcalSend" },
    { LcalType::RECV, "LcalRecv" },
    { LcalType::ALLTOALLV_ALLGATHER_MATMUL, "LcalAllToAllVAllGatherMatmul" },
    { LcalType::MATMUL_REDUCESCATTER_ALLTOALLV, "LcalMatmulReduceScatterAllToAllV" },

    { LcalType::ALLTOALLVC_ALLGATHER_MATMUL, "LcalAllToAllVAllGatherMatmul" },
    { LcalType::MATMUL_REDUCESCATTER_ALLTOALLVC, "LcalMatmulReduceScatterAllToAllV" }
};


} // namespace Lcal
#endif // LCAL_TYPES_H
