/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
*/

#pragma once

namespace ckernel {
enum ReduceDim {
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};

enum TileDim {
    R_IDX = 0,
    C_IDX = 1,
};

enum PoolType {
    SUM,
    AVG,
    MAX,
};

enum DataCopyType {
    A2D,
    B2D,
};

enum EltwiseBinaryType {
    ELWMUL,
    ELWDIV,
    ELWADD,
    ELWSUB,
    ELWLESS,
};

enum class EltwiseBinaryReuseDestType {
    NONE = 0,
    DEST_TO_SRCA = 1,
    DEST_TO_SRCB = 2,
};

enum DstSync {
    SyncHalf = 0,
    SyncFull = 1,
    SyncTile16 = 2,
    SyncTile2 = 3,
};

enum BroadcastType {
    NONE = 0x0,    // A - None || B - None
    COL = 0x1,     // A - None || B - Col Broadcast
    ROW = 0x2,     // A - None || B - Row Broadcast
    SCALAR = 0x3,  // A - None || B - Scalar Broadcast
};

enum src_op_id_e {
    OP_SRC0 = 0,
    OP_SRC1 = 1,
    OP_SRC2 = 2,
    OP_SRC3 = 3,
    OP_SRC4 = 4,
};

enum local_op_id_e {
    OP_LOCAL0 = 0,
    OP_LOCAL1 = 1,
    OP_LOCAL2 = 2,
    OP_LOCAL3 = 3,
    OP_LOCAL4 = 4,
};

enum out_op_id_e {
    OUT_ID0 = 0,
    OUT_ID1 = 1,
    OUT_ID2 = 2,
    OUT_ID3 = 3,
    OUT_ID4 = 4,
};

enum ReluType {
    NO_RELU,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};

}  // namespace ckernel
