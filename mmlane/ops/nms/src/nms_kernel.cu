#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Hard-coded maximum. Increase if needed.
#define MAX_COL_BLOCKS 1000
#define DATASET_OFFSET 0
#define N_OFFSETS 72 // if you use more than 73 offsets you will have to adjust this value
#define N_STRIPS (N_OFFSETS - 1)
#define PROP_SIZE (3 + N_OFFSETS)

#define DIVUP(m,n) (((m)+(n)-1) / (n))
int64_t const threadsPerBlock = sizeof(unsigned long long) * 8;

// The functions below originates from Fast R-CNN
// See https://github.com/rbgirshick/py-faster-rcnn
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Written by Shaoqing Ren

template <typename scalar_t>
// __device__ inline scalar_t devIoU(scalar_t const * const a, scalar_t const * const b) {
__device__ inline bool devIoU(scalar_t const * const a, scalar_t const * const b, const float threshold) {
  const int start_a = (int) (a[0] * N_STRIPS - DATASET_OFFSET + 0.5); // 0.5 rounding trick
  const int start_b = (int) (b[0] * N_STRIPS - DATASET_OFFSET + 0.5);
  const int start = max(start_a, start_b);
  const int end_a = start_a + a[2] - 1 + 0.5 - ((a[2] - 1) < 0); //  - (x<0) trick to adjust for negative numbers (in case length is 0)
  const int end_b = start_b + b[2] - 1 + 0.5 - ((b[2] - 1) < 0);
  const int end = min(min(end_a, end_b), N_STRIPS);
  // if (end < start) return 1e9;
  if (end < start) return false;
  scalar_t dist = 0;
  for(unsigned char i = 3 + start; i <= 3 + end; ++i) {
    if (a[i] < b[i]) {
      dist += b[i] - a[i];
    } else {
      dist += a[i] - b[i];
    }
  }
  // return (dist / (end - start + 1)) < threshold;
  return dist < (threshold * (end - start + 1));
  // return dist / (end - start + 1);
}

template <typename scalar_t>
__global__ void nms_kernel(const int64_t n_predictions, const scalar_t nms_overlap_thresh,
                           const scalar_t *dev_predictions, const int64_t *idx, int64_t *dev_mask) {
  const int64_t row_start = blockIdx.y;
  const int64_t col_start = blockIdx.x;

  if (row_start > col_start) return;    // 只计算上三角的iou矩阵

  // 最大为threadsPerBlock，因为n_predictions可能不能被threadsPerBlock整除，获得余数.
  const int row_size =
        min(n_predictions - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_predictions - col_start * threadsPerBlock, threadsPerBlock);

  // 共享内存，把同一线程块中频繁访问的64个predictions的信息放到共享内存.
  // 共享内存对同一线程块中的所有内存共享.
  // 这里每个线程，负责把一个prediction放到共享内存中.
  __shared__ scalar_t block_predictions[threadsPerBlock * PROP_SIZE];
  if (threadIdx.x < col_size) {
    for (int i = 0; i <  PROP_SIZE; ++i) {
      block_predictions[threadIdx.x * PROP_SIZE + i] = dev_predictions[idx[(threadsPerBlock * col_start + threadIdx.x)] * PROP_SIZE + i];
    }
  }
  __syncthreads();  // 同步！使用共享内存一定要同步，等64个线程把predictions放到共享内存后，再计算后面的iou.

  if (threadIdx.x < row_size) {
    const int cur_prediction_idx = threadsPerBlock * row_start + threadIdx.x;
    const scalar_t *cur_prediction = dev_predictions + idx[cur_prediction_idx] * PROP_SIZE;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;  // 只计算上三角的iou矩阵
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_prediction, block_predictions + i * PROP_SIZE, nms_overlap_thresh)) {
        t |= 1ULL << i;   // 如果iou大于阈值，通过位运算，t为64位0 or 1，把t的第i位设为1
      }
    }
    const int col_blocks = DIVUP(n_predictions, threadsPerBlock);
    dev_mask[cur_prediction_idx * col_blocks + col_start] = t;
  }
}


__global__ void nms_collect(const int64_t predictions_num, const int64_t col_blocks, int64_t top_k, const int64_t *idx, const int64_t *mask, int64_t *keep, int64_t *parent_object_index, int64_t *num_to_keep) {
  int64_t remv[MAX_COL_BLOCKS];
  int64_t num_to_keep_ = 0;
  // remv将用来保存该prediction是否抛弃，若抛弃则为1，即存在iou>阈值.
  // 同样的，要和mask保持一致，所以也会采用位运算，col_blocks * 64就是predictions_num.
  for (int i = 0; i < col_blocks; i++) {
      remv[i] = 0;
  }

  // 先初始化为背景预测.
  for (int i = 0; i < predictions_num; ++i) {
      parent_object_index[i] = 0;
  }

  for (int i = 0; i < predictions_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    // remv[nblock] & (1ULL << inblock) 与运算，将获得remv[nblock]的第inblock位.
    // 根据这个值判断这个预测有没有被抛弃.
    if (!(remv[nblock] & (1ULL << inblock))) {
      int64_t idxi = idx[i];    // 当前预测对应的原始索引.
      keep[num_to_keep_] = idxi;
      const int64_t *p = &mask[0] + i * col_blocks;     // 找到当前预测与其他预测对应的mask, mask为1, 表示重叠过大.
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];    // 抛弃掉与当前预测重叠过大的预测.
      }

      // 被抛弃的重叠框对应的parent_object_index 保存当前预测的索引.
      for (int j = i; j < predictions_num; j++) {
        int nblockj = j / threadsPerBlock;
        int inblockj = j % threadsPerBlock;
        // 位运算，1ULL是声明unsigned long long 1
        // p[nblock] & (1ULL << inblock) 与运算，将获得p[nblock]的第inblock位
        if (p[nblockj] & (1ULL << inblockj))
            parent_object_index[idx[j]] = num_to_keep_+1;
      }
      parent_object_index[idx[i]] = num_to_keep_+1;

      num_to_keep_++;

      if (num_to_keep_==top_k)
          break;
    }
  }

  // Initialize the rest of the keep array to avoid uninitialized values.
  for (int i = num_to_keep_; i < predictions_num; ++i)
      keep[i] = 0;

  *num_to_keep = min(top_k,num_to_keep_);
}

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::vector<at::Tensor> nms_cuda_forward(
        at::Tensor predictions,  // (N, 3+S)
        at::Tensor idx,     // (N, )  根据score进行排序后得到的索引
        float nms_overlap_thresh,
        unsigned long top_k) {

  const auto predictions_num = predictions.size(0);
  TORCH_CHECK(predictions.size(1) == PROP_SIZE, "Wrong number of offsets. Please adjust `PROP_SIZE`");

  const int col_blocks = DIVUP(predictions_num, threadsPerBlock);

  AT_ASSERTM (col_blocks < MAX_COL_BLOCKS, "The number of column blocks must be less than MAX_COL_BLOCKS. Increase the MAX_COL_BLOCKS constant if needed.");

  auto longOptions = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong);
  // mask负责记录两两预测之间的iou关系,若iou>iou_thr, 则mask的值为1, 否则为0.
  // 正常mask的shape应当是（predictions_num， predictions_num）， 但是为了节省内存,
  auto mask = at::empty({predictions_num * col_blocks}, longOptions);

  // 每一个线程将计算一个prediction与其他threadsPerBlock（64）个predictions之间的iou.
  dim3 blocks(DIVUP(predictions_num, threadsPerBlock),
              DIVUP(predictions_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  CHECK_CONTIGUOUS(predictions);
  CHECK_CONTIGUOUS(idx);
  CHECK_CONTIGUOUS(mask);

  AT_DISPATCH_FLOATING_TYPES(predictions.type(), "nms_cuda_forward", ([&] {
    nms_kernel<<<blocks, threads>>>(predictions_num,
                                    (scalar_t)nms_overlap_thresh,
                                    predictions.data<scalar_t>(),
                                    idx.data<int64_t>(),
                                    mask.data<int64_t>());
  }));

  auto keep = at::empty({predictions_num}, longOptions);  // 记录 保留预测的索引.
  auto parent_object_index = at::empty({predictions_num}, longOptions); // 记录每个预测对应的 最高分重叠预测 的索引， 根据这个索引，可以划分出属于同一目标的预测.
  auto num_to_keep = at::empty({}, longOptions);

  nms_collect<<<1, 1>>>(predictions_num, col_blocks, top_k,
                        idx.data<int64_t>(),
                        mask.data<int64_t>(),
                        keep.data<int64_t>(),
                        parent_object_index.data<int64_t>(),
                        num_to_keep.data<int64_t>());


  return {keep,num_to_keep,parent_object_index};
}

