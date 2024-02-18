#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_cl_init(void);

void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void * ggml_cl_host_malloc(size_t size);
void   ggml_cl_host_free(void * ptr);

void ggml_cl_free_data(const struct ggml_tensor* tensor);

void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);

void ggml_compute_cl_dup_same_cont(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_dup_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_dup_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_cl_softmax(const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst);

void ggml_compute_cl_abs_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);


void ggml_compute_cl_sgn_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_neg_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_step_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_tanh_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_elu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_relu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_gelu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_gelu_quick_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_silu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_cl_permute(const struct ggml_compute_params * params, const struct ggml_tensor * src0, struct ggml_tensor * dst);

// xr NOP
void ggml_cl_transpose(const struct ggml_compute_params * params, const struct ggml_tensor * src0, struct ggml_tensor * dst);

void ggml_cl_view(const struct ggml_compute_params * params, const struct ggml_tensor * src0, struct ggml_tensor * dst);

void ggml_cl_reshape(const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_cl_rope_f32(const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst,
        const bool forward);

void ggml_cl_rope_f16(const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst,
        const bool forward);

void ggml_cl_rms_norm_f32(const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst);

void ggml_compute_cl_add_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst);


void ggml_compute_cl_add_f16_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst);

void ggml_compute_cl_add_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst);

void ggml_compute_cl_add_q_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst);

void ggml_compute_cl_get_rows_q(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst);

void ggml_compute_cl_get_rows_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst);

void ggml_compute_cl_get_rows_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst);

#ifdef  __cplusplus
}
#endif
