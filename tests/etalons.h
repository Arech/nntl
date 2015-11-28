#pragma once

using floatmtx_t = nntl::math_types::floatmtx_ty;
using float_t_ = nntl::math_types::float_ty;
using vec_len_t = floatmtx_t::vec_len_t;
using numel_cnt_t = floatmtx_t::numel_cnt_t;

// declare etalon functions here

float_t_ rowvecs_renorm_ET(floatmtx_t& m, float_t_* pTmp)noexcept;