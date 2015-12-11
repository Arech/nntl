#pragma once

using realmtx_t = nntl::math_types::realmtx_ty;
using real_t = nntl::math_types::real_ty;
using vec_len_t = realmtx_t::vec_len_t;
using numel_cnt_t = realmtx_t::numel_cnt_t;

void ASSERT_REALMTX_EQ(const realmtx_t& c1, const realmtx_t& c2, const char* descr = "", const real_t eps = 0) noexcept;

// declare etalon functions here

real_t rowvecs_renorm_ET(realmtx_t& m, real_t* pTmp)noexcept;
