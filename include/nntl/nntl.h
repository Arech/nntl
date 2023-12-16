/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2021, Arech (aradvert@gmail.com; https://github.com/Arech)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of NNTL nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

//////////////////////////////////////////////////////////////////////////
#include <nntl/compiler.h>
#include <nntl/_defs.h>
#include <nntl/common.h>
#include <nntl/math_details.h>
#include <nntl/utils.h>

#include <nntl/errors.h>
#include <nntl/train_data.h>

#include <nntl/weights_init.h>

#include <nntl/activation.h>
#include <nntl/layers.h>
#include <nntl/layer/_layer_base.h>
#include <nntl/layer/input.h>
#include <nntl/layer/output.h>
#include <nntl/layer/fully_connected.h>
#include <nntl/layer/pack_vertical.h>
#include <nntl/layer/pack_horizontal.h>
#include <nntl/layer/identity.h>
#include <nntl/layer/pack_horizontal_optional.h>
#include <nntl/layer/pack_tile.h>
#include <nntl/layer/extensions.h>
#include <nntl/nnet.h>
