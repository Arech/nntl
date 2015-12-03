/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015, Arech (aradvert@gmail.com; https://github.com/Arech)
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

namespace nntl {

	//This class is also used in layer* and layer_pack classes
	struct _nnet_errs {
		enum ErrorCode {
			Success = 0,
			InvalidTD,
			BatchSizeMustBeMultipleOfTrainDataLength,
			InvalidInputLayerNeuronsCount,
			InvalidOutputLayerNeuronsCount,
			CantAllocateMemoryForActivations,
			CantAllocateMemoryForDropoutMask,
			CantAllocateMemoryForTempData,
			CantAllocateMemoryForWeights,
			CantInitializeIMath,
			CantInitializeObserver,
			CantInitializeGradWorks,
			CantInitializeWeights,
			NNDiverged,
		};

		//TODO: table lookup would be better here. But it's not essential
		static const nntl::strchar_t* get_error_str(const ErrorCode ec) noexcept {
			switch (ec) {
			case Success: return NNTL_STRING("No error / success.");
			case InvalidTD: return NNTL_STRING("Invalid training data passed.");
			case BatchSizeMustBeMultipleOfTrainDataLength: return NNTL_STRING("Batch size must be multiple of training data length.");
			case InvalidInputLayerNeuronsCount: return NNTL_STRING("Input layer neurons count mismatches train_x width.");
			case InvalidOutputLayerNeuronsCount: return NNTL_STRING("Output layer neurons count mismatches train_y width.");
			case CantAllocateMemoryForActivations: return NNTL_STRING("Cant allocate memory for neuron activations");
			case CantAllocateMemoryForDropoutMask: return NNTL_STRING("Cant allocate memory for dropout mask");
			case CantAllocateMemoryForTempData: return NNTL_STRING("Cant allocate memory for temporarily data");
			case CantAllocateMemoryForWeights: return NNTL_STRING("Cant allocate memory for weight matrix");
			case CantInitializeIMath: return NNTL_STRING("Cant initialize iMath interface");
			case CantInitializeObserver: return NNTL_STRING("Cant initialize observer");
			case CantInitializeGradWorks: return NNTL_STRING("Cant initialize grad_works object");
			case CantInitializeWeights: return NNTL_STRING("Weights initialization failed");
			case NNDiverged: return NNTL_STRING("NN diverged! (Training loss value surpassed the threshold from opts.divergenceCheckThreshold())");
			default: NNTL_ASSERT(!"WTF?"); return NNTL_STRING("Unknown code.");
			}
		}

	};

}