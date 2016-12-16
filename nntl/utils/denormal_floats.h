#pragma once

#include <xmmintrin.h>
#include <pmmintrin.h>

namespace nntl {
	void inline global_denormalized_floats_mode() noexcept {
#if NNTL_DENORMALS2ZERO
		//_controlfp(_DN_FLUSH, _MCW_DN);
		unsigned int current_word = 0;
		_controlfp_s(&current_word, _DN_FLUSH, _MCW_DN);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
	}

}