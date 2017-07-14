#pragma once

#include <xmmintrin.h>
#include <pmmintrin.h>

namespace nntl {

	void inline disable_denormals()noexcept {
		unsigned int current_word = 0;
		_controlfp_s(&current_word, _DN_FLUSH, _MCW_DN);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	}

	void inline enable_denormals()noexcept {
		unsigned int current_word = 0;
		_controlfp_s(&current_word, _DN_SAVE, _MCW_DN);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
	}

	void inline global_denormalized_floats_mode() noexcept {
#if NNTL_DENORMALS2ZERO
		disable_denormals();
#endif
	}

	
}