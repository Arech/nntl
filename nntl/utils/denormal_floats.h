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

	bool inline isDenormalsOn()noexcept {
		unsigned int current_word = 0;
		const auto err = _controlfp_s(&current_word, 0, 0);
		bool bDisabled = false;
		if (!err) {
			if ((current_word & _MCW_DN) == _DN_FLUSH) {
				bDisabled = true;
			}
		}

		const auto zm = _MM_GET_DENORMALS_ZERO_MODE();
		if (zm == _MM_DENORMALS_ZERO_OFF) bDisabled = false;

		const auto fm = _MM_GET_FLUSH_ZERO_MODE();
		if (fm == _MM_FLUSH_ZERO_OFF) bDisabled = false;

		return !bDisabled;
	}

	
}