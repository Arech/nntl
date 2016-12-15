#pragma once

#include <xmmintrin.h>
#include <pmmintrin.h>

namespace nntl {
	void inline global_denormalized_floats_mode() noexcept {
#if NNTL_DENORMALS2ZERO==1

#ifndef NNTL_IM_AWARE_OF_DENORMALS_THANK_YOU
#pragma message( __FILE__ ": *** denormalized floats will be flushed to zero in global_denormalized_floats_mode()" )
#endif // !NNTL_IM_AWARE_OF_DENORMALS_THANK_YOU

		//_controlfp(_DN_FLUSH, _MCW_DN);
		unsigned int current_word = 0;
		_controlfp_s(&current_word, _DN_FLUSH, _MCW_DN);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#else

#ifndef NNTL_IM_AWARE_OF_DENORMALS_THANK_YOU
#pragma message ( __FILE__ ": *** denormalized floats settings are left as is in global_denormalized_floats_mode()");
#endif // !NNTL_IM_AWARE_OF_DENORMALS_THANK_YOU

#endif
	}

}