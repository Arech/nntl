/*
This file is a part of NNTL project (https://github.com/Arech/nntl)

Copyright (c) 2015-2019, Arech (al.rech@gmail.com; https://github.com/Arech)
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

//ignore this file, to be done later

#include "../../utils/tictoc.h"

namespace nntl {
namespace mt {

	template<typename BaseClass_t>
	class profiler : public BaseClass_t {
	public:
		typedef BaseClass_t base_class_t;
		typedef utils::tictoc tictoc_t;

		//////////////////////////////////////////////////////////////////////////
		//members
	public:

		
		//////////////////////////////////////////////////////////////////////////
		//methods
	public:
		~profiler()noexcept {};
		profiler()noexcept : base_class_t() {}

		//////////////////////////////////////////////////////////////////////////
		//overridings

		// ������� � ����������
		
		// ������ ������ ������������� <������ ������> => <������ ������� ��� ��������� ������� �������>
		// ���, ����� ��� ����� ����� ������������ �������� �������� ������� ��� ����� �������. ������, �������� � ���, ���
		// ���� ����������, � ��� ���� �� �� �������� (??? ��� ����� ���� � ����������� ������ ��������?), ��� �������� �����.
		// ���-�� ��������� �������� ��������� �� ����� �������, ���-�� ������� - �� ����� �����.
		// ���-�� ��������� �������� ������ �� �������� � ����� �������� � ���������.
		// ���-�� ������� - �������� � ����� ���� ������ ������������.
		// ���� �������� ����������� ��������� ������ ������� ����������. ����� �����������, ��� ����� �������. ����
		// ������������ ����� ������� ������� - ��������� �� ������� �������� � ������� � ���� void*
		//		* ����� ���� ����� ���������� ��������� �� ������� � �������� ����� �� ���������.
		//		* ����� ���� ����� ���������� ������ ID �������, � ������� �������� ���������� ������ ������� ��� �������
		//			(��� ������� �������� � ��������� ������)
		// ������:
		//		* ����� �������� ���������� ����� ������������� ������� ���������� � ������� �� �������������
		//		* ����� ��������� ������������ ������ � ����������� �������� ������, ��������� ��������������� ������.
		struct mt_router {
			//��������� ������� ��� ��������� ������� ������� ������. nullptr ��� �������������� �����������.
			void* get(auto decisionValue)noexcept {};
			//����������==������ ����������� ����������. ������ ���������� � ������� ������� �������������.
			void update(auto updateDescription)noexcept {};
		};

		mt_router m_evMul_ip_router;
		typedef void(base_class_t::*evMul_ip_func_t)(realmtx_t& A, const realmtx_t& B)noexcept;

		void evMul_ip(realmtx_t& A, const realmtx_t& B)noexcept {
			const auto decisionValue = A.numel();

			//���� � ������� ���� �������� ��� ������� ������� - �������� �������, ����� - ����������� �������� � ���� ������
			evMul_ip_func_t pF = static_cast<evMul_ip_func_t>m_evMul_ip_router.get(decisionValue);
			if (nullptr==pF) {
				//����� ����� ������ ��� ������� ������� ������



			} else {
				//������ �������� ������ � ��.
				pF(A, B);
				//TODO: ����� ����� ��� �� �������� ������������������ ������ ������ ������� � ���������� � � ���, ���
				// �������� � �������� ��������. ����� ���� ������������ �����������, ��������� ������ ������ �������.
			}

		}


	};

}
}