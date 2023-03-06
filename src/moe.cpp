#include <algorithm>
#include <cmath> // sin, tan, cos
#include <cstring> // memcpy
#include <iostream>
#include <iomanip>
#include <numeric>
#include <unordered_map>
#include <vector>
#include "../3rd_party/cnpy/cnpy.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"

// Uncomment for using oneDNN for index_select. Otherwise memcpy will be used.
#define CONCAT_INDEX_SELECT

// Uncomment for index_restore using oneDNN concat at the end. Otherwise a
// memcpy after each expert will be used.
#define CONCAT_INDEX_RESTORE

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

// Shortcut for getting size of a {a,b,c} sized tensor.
dnnl::memory::dim product(const dnnl::memory::dims &dims) {
	return std::accumulate(
		dims.begin(),
		dims.end(),
		(dnnl::memory::dim) 1,
		std::multiplies<dnnl::memory::dim>());
}

// Utility templates for printing 2D tensors to ostreams.
template <typename T>
struct matrix {
	size_t w;
	size_t h;
	T* data;
};

template <typename T>
struct number {
	T const &val;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, number<T> const &number) {
	return out << std::fixed << std::setw(8) << number.val;
}

// (Bloody hack to print uint8_t as numbers, not characters)
template <>
std::ostream &operator<<(std::ostream &out, number<uint8_t> const &number) {
	return out << std::fixed << std::setw(4) << static_cast<size_t>(number.val);
}

template <typename T>
std::ostream &operator<<(std::ostream &out, matrix<T> const &matrix) {
	// store precision so we can restore it.
	const auto default_precision(out.precision());
	const auto default_width(out.width());

	out << std::setprecision(4);
	out << "(" << matrix.w << "x" << matrix.h << ") " << typeid(T).name() << " {\n";
	for (size_t x = 0; x < matrix.w; ++x) {
		for (size_t y = 0; y < matrix.h; ++y) {
			out << (y > 0 ? "," : " ") << number<T>{matrix.data[x*matrix.h + y]};
		}
		out << "\n";
	}
	out << "}\n";

	// Restore previous precision
	return out << std::setprecision(default_precision) << std::setw(default_width);
}

// Very simple replacement for std::format introduced in C++20. Only supports
// replacing `{}` in the template string with whatever `operator<<` for that
// type turns it into.
std::string format(std::string const &formatTemplate) { return formatTemplate; }

template <typename Arg>
std::string format(std::string const &formatTemplate, Arg arg) {
  std::ostringstream os;
  auto index = formatTemplate.find("{}");
  assert(index != std::string::npos);
  os << formatTemplate.substr(0, index) << arg << formatTemplate.substr(index + 2);
  return os.str();
}

template <typename Arg, typename... Args>
std::string format(std::string const &formatTemplate, Arg arg, Args... args) {
  std::ostringstream os;
  auto index = formatTemplate.find("{}");
  assert(index != std::string::npos);
  os << formatTemplate.substr(0, index) << arg << format(formatTemplate.substr(index + 2), std::forward<Args>(args)...);
  return os.str();
}

// Memory structure for an Expert, so we can have std::vector<Expert> instead of
// a bunch of vectors. Efficiency was not on my mind when deciding this.
struct Expert {
	std::size_t size;
	std::vector<float> weights;
	std::vector<float> bias;
	
	Expert(std::size_t size, const float *w_data, const float *b_data) :
		size(size),
		weights(w_data, w_data + size * size),
		bias(b_data, b_data + size)
	{
		//
	}
};

int main(int argc, char** argv) {
	assert(argc > 1);
	auto data = cnpy::npz_load(argv[1]);

	dnnl::engine engine(dnnl::engine::kind::cpu, 0);

	dnnl::stream engine_stream(engine);

	// Tensor dimensions.
	const memory::dim expert_count = 3, token_count = 7, embedding_size = 5;

	float b = 0.1; // bypass multiplier

	// Source (src), weights, bias, and destination (dst) tensors dimensions.
	memory::dims src_dims = {token_count, embedding_size};
	memory::dims dst_dims = {token_count, embedding_size};

	memory::dims expert_w_dims = {embedding_size, embedding_size};
	memory::dims expert_b_dims = {1, embedding_size};

	// Initialize some source data
	assert(product(src_dims) == data["src"].num_vals);
	assert(product(src_dims) * sizeof(float) == data["src"].num_bytes());
	std::vector<float> src_data(data["src"].as_vec<float>());

	std::vector<float> dst_data(product(dst_dims));

	// Initialize router with data from npz. All of this assumes we

	assert(expert_count * token_count == data["router"].num_vals);
	assert(expert_count * token_count * sizeof(uint8_t) == data["router"].num_bytes());
	std::vector<uint8_t> router(data["router"].as_vec<uint8_t>());

	// Initialize experts with data from npz

	std::vector<Expert> experts;

	for (size_t i = 0; i < expert_count; ++i) {
		experts.emplace_back(
			embedding_size,
			data["experts_w"].data<float>() + i * embedding_size * embedding_size,
			data["experts_b"].data<float>() + i * embedding_size
		);
	}

	// ... and so it begins ...

	// Create memory descriptors and memory objects
	auto src_desc = memory::desc(src_dims, dt::f32, tag::ab);
	auto dst_desc = memory::desc(dst_dims, dt::f32, tag::ab);

	auto src_mem = memory(src_desc, engine, src_data.data());
	auto dst_mem = memory(dst_desc, engine, dst_data.data());

	std::cout << "src_data = " << matrix<float>{token_count, embedding_size, src_data.data()};

	std::cout << "router = " << matrix<uint8_t>{expert_count, token_count, router.data()};

	// === Copy src directly to dst which can serve as bypass for when a token is not selected by any expert ===
	auto bypass_desc = dnnl::eltwise_forward::primitive_desc(engine, prop_kind::forward_training, algorithm::eltwise_linear, src_desc, dst_desc, b, 0.0f);
	auto bypass_prim = eltwise_forward(bypass_desc);
	std::unordered_map<int, memory> bypass_args{
		{DNNL_ARG_SRC, src_mem},
		{DNNL_ARG_DST, dst_mem}
	};
	bypass_prim.execute(engine_stream, bypass_args);

	// Memory used by concat primitive to create an expert's input
#ifdef CONCAT_INDEX_SELECT
	std::vector<memory::desc> concat_src_desc;
	std::vector<memory> concat_src_mems;

	concat_src_desc.reserve(token_count);
	concat_src_mems.reserve(token_count);
	std::unordered_map<int, memory> concat_args;
#endif

#ifdef CONCAT_INDEX_RESTORE
	std::vector<memory::desc> expert_dst_descs;
	std::vector<memory> expert_dst_mems;
#endif

	for (size_t i = 0; i < expert_count; ++i) {
#ifdef CONCAT_INDEX_SELECT
		// === oneDNN call to concat() to select rows for expert ===
		concat_src_desc.clear();
		concat_src_mems.clear();
		concat_args.clear();

		for (auto j = 0; j != token_count; ++j) {
			if (!router[token_count * i + j])
				continue;

			// make the dnnl::memory object a view of a small part of the original input memory
			concat_src_desc.emplace_back(src_desc.submemory_desc(memory::dims{1, embedding_size}, {j, 0}));
			concat_src_mems.emplace_back(concat_src_desc.back(), engine, src_mem.get_data_handle());
			concat_args.insert({DNNL_ARG_MULTIPLE_SRC + (concat_src_mems.size() - 1), concat_src_mems.back()});
		}

		memory::dim expert_token_count = concat_src_mems.size();

		// Create concat primitive descriptor.
		auto concat_desc = concat::primitive_desc(engine, 0, concat_src_desc);
		auto concat_prim = concat(concat_desc);

		auto expert_src_desc = memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_src_mem = memory(expert_src_desc, engine);
		concat_args.insert({DNNL_ARG_DST, expert_src_mem});

		concat_prim.execute(engine_stream, concat_args);

		engine_stream.wait(); // Purely for debugging
#else
		// === Naive implementation of concat ===
		memory::dim expert_token_count = std::accumulate(&router[token_count * i], &router[token_count * (i + 1)], 0, [](uint8_t acc, uint8_t val) {
			return acc + (val ? 1 : 0);
		});

		auto expert_src_desc = memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_src_mem = memory(expert_src_desc, engine);

		for (auto j = 0, dst_offset = 0; j < token_count; ++j) {
			// Find the next True in the router to figure out the offset in dst_mem
			if (!router[token_count * i + j])
				continue;

			std::cerr << "memcpy(" << (embedding_size * dst_offset) << ", " << (embedding_size * j) << ")" << std::endl;
			std::memcpy(
				reinterpret_cast<float*>(expert_src_mem.get_data_handle()) + (embedding_size * dst_offset++),
				reinterpret_cast<float*>(src_mem.get_data_handle()) + (embedding_size * j),
				embedding_size * sizeof(float));
		}
#endif

		std::cout << "expert_" << i << "_src_mem = " << matrix<float>{
			static_cast<size_t>(expert_token_count), embedding_size,
			reinterpret_cast<float*>(expert_src_mem.get_data_handle())};

#ifdef CONCAT_INDEX_RESTORE
		auto expert_dst_desc = expert_dst_descs.emplace_back(memory::dims{expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_dst_mem = expert_dst_mems.emplace_back(expert_dst_descs.back(), engine);
#else
		auto expert_dst_desc = memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_dst_mem = memory(expert_dst_desc, engine);
#endif

		auto expert_weights_desc = memory::desc(expert_w_dims, dt::f32, tag::ab);
		auto expert_weights_mem = memory(expert_weights_desc, engine, experts[i].weights.data());
		
		auto expert_bias_desc = memory::desc(expert_b_dims, dt::f32, tag::ab);
		auto expert_bias_mem = memory(expert_bias_desc, engine, experts[i].bias.data());

		// Create primitive descriptor.
		auto matmul_desc = matmul::primitive_desc(engine,
			expert_src_desc,
			expert_weights_desc,
			expert_bias_desc,
			expert_dst_desc);

		// Create the primitive.
		auto matmul_prim = matmul(matmul_desc);

		// Primitive arguments.
		std::unordered_map<int, memory> matmul_args;
		matmul_args.insert({DNNL_ARG_SRC,     expert_src_mem});
		matmul_args.insert({DNNL_ARG_WEIGHTS, expert_weights_mem});
		matmul_args.insert({DNNL_ARG_BIAS,    expert_bias_mem});
		matmul_args.insert({DNNL_ARG_DST,     expert_dst_mem});

		// Primitive execution: matrix multiplication
		matmul_prim.execute(engine_stream, matmul_args);

		// std::cout << "expert_" << i << "_weights = " << matrix<float>{embedding_size,embedding_size,reinterpret_cast<float*>(expert_weights_mem.get_data_handle())};
		std::cout << "expert_" << i << "_dst = " << matrix<float>{static_cast<size_t>(expert_token_count), embedding_size, reinterpret_cast<float*>(expert_dst_mem.get_data_handle())};
		std::cout << "expected expert_" << i << "_dst_mem = " << matrix<float>{static_cast<size_t>(expert_token_count), embedding_size, data[format("expert_{}_dst", i)].data<float>()};
		
#ifndef CONCAT_INDEX_RESTORE
		// Copy results from expert back to final output memory
		engine_stream.wait();

		for (auto j = 0, dst_offset = 0; j < expert_token_count; ++j, ++dst_offset) {
			// Find the next True in the router to figure out the offset in dst_mem
			while (!router[token_count * i + dst_offset])
				++dst_offset;

			std::memcpy(
				reinterpret_cast<float*>(dst_mem.get_data_handle()) + (embedding_size * dst_offset),
				reinterpret_cast<float*>(expert_dst_mem.get_data_handle()) + (embedding_size * j),
				embedding_size * sizeof(float));
		}

		engine_stream.wait(); // for debugging only
		std::cout << "dst_data (after expert " << i << ") = " << matrix<float>{token_count, embedding_size, reinterpret_cast<float*>(dst_mem.get_data_handle())};
#endif
	}

#ifdef CONCAT_INDEX_RESTORE
	// Silly way of restoring the final output matrix by selectively concatinating
	// from the various expert output matrices. I don't see how this can scale to
	// a large number of experts.
	std::unordered_map<int, memory> dst_concat_args;
	std::vector<memory::desc> dst_concat_descs;
	std::vector<memory> dst_concat_mems;

	// Offsets which row inside each expert we're at
	std::vector<memory::dim> expert_src_offsets(expert_count, 0);

	for (auto j = 0; j < token_count; ++j) {
		for (auto i = 0; i < expert_count; ++i) {
			// Find the expert that is responsible for this token j.
			if (!router[token_count * i + j])
				continue;

			// std::cerr << "For token " << j << " we take row " << expert_src_offsets[i] << " from expert " << i << std::endl;
			dst_concat_descs.emplace_back(expert_dst_descs[i].submemory_desc({1, embedding_size}, {expert_src_offsets[i]++, 0}));
			dst_concat_mems.emplace_back(dst_concat_descs.back(), engine, expert_dst_mems[i].get_data_handle());
			dst_concat_args.insert({DNNL_ARG_MULTIPLE_SRC + j, dst_concat_mems.back()});
			break; // Go to next token
		}
	}

	auto dst_concat_desc = concat::primitive_desc(engine, 0, dst_concat_descs);
	auto dst_concat_prim = concat(dst_concat_desc);

	dst_concat_args.insert({DNNL_ARG_DST, dst_mem});
	dst_concat_prim.execute(engine_stream, dst_concat_args);
#endif

	// Wait for the computation to finalize.
	engine_stream.wait();
	std::cout << "final dst_data =    " << matrix<float>{token_count, embedding_size, reinterpret_cast<float*>(dst_mem.get_data_handle())};
	std::cout << "expected dst_data = " << matrix<float>{token_count, embedding_size, data["dst"].data<float>()};

	// === Compare output ===
	float summed_abs_diff = 0;
	float* dst_mem_ptr = reinterpret_cast<float*>(dst_mem.get_data_handle());
	float* ref_mem_ptr = data["dst"].data<float>();
	for (size_t i = 0; i < token_count * embedding_size; ++i) {
		summed_abs_diff += std::fabs(dst_mem_ptr[i] - ref_mem_ptr[i]);
	}
	std::cout << "sum(|a-b|) = " << summed_abs_diff << std::endl;

	return 0;
}
