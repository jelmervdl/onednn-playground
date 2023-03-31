#include <algorithm>
#include <cstring> // memcpy
#include <iostream>
#include <iomanip>
#include <numeric>
#include <unordered_map>
#include <vector>
#include <cmath>
#include "../3rd_party/cnpy/cnpy.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_types.h"

// Uncomment for using oneDNN for index_select. Otherwise memcpy will be used.
#define CONCAT_INDEX_SELECT

// Uncomment to collect the router weights while selecting tokens for the
// expert and execute the binary-mul as a post-op of the expert's linear layer.
// Otherwise the weight multiplier will be applied when the expert's output
// is copied back into the final merged output.
// #define WEIGH_EXPERT_JOINED

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
	memory::dim h;
	memory::dim w;
	T const *data;

	// Explicit constructor
	matrix(memory::dim h, memory::dim w, const T *data) :
		h(h),
		w(w),
		data(data) {}

	matrix(memory::dim h, memory::dim w, cnpy::NpyArray const &data) :
		h(h),
		w(w),
		data(data.data<T>()) {
		assert(data.num_vals == w * h);
		assert(data.num_bytes() == w * h * sizeof(T));
	}

	// One that derives from oneDNN
	explicit matrix(dnnl::memory const &mem) :
		h(mem.get_desc().get_dims()[0]),
		w(mem.get_desc().get_dims()[1]),
		data(reinterpret_cast<T const *>(mem.get_data_handle())) {
			// Make sure you've executed the command stream until this point in the
			// code and we're ready to read memory. This isn't done for you.
	}
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

template <typename T, std::size_t N=4>
std::ostream &operator<<(std::ostream &out, matrix<T> const &matrix) {
	// store precision so we can restore it.
	const auto default_precision(out.precision());
	const auto default_width(out.width());

	out << std::setprecision(4);
	out << "(" << matrix.h << "x" << matrix.w << ") " << typeid(T).name() << " {\n";
	for (size_t y = 0; y < matrix.h; ++y) {
		// Jump forward
		if (y == N && y+N < matrix.h) {
			out << "   ...\n";
			y = matrix.h - N;
		}

		for (size_t x = 0; x < matrix.w; ++x) {
			if (x == N && x+N < matrix.w) {
				out << ", ...";
				x = matrix.w - N;
			}
			out << (x > 0 ? "," : " ") << number<T>{matrix.data[y*matrix.w + x]};
		}
		out << "\n";
	}
	out << "}\n";

	// Restore previous precision
	return out << std::setprecision(default_precision) << std::setw(default_width);
}

// Print a loaded npz (well, names and sizes)
std::ostream &operator<<(std::ostream &out, cnpy::npz_t const &map) {
	out << "{\n";
	for (auto &&entry : map)
		out << "  " << entry.first << " = <" << entry.second.word_size << "b * " << entry.second.num_vals << ">\n";
	return out << "}";
}

// Easy dims printing
std::ostream &operator<<(std::ostream &out, memory::dims const &dims) {
	out << "[";
	for (auto it = dims.begin(); it != dims.end(); ++it) {
		if (it != dims.begin())
			out << ", ";
		out << *it;
	}
	return out << "]";
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
	std::vector<float> w1;
	std::vector<float> b1;
	std::vector<float> w2;
	std::vector<float> b2;
};

// Get scalar from our loaded `data` npz file
template <typename T>
T scalar(cnpy::NpyArray const &arr) {
	assert(arr.num_vals == 1);
	return arr.data<T>()[0];
}

// Little macro to give me the i'th slice of a bigger chunk of data. Think of
// the difference between calloc and malloc: does the same thing, but
// communicates different intention.
template <typename T>
std::vector<T> take_slice(std::size_t i, T const *data, const dnnl::memory::dims &dims) {
	auto size = product(dims);
	return std::vector<T>(data + i * size, data + (i + 1) * size);
}

int main(int argc, char** argv) {
	assert(argc > 1);
	auto data = cnpy::npz_load(argv[1]);

	std::cout << "Loaded npz from " << argv[1] << " " << data << std::endl; 

	dnnl::engine engine(dnnl::engine::kind::cpu, 0);

	dnnl::stream engine_stream(engine);

	// Tensor dimensions.
	const memory::dim expert_count   = scalar<long>(data["expert_count"]),
	                  expert_size    = scalar<long>(data["expert_size"]),
	                  embedding_size = scalar<long>(data["embedding_size"]),
	                  token_count    = scalar<long>(data["token_count"]);

	// bypass multiplier
	float b = scalar<float>(data["b"]);

	std::cout << "parameters { \n"
	          << "  expert_count   = " << expert_count << "\n"
	          << "  expert_size    = " << expert_size << "\n"
	          << "  embedding_size = " << embedding_size << "\n"
	          << "  token_count    = " << token_count << "\n"
	          << "  b              = " << b << "\n"
	          << "}\n";

	// Source (src), weights, bias, and destination (dst) tensors dimensions.
	memory::dims src_dims = {token_count, embedding_size};
	memory::dims dst_dims = {token_count, embedding_size};

	memory::dims expert_w1_dims = {embedding_size, expert_size};
	memory::dims expert_b1_dims = {1, expert_size};

	memory::dims expert_w2_dims = {expert_size, embedding_size};
	memory::dims expert_b2_dims = {1, embedding_size};

	// Initialize some source data
	assert(product(src_dims) == data["src"].num_vals);
	assert(product(src_dims) * sizeof(float) == data["src"].num_bytes());
	std::vector<float> src_data(data["src"].as_vec<float>());

	std::vector<float> dst_data(product(dst_dims));

	// Initialize router with data from npz. All of this assumes we

	assert(expert_count * token_count == data["router"].num_vals);
	assert(expert_count * token_count * sizeof(float) == data["router"].num_bytes());
	std::vector<float> router(data["router"].as_vec<float>());

	auto router_desc = memory::desc({expert_count, token_count}, dt::f32, tag::ab);
	auto router_mem = memory(router_desc, engine, router.data());

	// Initialize experts with data from npz

	std::vector<Expert> experts;

	for (size_t i = 0; i < expert_count; ++i) {
		experts.emplace_back(Expert{
			take_slice(i, data["experts_w1"].data<float>(), expert_w1_dims),
			take_slice(i, data["experts_b1"].data<float>(), expert_b1_dims),
			take_slice(i, data["experts_w2"].data<float>(), expert_w2_dims),
			take_slice(i, data["experts_b2"].data<float>(), expert_b2_dims)
		});
	}

	// ... and so it begins ...

	// Create memory descriptors and memory objects
	auto src_desc = memory::desc(src_dims, dt::f32, tag::ab);
	auto dst_desc = memory::desc(dst_dims, dt::f32, tag::ab);

	auto src_mem = memory(src_desc, engine, src_data.data());
	auto dst_mem = memory(dst_desc, engine, dst_data.data());

	std::cout << "src_data = " << matrix<float>{token_count, embedding_size, src_data.data()};

	std::cout << "router = " << matrix<float>{expert_count, token_count, router.data()};

	// === Copy src directly to dst which can serve as bypass for when a token is not selected by any expert ===
	auto bypass_desc = dnnl::eltwise_forward::primitive_desc(engine, prop_kind::forward_training, algorithm::eltwise_linear, src_desc, dst_desc, b, 0.0f);
	auto bypass_prim = eltwise_forward(bypass_desc);
	std::unordered_map<int, memory> bypass_args{
		{DNNL_ARG_SRC, src_mem},
		{DNNL_ARG_DST, dst_mem}
	};
	bypass_prim.execute(engine_stream, bypass_args);

	// Memory used by concat primitive to create an expert's input (and output)
#if defined(CONCAT_INDEX_SELECT)
	std::vector<memory::desc> concat_src_desc;
	std::vector<memory> concat_src_mems;

	concat_src_desc.reserve(token_count);
	concat_src_mems.reserve(token_count);
	std::unordered_map<int, memory> concat_args;
#endif

	for (size_t i = 0; i < expert_count; ++i) {
		/**
		 * Every expert will basically execute:
		 * ```py
		 *   # Create a filter that selects which tokens go into this expert
		 *   mask = router[expert,:] > 0
		 *   expert_input = tokens[mask]
		 * 
		 *   # Linear layer + relu
     *   expert_output1 = np.maximum(0.0, np.add(np.matmul(expert_input, expert_w1), expert_b1))
     *   expert_output2 = np.maximum(0.0, np.add(np.matmul(expert_output1, expert_w2), expert_b2))
     * 
     *   # Multiply by the per-token weights in the router, add to final output in the same slots
     *   # as the tokens appeared in the original input
     *   total_output[mask] += router[mask][:,np.newaxis] * expert_output2
     * ```
     */

		// Token count per expert, so we know how much memory to allocate
		memory::dim expert_token_count = std::accumulate(&router[token_count * i], &router[token_count * (i + 1)], 0, [](memory::dim acc, float val) {
			return acc + (val ? 1 : 0);
		});

		std::cout << "Expert " << i << " gets " << expert_token_count << " tokens\n";

		auto expert_src_desc = memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_src_mem = memory(expert_src_desc, engine);
		
		// Temporary memory used for expert's intermediate result
		auto expert_tmp_desc = memory::desc({expert_token_count, expert_size}, dt::f32, tag::ab);
		auto expert_tmp_mem = memory(expert_tmp_desc, engine);

		// Expert output memory, to be selectively added to dst_mem eventually
		auto expert_dst_desc = memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_dst_mem = memory(expert_dst_desc, engine);

		// Cut short excluded experts. Bit late, but at this point the expert will
		// at least appear in the exert_dst_descs list.
		if (expert_token_count == 0)
			continue;

#ifdef WEIGH_EXPERT_JOINED
		// Also collect the router's weights for the tokens we accepted.
		// TODO: Maybe if there was a smarter way to store a sparse router, this
		// could be a cheap operation as we're just selecting by 1 dimension and
		// then ditch all the zeros.
		auto expert_router_desc = memory::desc({expert_token_count, 1}, dt::f32, tag::ab);
		auto expert_router_mem = memory(expert_router_desc, engine);

		// (quick raw pointer for easy access)
		auto expert_router_mem_ptr = reinterpret_cast<float*>(expert_router_mem.get_data_handle());
#endif

#ifdef CONCAT_INDEX_SELECT
		// === oneDNN call to concat() to select rows for expert ===
		concat_src_desc.clear();
		concat_src_mems.clear();
		concat_args.clear();

		for (auto j = 0; j != token_count; ++j) {
			if (!router[token_count * i + j])
				continue;

#ifdef WEIGH_EXPERT_JOINED
			*expert_router_mem_ptr++ = router[token_count * i + j];
#endif

			// make the dnnl::memory object a view of a small part of the original input memory
			concat_src_desc.emplace_back(src_desc.submemory_desc(memory::dims{1, embedding_size}, {j, 0}));
			concat_src_mems.emplace_back(concat_src_desc.back(), engine, src_mem.get_data_handle());
			concat_args.insert({DNNL_ARG_MULTIPLE_SRC + (concat_src_mems.size() - 1), concat_src_mems.back()});
		}

		// Create concat primitive descriptor.
		auto concat_desc = concat::primitive_desc(engine, 0, concat_src_desc);
		auto concat_prim = concat(concat_desc);

		concat_args.insert({DNNL_ARG_DST, expert_src_mem});
		concat_prim.execute(engine_stream, concat_args);
#else
		// === Naive implementation of concat to select rows for expert ===
		for (auto j = 0, dst_offset = 0; j < token_count; ++j) {
			// Find the next True in the router to figure out the offset in dst_mem
			if (!router[token_count * i + j])
				continue;

#ifdef WEIGH_EXPERT_JOINED
			*expert_router_mem_ptr++ = router[token_count * i + j];
#endif

			std::cerr << "memcpy(" << (embedding_size * dst_offset) << ", " << (embedding_size * j) << ")" << std::endl;
			std::memcpy(
				reinterpret_cast<float*>(expert_src_mem.get_data_handle()) + (embedding_size * dst_offset++),
				reinterpret_cast<float*>(src_mem.get_data_handle()) + (embedding_size * j),
				embedding_size * sizeof(float));
		}
#endif


		engine_stream.wait(); // Purely for debugging
		std::cout << "expert_" << i << "_src_mem = " << matrix<float>(expert_src_mem);
		std::cout << "expected expert_" << i << "_src_mem = " << matrix<float>(expert_token_count, embedding_size, data[format("expert_{}_src", i)]);

		// Expert weights and biases (these are all the same shape for every expert)
		auto expert_w1_desc = memory::desc(expert_w1_dims, dt::f32, tag::ab);
		auto expert_w1_mem = memory(expert_w1_desc, engine, experts[i].w1.data());
		
		auto expert_b1_desc = memory::desc(expert_b1_dims, dt::f32, tag::ab);
		auto expert_b1_mem = memory(expert_b1_desc, engine, experts[i].b1.data());

		auto expert_w2_desc = memory::desc(expert_w2_dims, dt::f32, tag::ab);
		auto expert_w2_mem = memory(expert_w2_desc, engine, experts[i].w2.data());
		
		auto expert_b2_desc = memory::desc(expert_b2_dims, dt::f32, tag::ab);
		auto expert_b2_mem = memory(expert_b2_desc, engine, experts[i].b2.data());

		const float alpha = 0.f;
		const float beta = 0.f;

		// Post-op for relu, relevant for both matmuls
		post_ops matmul_ops;
		matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);

		primitive_attr matmul_attr;
		matmul_attr.set_post_ops(matmul_ops);

		// Create primitive descriptor.
		auto matmul_m1_desc = matmul::primitive_desc(engine,
			expert_src_desc,
			expert_w1_desc,
			expert_b1_desc,
			expert_tmp_desc,
			matmul_attr);

		// Create the primitive.
		auto matmul_m1_prim = matmul(matmul_m1_desc);

		// Primitive execution: matrix multiplication
		matmul_m1_prim.execute(engine_stream, {
			{DNNL_ARG_SRC,     expert_src_mem},
			{DNNL_ARG_WEIGHTS, expert_w1_mem},
			{DNNL_ARG_BIAS,    expert_b1_mem},
			{DNNL_ARG_DST,     expert_tmp_mem},
		});

		post_ops final_matmul_ops;
		
		// Same as previous post-op...
		final_matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);

#ifdef WEIGH_EXPERT_JOINED	
		// ... but multiply the matmul outputs by the router (per-token per-expert weights)
		auto binary_mul_idx = final_matmul_ops.len();
		final_matmul_ops.append_binary(algorithm::binary_mul, expert_router_desc);
#endif

		primitive_attr final_matmul_attr;
		final_matmul_attr.set_post_ops(final_matmul_ops);

		// Create primitive descriptor.
		auto matmul_m2_desc = matmul::primitive_desc(engine,
			expert_tmp_desc,
			expert_w2_desc,
			expert_b2_desc,
			expert_dst_desc,
			final_matmul_attr);

		// Create the primitive.
		auto matmul_m2_prim = matmul(matmul_m2_desc);

		matmul_m2_prim.execute(engine_stream, {
			{DNNL_ARG_SRC,     expert_tmp_mem},
			{DNNL_ARG_WEIGHTS, expert_w2_mem},
			{DNNL_ARG_BIAS,    expert_b2_mem},
			{DNNL_ARG_DST,     expert_dst_mem},
#ifdef WEIGH_EXPERT_JOINED
			{DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_mul_idx) | DNNL_ARG_SRC_1, expert_router_mem},
#endif
		});

#ifndef WEIGH_EXPERT_JOINED
		// Only printing this in the scenario where we didn't add the binary_mul
		// post-op to multiply by the weights. Because we don't do that in Python,
		// so the comparison is void at that point.
		engine_stream.wait(); // For debugging only
		std::cout << "expert_" << i << "_dst = " << matrix<float>(expert_dst_mem);
		std::cout << "expected expert_" << i << "_dst_mem = " << matrix<float>(expert_token_count, embedding_size, data[format("expert_{}_dst", i)]);
#endif

		// Copy data back to the final output. Since we're doing Top-K and we want
		// to sum the expert of multiple experts per token, we're just going to
		// for-loop binary add for each row. I'm open to better implementations!
		for (auto j = 0, dst_offset = 0; j < expert_token_count; ++j, ++dst_offset) {
			// Find the next True in the router to figure out the offset in dst_mem
			while (!router[token_count * i + dst_offset])
				++dst_offset;

			auto expert_slice_desc = expert_dst_desc.submemory_desc({1, embedding_size}, {j, 0});
			auto dst_slice_desc = dst_desc.submemory_desc({1, embedding_size}, {dst_offset, 0});

#ifdef WEIGH_EXPERT_JOINED
			auto binary_desc = binary::primitive_desc(engine, algorithm::binary_add,
				expert_slice_desc, // source 0
				dst_slice_desc,    // source 1
				dst_slice_desc);   // dest

			auto binary_prim = binary(binary_desc);
			binary_prim.execute(engine_stream, {
				{DNNL_ARG_SRC_0, expert_dst_mem},
				{DNNL_ARG_SRC_1, dst_mem},
				{DNNL_ARG_DST, dst_mem}
			});
#else
			// Select weight from router. This will be broadcasted to {1,embedding_size}.
			auto router_slice_desc = router_desc.submemory_desc({1, 1}, {static_cast<memory::dim>(i), dst_offset});

			post_ops binary_post_ops;
			binary_post_ops.append_sum();

			primitive_attr binary_attr;
			binary_attr.set_post_ops(binary_post_ops);

			auto binary_desc = binary::primitive_desc(engine, algorithm::binary_mul,
				expert_slice_desc, // source 0
				router_slice_desc, // source 1
				dst_slice_desc,    // dest
				binary_attr);

			auto binary_prim = binary(binary_desc);
			binary_prim.execute(engine_stream, {
				{DNNL_ARG_SRC_0, expert_dst_mem},
				{DNNL_ARG_SRC_1, router_mem},
				{DNNL_ARG_DST, dst_mem}
			});
#endif
		}
	}

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
