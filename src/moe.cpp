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

	// Memory used by concat primitive to create an expert's input (and output)
#if defined(CONCAT_INDEX_SELECT) || defined(CONCAT_INDEX_RESTORE)
	std::vector<memory::desc> concat_src_desc;
	std::vector<memory> concat_src_mems;

	concat_src_desc.reserve(token_count);
	concat_src_mems.reserve(token_count);
	std::unordered_map<int, memory> concat_args;
#endif

#ifdef CONCAT_INDEX_RESTORE
	// Memory for the experts. We keep it around so we can concat the bits we
	// want at the end into the final MoE output.
	std::vector<memory::desc> expert_dst_descs;
	std::vector<memory> expert_dst_mems;
	expert_dst_descs.reserve(expert_count);
	expert_dst_mems.reserve(expert_count);
#endif

	for (size_t i = 0; i < expert_count; ++i) {
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

#ifdef CONCAT_INDEX_RESTORE
		// Expert output memory, to be selectively added to dst_mem eventually.
		// We keep them around in vectors so we can use them later in the final concat.
		auto expert_dst_desc = expert_dst_descs.emplace_back(memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab));
		auto expert_dst_mem = expert_dst_mems.emplace_back(memory(expert_dst_desc, engine));
#else
		// Expert output memory, to be selectively added to dst_mem eventually
		auto expert_dst_desc = memory::desc({expert_token_count, embedding_size}, dt::f32, tag::ab);
		auto expert_dst_mem = memory(expert_dst_desc, engine);
#endif

		// Cut short excluded experts. Bit late, but at this point the expert will
		// at least appear in the exert_dst_descs list.
		if (expert_token_count == 0)
			continue;

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

		// Initialise expert's output with the bypass values (we'll be adding to it)
		auto expert_bypass_desc = eltwise_forward::primitive_desc(engine,
			prop_kind::forward_training,
			algorithm::eltwise_linear,
			expert_src_desc,
			expert_dst_desc,
			b,
			0.0f);

		auto expert_bypass_prim = eltwise_forward(expert_bypass_desc);
		expert_bypass_prim.execute(engine_stream, std::unordered_map<int, memory>{
			{DNNL_ARG_SRC, expert_src_mem},
			{DNNL_ARG_DST, expert_dst_mem}
		});

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

		// Same as previous post-op, but also tells it to sum output to expert's output
		post_ops final_matmul_ops;
		final_matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
		final_matmul_ops.append_sum();

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
		});

		engine_stream.wait(); // For debugging only

		std::cout << "expert_" << i << "_dst = " << matrix<float>(expert_dst_mem);
		std::cout << "expected expert_" << i << "_dst_mem = " << matrix<float>(expert_token_count, embedding_size, data[format("expert_{}_dst", i)]);

#ifndef CONCAT_INDEX_RESTORE
		engine_stream.wait(); // Wait for memory to settle

		for (auto j = 0, dst_offset = 0; j < token_count; ++j) {
			// Find the next True in the router to figure out the offset in dst_mem
			if (!router[token_count * i + j])
				continue;

			std::memcpy(
				reinterpret_cast<float*>(dst_mem.get_data_handle()) + (embedding_size * j),
				reinterpret_cast<float*>(expert_dst_mem.get_data_handle()) + (embedding_size * dst_offset++),
				embedding_size * sizeof(float));
		}
#endif
	}

#ifdef CONCAT_INDEX_RESTORE
	concat_src_desc.clear();
	concat_src_mems.clear();
	concat_args.clear();

	// token offset per expert
	std::vector<memory::dim> expert_offsets(expert_count, 0);
	
	for (auto j = 0; j < token_count; ++j) {
		// TODO: I wish I could transpose the router for a bit, and get it
		// token-major instead of expert-major.
		
		// Find out which expert this token has to be read from.
		size_t expert_i = expert_count;
		for (size_t i = 0; i < expert_count; ++i) {
			if (router[i * token_count + j]) {
				expert_i = i;
				break;
			}
		}

		// Was this token routed to any expert? We assume so, otherwise our indices
		// will be off.
		assert(expert_i < expert_count);

		std::cerr << "Taking token " << j << " from expert " << expert_i << std::endl;

		// make the dnnl::memory object a view of a small part of the original expert's output memory
		concat_src_desc.emplace_back(expert_dst_descs[expert_i].submemory_desc(memory::dims{1, embedding_size}, {expert_offsets[expert_i]++, 0}));
		concat_src_mems.emplace_back(concat_src_desc.back(), engine, expert_dst_mems[expert_i].get_data_handle());
		concat_args.insert({DNNL_ARG_MULTIPLE_SRC + (concat_src_mems.size() - 1), concat_src_mems.back()});
	}

	// Create concat primitive descriptor.
	auto concat_desc = concat::primitive_desc(engine, 0, concat_src_desc);
	auto concat_prim = concat(concat_desc);

	// output to the final massive matrix
	concat_args.insert({DNNL_ARG_DST, dst_mem});

	concat_prim.execute(engine_stream, concat_args);
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
