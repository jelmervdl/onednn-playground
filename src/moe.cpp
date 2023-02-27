#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;


dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(
  	dims.begin(),
  	dims.end(),
  	(dnnl::memory::dim) 1,
		std::multiplies<dnnl::memory::dim>());
}


void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
	dnnl::engine eng = mem.get_engine();
	size_t size = mem.get_desc().get_size();
	uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
  if (!src) throw std::runtime_error("get_data_handle returned nullptr.");
  for (size_t i = 0; i < size; ++i)
    ((uint8_t *) handle)[i] = src[i];
}


void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();
  if (!handle) throw std::runtime_error("handle is nullptr.");
	uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
  if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
  for (size_t i = 0; i < size; ++i)
      dst[i] = ((uint8_t *) handle)[i];
}


int main() {
	dnnl::engine engine(dnnl::engine::kind::cpu, 0);

	dnnl::stream engine_stream(engine);

	// Tensor dimensions.
  const memory::dim MB = 3, // batch size
          M = 128, K = 256, N = 512;

  // Source (src), weights, bias, and destination (dst) tensors dimensions.
  memory::dims src_dims = {MB, M, K};
  memory::dims weights_dims = {MB, K, N};
  memory::dims bias_dims = {1, 1, N};
  memory::dims dst_dims = {MB, M, N};

  // Allocate buffers.
  std::vector<float> src_data(product(src_dims));
  std::vector<float> weights_data(product(weights_dims));
  std::vector<float> bias_data(product(bias_dims));
  std::vector<float> dst_data(product(dst_dims));

  // Initialize src, weights, bias.
  std::generate(src_data.begin(), src_data.end(), []() {
      static int i = 0;
      return std::cos(i++ / 10.f);
  });
  std::generate(weights_data.begin(), weights_data.end(), []() {
      static int i = 0;
      return std::sin(i++ * 2.f);
  });
  std::generate(bias_data.begin(), bias_data.end(), []() {
      static int i = 0;
      return std::tanh(float(i++));
  });

  // Create memory descriptors and memory objects for src, weights, bias, and
  // dst.
  auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
  auto weights_md = memory::desc(weights_dims, dt::f32, tag::abc);
  auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
  auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);

  auto src_mem = memory(src_md, engine);
  auto weights_mem = memory(weights_md, engine);
  auto bias_mem = memory(bias_md, engine);
  auto dst_mem = memory(dst_md, engine);

  // Write data to memory object's handles.
  write_to_dnnl_memory(src_data.data(), src_mem);
  write_to_dnnl_memory(weights_data.data(), weights_mem);
  write_to_dnnl_memory(bias_data.data(), bias_mem);

  // Create primitive post-ops (ReLU).
  const float alpha = 0.f;
  const float beta = 0.f;
  post_ops matmul_ops;
  matmul_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
  primitive_attr matmul_attr;
  matmul_attr.set_post_ops(matmul_ops);

  // Create primitive descriptor.
  auto matmul_pd = matmul::primitive_desc(
          engine, src_md, weights_md, bias_md, dst_md, matmul_attr);

  // Create the primitive.
  auto matmul_prim = matmul(matmul_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
  matmul_args.insert({DNNL_ARG_DST, dst_mem});

  // Primitive execution: matrix multiplication with ReLU.
  matmul_prim.execute(engine_stream, matmul_args);

  // Wait for the computation to finalize.
  engine_stream.wait();

  // Read data from memory object's handle.
  read_from_dnnl_memory(dst_data.data(), dst_mem);

  return 0;
}
