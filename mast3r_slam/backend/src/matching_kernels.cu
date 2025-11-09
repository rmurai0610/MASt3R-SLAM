#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <cuda/std/limits>

#define BLOCK 16

__forceinline__ __device__ bool inside_image(int u, int v, int W, int H) {
  return v >= 0 && v < H && u >= 0 && u < W;
}

__forceinline__ __device__ void clamp(float& x, const float min, const float max) {
  x = fmin(fmax(x, min), max);
}

template <typename scalar_t>
__global__ void refine_matches_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> D11,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> D21,
    const torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> p1,
    torch::PackedTensorAccessor32<long,3,torch::RestrictPtrTraits> p1_new,
    const int radius,
    const int dilation_max
    )
{
  // batch index
  const uint64_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t b = blockIdx.y;

  const int h = D11.size(1);
  const int w = D11.size(2);
  const int fdim = D11.size(3);

  // Get pixel and its features
  long u0 = p1[b][n][0];
  long v0 = p1[b][n][1];

  scalar_t max_score = ::cuda::std::numeric_limits<scalar_t>::min();
  long u_new = u0;
  long v_new = v0;

  for (int d=dilation_max; d>0; d--) {
    const int rd = radius*d;
    const int diam = 2*rd + 1;
    for (int i=0; i<diam; i+=d) {
      for (int j=0; j<diam; j+=d) {
        const long u = u0 - rd + i;
        const long v = v0 - rd + j;

        if (inside_image(u, v, w, h)) {
          scalar_t score = 0.0;
          for (int k=0; k<fdim; k++) {
            score += D21[b][n][k] * D11[b][v][u][k];
          }

          if (score > max_score) {
            max_score = score;
            u_new = u;
            v_new = v;
          }
    
        }
      }
    }
    // Update where search is centered from previous update
    u0 = u_new;
    v0 = v_new;
  }

  p1_new[b][n][0] = u_new;
  p1_new[b][n][1] = v_new;
}


std::vector<torch::Tensor> refine_matches_cuda(
    torch::Tensor D11,
    torch::Tensor D21,
    torch::Tensor p1,
    const int radius,
    const int dilation)
{
  const auto batch_size = p1.size(0);
  const auto n = p1.size(1);

  const dim3 blocks((n + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK);

  auto opts = p1.options();
  torch::Tensor p1_new = torch::zeros(
    {batch_size, n, 2}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(D11.scalar_type(), "refine_matches_kernel", ([&] {
    refine_matches_kernel<scalar_t><<<blocks, threads>>>(
      D11.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      D21.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
      p1.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
      p1_new.packed_accessor32<long,3,torch::RestrictPtrTraits>(),
      radius,
      dilation
    );
   }));

  return {p1_new};

}


__global__ void iter_proj_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> rays_img,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> pts_3d_norm,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> p_init,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> p_new,
    torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> converged,
    const int max_iter,
    const float lambda_init,
    const float cost_thresh
    )
{
  // batch index
  const uint64_t n = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t b = blockIdx.y;

  const int h = rays_img.size(1);
  const int w = rays_img.size(2);
  const int c = rays_img.size(3); // 9

  // Get pixel
  float u = p_init[b][n][0];
  float v = p_init[b][n][1];

  // Clamp init if outisde
  clamp(u, 1, w-2);
  clamp(v, 1, h-2);

  // Setup rays and gradients
  float r[3];
  float gx[3];
  float gy[3];
  float err[3];

  float lambda = lambda_init;
  for (int i=0; i<max_iter; i++) {
    // Bilinear interpolation
    int u11 = static_cast<int>(floor(u));
    int v11 = static_cast<int>(floor(v));
    float du = u - static_cast<float>(u11);
    float dv = v - static_cast<float>(v11);

    // Clamping always ensures full bilinear is fine to calculate
    float w11 = du * dv; // top left
    float w12 = (1.0-du) * dv; // top right
    float w21 = du * (1.0-dv); // bottom left
    float w22 = (1.0-du) * (1.0-dv); // bottom right

    // NOTE: Pixels are opposite the area calc!
    float const* r11 = &rays_img[b][v11+1][u11+1][0]; // bottom right
    float const* r12 = &rays_img[b][v11+1][u11][0]; // bottom left
    float const* r21 = &rays_img[b][v11][u11+1][0]; // top right
    float const* r22 = &rays_img[b][v11][u11][0]; // top left

    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }
    #pragma unroll
    for (int j=3; j<6; j++) {
      gx[j-3] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }
    #pragma unroll
    for (int j=6; j<9; j++) {
      gy[j-6] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }

    // Normalize ray
    float r_norm = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    float r_norm_inv = 1.0/r_norm;
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] *= r_norm_inv;
    }

    // Calculate error
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j];
    }
    float cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

    // Setup system
    // J^T J
    float A00 = gx[0]*gx[0] + gx[1]*gx[1] + gx[2]*gx[2];
    float A01 = gx[0]*gy[0] + gx[1]*gy[1] + gx[2]*gy[2];
    float A11 = gy[0]*gy[0] + gy[1]*gy[1] + gy[2]*gy[2];
    // - J^T r
    float b0 = - (err[0]*gx[0] + err[1]*gx[1] + err[2]*gx[2]);
    float b1 = - (err[0]*gy[0] + err[1]*gy[1] + err[2]*gy[2]);
    // LM diagonal
    A00 += lambda;
    A11 += lambda;

    // Solve system
    float det_inv = 1.0/(A00*A11 - A01*A01);
    float delta_u = det_inv * ( A11*b0 - A01*b1);
    float delta_v = det_inv * (-A01*b0 + A00*b1);

    // Get new pixel
    float u_new = u + delta_u;
    float v_new = v + delta_v;
    clamp(u_new, 1, w-2);
    clamp(v_new, 1, h-2);


    // Test new cost
    u11 = static_cast<int>(floor(u_new));
    v11 = static_cast<int>(floor(v_new));
    du = u_new - u11;
    dv = v_new - v11;

    w11 = du * dv; // top left
    w12 = (1.0-du) * dv; // top right
    w21 = du * (1.0-dv); // bottom left
    w22 = (1.0-du) * (1.0-dv); // bottom right

    // NOTE: Pixels are opposite the area calc!
    r11 = &rays_img[b][v11+1][u11+1][0]; // bottom right
    r12 = &rays_img[b][v11+1][u11][0]; // bottom left
    r21 = &rays_img[b][v11][u11+1][0]; // top right
    r22 = &rays_img[b][v11][u11][0]; // top left

    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] = w11*r11[j] + w12*r12[j] + w21*r21[j] + w22*r22[j];
    }
    r_norm = sqrtf(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    r_norm_inv = 1.0/r_norm;
    #pragma unroll
    for (int j=0; j<3; j++) {
      r[j] *= r_norm_inv;
    }
    // Calculate error
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j];
    }
    float new_cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

    // Update pixel and lambda
    if (new_cost < cost) {
      u = u_new;
      v = v_new;
      lambda *= 0.1;
      converged[b][n] = new_cost < cost_thresh;
    }
    else {
      lambda *= 10.0;
      converged[b][n] = cost < cost_thresh;
    }

  }

  p_new[b][n][0] = u;
  p_new[b][n][1] = v;

}



std::vector<torch::Tensor> iter_proj_cuda(
    torch::Tensor rays_img_with_grad,
    torch::Tensor pts_3d_norm,
    torch::Tensor p_init,
    const int max_iter,
    const float lambda_init,
    const float cost_thresh)
{
  const auto batch_size = p_init.size(0);
  const auto n = p_init.size(1);

  const dim3 blocks((n + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK);

  auto opts = p_init.options();
  torch::Tensor p_new = torch::zeros(
    {batch_size, n, 2}, opts);

  auto opts_bool = opts.dtype(torch::kBool);
  torch::Tensor converged = torch::zeros(
    {batch_size, n}, opts_bool);

  iter_proj_kernel<<<blocks, threads>>>(
    rays_img_with_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    pts_3d_norm.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    p_init.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    p_new.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    converged.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
    max_iter,
    lambda_init,
    cost_thresh
  );

  return {p_new, converged};

}