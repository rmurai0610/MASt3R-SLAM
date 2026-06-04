// Part of this source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
// Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License
//
// Any modifications made are licensed under the CC BY-NC-SA 4.0 License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;


#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)

#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)

#define EPS 1e-6

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;

    SparseBlock(int N, int M) : N(N), M(M) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              tripletList.push_back(T(M*i + k, M*j + l, val));
            }
          }
        }
      }
      A.setFromTriplets(tripletList.begin(), tripletList.end());
    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    torch::Tensor solve(const float lm=0.0, const float ep=0.0) {

      torch::Tensor dx;

      Eigen::SparseMatrix<double> L(A);
      L.diagonal().array() += ep + lm * L.diagonal().array();

      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      if (solver.info() == Eigen::Success) {
        Eigen::VectorXd x = solver.solve(b);
        dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
          .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
      }
      else {
        dx = torch::zeros({N, M}, torch::TensorOptions()
          .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      
      return dx;
    }

  private:
    const int N;
    const int M;

};

torch::Tensor get_unique_kf_idx(torch::Tensor ii, torch::Tensor jj) {
  std::tuple<torch::Tensor, torch::Tensor> unique_kf_idx = torch::_unique(torch::cat({ii,jj}), /*sorted=*/ true);
  return std::get<0>(unique_kf_idx);
}

std::vector<torch::Tensor> create_inds(torch::Tensor unique_kf_idx, const int pin, torch::Tensor ii, torch::Tensor jj) {
  torch::Tensor ii_ind = torch::searchsorted(unique_kf_idx, ii) - pin;
  torch::Tensor jj_ind = torch::searchsorted(unique_kf_idx, jj) - pin;
  return {ii_ind, jj_ind};
}

__forceinline__ __device__ float huber(float r) {
  const float r_abs = fabs(r);
  return r_abs < 1.345 ? 1.0 : 1.345 / r_abs;
}

// Returns qi * qj
__device__ void 
quat_comp(const float *qi, const float *qj, float *out) {
  out[0] = qi[3] * qj[0] + qi[0] * qj[3] + qi[1] * qj[2] - qi[2] * qj[1];
  out[1] = qi[3] * qj[1] - qi[0] * qj[2] + qi[1] * qj[3] + qi[2] * qj[0];
  out[2] = qi[3] * qj[2] + qi[0] * qj[1] - qi[1] * qj[0] + qi[2] * qj[3];
  out[3] = qi[3] * qj[3] - qi[0] * qj[0] - qi[1] * qj[1] - qi[2] * qj[2];
}

// Inverts quat
__device__ void 
quat_inv(const float *q, float *out) {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] =  q[3];
}

__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSim3(const float *t, const float *q, const float *s, const float *X, float *Y) {
  // Rotation
  actSO3(q, X, Y);
  // Scale
  Y[0] *= s[0];
  Y[1] *= s[0];
  Y[2] *= s[0];
  // Translation
  Y[0] += t[0];
  Y[1] += t[1];
  Y[2] += t[2];
}

// Inverts quat
__device__ void 
scale_vec3_inplace(float *t, float s) {
  t[0] *= s;
  t[1] *= s;
  t[2] *= s;
}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__forceinline__ __device__ float
dot3(const float *t, const float *s) {
  return t[0]*s[0] + t[1]*s[1] + t[2]*s[2];
}

__forceinline__ __device__ float
squared_norm3(const float *v) {
  return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

__device__ void 
relSim3(const float *ti, const float *qi, const float* si,
        const float *tj, const float *qj, const float* sj,
        float *tij, float *qij, float *sij) {
  
  // 1. Setup scale
  float si_inv = 1.0/si[0];
  sij[0] = si_inv * sj[0];

  // 2. Relative rotation
  float qi_inv[4];
  quat_inv(qi, qi_inv);
  quat_comp(qi_inv, qj, qij);

  // 3. Translation
  tij[0] = tj[0] - ti[0];
  tij[1] = tj[1] - ti[1];
  tij[2] = tj[2] - ti[2];
  actSO3(qi_inv, tij, tij);
  scale_vec3_inplace(tij, si_inv);
}

// Order of X,Y is tau, omega, s 
// NOTE: This is applying adj inv on the right to a row vector on the left,
// The equivalent is transposing the adjoint and multiplying a column vector
__device__ void
apply_Sim3_adj_inv(const float *t, const float *q, const float *s, const float *X, float *Y) {
  // float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  const float s_inv = 1.0/s[0];

  // First component = s_inv R a
  float Ra[3];
  actSO3(q, &X[0], Ra);
  Y[0] = s_inv * Ra[0];
  Y[1] = s_inv * Ra[1];
  Y[2] = s_inv * Ra[2];
  
  // Second component = s_inv [t]x Ra + Rb
  actSO3(q, &X[3], &Y[3]); // Init to Rb
  Y[3] += s_inv*(t[1]*Ra[2] - t[2]*Ra[1]);
  Y[4] += s_inv*(t[2]*Ra[0] - t[0]*Ra[2]);
  Y[5] += s_inv*(t[0]*Ra[1] - t[1]*Ra[0]);

  // Third component = s_inv t^T R a + c
  Y[6] = X[6] + ( s_inv * dot3(t, Ra) );
}

__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];

  float imag, real;

  if (theta_sq < EPS) {
    float theta_p4 = theta_sq * theta_sq;
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    float theta = sqrtf(theta_sq);
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
expSim3(const float *xi, float* t, float* q, float* s) {
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};
  float sigma = xi[6];

  // New for sim3
  float scale = expf(sigma);

  // 1. Rotation
  expSO3(phi, q);
  // 2. Scale
  s[0] = scale;

  // 3. Translation

  // TODO: Reuse this from expSO3?
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  // Coefficients for W
  https://github.com/princeton-vl/lietorch/blob/0fa9ce8ffca86d985eca9e189a99690d6f3d4df6/lietorch/include/rxso3.h#L190
  
  // TODO: Does this really match equations? Where is scale-1
  float A, B, C;
  const float one = 1.0;
  const float half = 0.5;
  if (fabs(sigma) < EPS) {
    C = one;
    if (fabs(theta) < EPS) {
      A = half;
      B = 1.0/6.0;
    } else {
      A = (one - cosf(theta)) / theta_sq;
      B = (theta - sinf(theta)) / (theta_sq * theta);
    }
  } else {
    C = (scale - one) / sigma;
    if (fabs(theta) < EPS) {
      float sigma_sq = sigma * sigma;
      A = ((sigma - one) * scale + one) / sigma_sq;
      B = (scale * half * sigma_sq + scale - one - sigma * scale) /
          (sigma_sq * sigma);
    } else {
      float a = scale * sinf(theta);
      float b = scale * cosf(theta);
      float c = theta_sq + sigma * sigma;
      A = (a * sigma + (one - b) * theta) / (theta * c);
      B = (C - ((b - one) * sigma + a * theta) / (c)) / (theta_sq); // Why is it C - ????? not +?
    }
  }

  // W = C * I + A * Phi + B * Phi2;
  // t = W tau
  t[0] = C * tau[0]; 
  t[1] = C * tau[1]; 
  t[2] = C * tau[2];

  crossInplace(phi, tau);
  t[0] += A * tau[0];
  t[1] += A * tau[1];
  t[2] += A * tau[2];

  crossInplace(phi, tau);
  t[0] += B * tau[0];
  t[1] += B * tau[1];
  t[2] += B * tau[2];
}

__device__ void
retrSim3(const float *xi, const float* t, const float* q, const float* s, float* t1, float* q1, float* s1) {
  
  // retraction on Sim3 manifold
  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  float ds[1] = {0};
  
  expSim3(xi, dt, dq, ds);

  // Compose transformation from left
  // R
  quat_comp(dq, q, q1);
  // t = ds dR R + ds
  actSO3(dq, t, t1);
  scale_vec3_inplace(t1, ds[0]);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
  // s
  s1[0] = ds[0] * s[0];
}

__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int num_fix) 
{
  const int num_poses = poses.size(0);

  for (int k=num_fix+threadIdx.x; k<num_poses; k+=blockDim.x) {
    float xi[7], q[4], q1[4], t[3], t1[3], s[1], s1[1];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];

    s[0] = poses[k][7];
    
    for (int n=0; n<7; n++) {
      xi[n] = dx[k-num_fix][n];
    }

    retrSim3(xi, t, q, s, t1, q1, s1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];

    poses[k][7] = s1[0];
  }
}

__global__ void point_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_point,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[3];
  float w[3];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
    // Parameters
  const float sigma_point_inv = 1.0/sigma_point;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // Error (difference in camera rays)
    err[0] = Xj_Ci[0] - Xi[0];
    err[1] = Xj_Ci[1] - Xi[1];
    err[2] = Xj_Ci[2] - Xi[2];
 
    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // Weight using confidences
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;
    
    const float sqrt_w_point = valid ? sigma_point_inv * sqrtf(conf_weight) : 0;
 
    // Robust weight
    w[0] = huber(sqrt_w_point * err[0]);
    w[1] = huber(sqrt_w_point * err[1]);
    w[2] = huber(sqrt_w_point * err[2]);
    
    // Add back in sigma
    const float w_const_point = sqrt_w_point * sqrt_w_point;
    w[0] *= w_const_point;
    w[1] *= w_const_point;
    w[2] *= w_const_point;
 
    // Jacobians
    
    // x coordinate
    Ji[0] = 1.0;
    Ji[1] = 0.0;
    Ji[2] = 0.0;
    Ji[3] = 0.0;
    Ji[4] = Xj_Ci[2]; // z
    Ji[5] = -Xj_Ci[1]; // -y
    Ji[6] = Xj_Ci[0]; // x

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // y coordinate
    Ji[0] = 0.0;
    Ji[1] = 1.0;
    Ji[2] = 0.0;
    Ji[3] = -Xj_Ci[2]; // -z
    Ji[4] = 0; 
    Ji[5] = Xj_Ci[0]; // x
    Ji[6] = Xj_Ci[1]; // y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // z coordinate
    Ji[0] = 0.0;
    Ji[1] = 0.0;
    Ji[2] = 1.0;
    Ji[3] = Xj_Ci[1]; // y
    Ji[4] = -Xj_Ci[0]; // -x 
    Ji[5] = 0;
    Ji[6] = Xj_Ci[2]; // z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }
 
 
  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}

std::vector<torch::Tensor> gauss_newton_points_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    point_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_point, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();
    
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}

__global__ void ray_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_ray,
    const float sigma_dist,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[4];
  float w[4];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
    // Parameters
  const float sigma_ray_inv = 1.0/sigma_ray;
  const float sigma_dist_inv = 1.0/sigma_dist;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // Normalize measurement point
    const float norm2_i = squared_norm3(Xi);
    const float norm1_i = sqrtf(norm2_i);
    const float norm1_i_inv = 1.0/norm1_i;    
    
    float ri[3];
    for (int i=0; i<3; i++) ri[i] = norm1_i_inv * Xi[i];
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // Get predicted point norm
    const float norm2_j = squared_norm3(Xj_Ci);
    const float norm1_j = sqrtf(norm2_j);
    const float norm1_j_inv = 1.0/norm1_j;

    float rj_Ci[3];
    for (int i=0; i<3; i++) rj_Ci[i] = norm1_j_inv * Xj_Ci[i];
 
    // Error (difference in camera rays)
    err[0] = rj_Ci[0] - ri[0];
    err[1] = rj_Ci[1] - ri[1];
    err[2] = rj_Ci[2] - ri[2];
    err[3] = norm1_j - norm1_i; // Distance
 
    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // Weight using confidences
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;
    
    const float sqrt_w_ray = valid ? sigma_ray_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_dist = valid ? sigma_dist_inv * sqrtf(conf_weight) : 0;
 
    // Robust weight
    w[0] = huber(sqrt_w_ray * err[0]);
    w[1] = huber(sqrt_w_ray * err[1]);
    w[2] = huber(sqrt_w_ray * err[2]);
    w[3] = huber(sqrt_w_dist * err[3]);
    
    // Add back in sigma
    const float w_const_ray = sqrt_w_ray * sqrt_w_ray;
    const float w_const_dist = sqrt_w_dist * sqrt_w_dist;
    w[0] *= w_const_ray;
    w[1] *= w_const_ray;
    w[2] *= w_const_ray;
    w[3] *= w_const_dist;
 
    // Jacobians
    
    const float norm3_j_inv = norm1_j_inv / norm2_j;
    const float drx_dPx = norm1_j_inv - Xj_Ci[0]*Xj_Ci[0]*norm3_j_inv;
    const float dry_dPy = norm1_j_inv - Xj_Ci[1]*Xj_Ci[1]*norm3_j_inv;
    const float drz_dPz = norm1_j_inv - Xj_Ci[2]*Xj_Ci[2]*norm3_j_inv;
    const float drx_dPy = - Xj_Ci[0]*Xj_Ci[1]*norm3_j_inv;
    const float drx_dPz = - Xj_Ci[0]*Xj_Ci[2]*norm3_j_inv;
    const float dry_dPz = - Xj_Ci[1]*Xj_Ci[2]*norm3_j_inv;
 
    // rx coordinate
    Ji[0] = drx_dPx;
    Ji[1] = drx_dPy;
    Ji[2] = drx_dPz;
    Ji[3] = 0.0;
    Ji[4] = rj_Ci[2]; // z
    Ji[5] = -rj_Ci[1]; // -y
    Ji[6] = 0.0; // x

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // ry coordinate
    Ji[0] = drx_dPy; // same as drx_dPy
    Ji[1] = dry_dPy;
    Ji[2] = dry_dPz;
    Ji[3] = -rj_Ci[2]; // -z
    Ji[4] = 0.0;
    Ji[5] = rj_Ci[0]; // x
    Ji[6] = 0.0; // y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // rz coordinate
    Ji[0] = drx_dPz; // same as drz_dPX
    Ji[1] = dry_dPz; // same as drz_dPy
    Ji[2] = drz_dPz;
    Ji[3] = rj_Ci[1]; // y
    Ji[4] = -rj_Ci[0]; // -x
    Ji[5] = 0.0;
    Ji[6] = 0.0; // z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }


    // dist coordinate
    Ji[0] = rj_Ci[0];
    Ji[1] = rj_Ci[1]; 
    Ji[2] = rj_Ci[2];
    Ji[3] = 0.0; 
    Ji[4] = 0.0; 
    Ji[5] = 0.0;
    Ji[6] = norm1_j;
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[3] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[3] * err[3] * Ji[n];
      vj[n] += w[3] * err[3] * Jj[n];
    }
 
 
  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}

std::vector<torch::Tensor> gauss_newton_rays_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    ray_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_ray, sigma_dist, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    //
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}


__global__ void calib_proj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const int height,
    const int width,
    const int pixel_border,
    const float z_eps,
    const float sigma_pixel,
    const float sigma_depth,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = K[0][0];
    fy = K[1][1];
    cx = K[0][2];
    cy = K[1][2];
  }
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[3];
  float w[3];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // Parameters
  const float sigma_pixel_inv = 1.0/sigma_pixel;
  const float sigma_depth_inv = 1.0/sigma_depth;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];

    // Get measurement pixel
    const int u_target = ind_Xi % width; 
    const int v_target = ind_Xi / width;
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);

    // // Check if in front of camera
    const bool valid_z = ((Xj_Ci[2] > z_eps) && (Xi[2] > z_eps));

    // Handle depth related vars
    const float zj_inv = valid_z ? 1.0/Xj_Ci[2] : 0.0;
    const float zj_log = valid_z ? logf(Xj_Ci[2]) : 0.0;
    const float zi_log = valid_z ? logf(Xi[2]) : 0.0; 

    // Project point
    const float x_div_z = Xj_Ci[0] * zj_inv;
    const float y_div_z = Xj_Ci[1] * zj_inv;
    const float u = fx * x_div_z + cx;
    const float v = fy * y_div_z + cy;

    // Handle proj
    const bool valid_u = ((u > pixel_border) && (u < width - 1 - pixel_border));
    const bool valid_v = ((v > pixel_border) && (v < height - 1 - pixel_border));

    // Error (difference in camera rays)
    err[0] = u - u_target;
    err[1] = v - v_target;
    err[2] = zj_log - zi_log; // Log-depth

    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid =
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh)
      & valid_u & valid_v & valid_z; // Check for valid image and depth
    
    // Weight using confidences
    const float conf_weight = q;
    
    const float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_depth = valid ? sigma_depth_inv * sqrtf(conf_weight) : 0;

    // Robust weight
    w[0] = huber(sqrt_w_pixel * err[0]);
    w[1] = huber(sqrt_w_pixel * err[1]);
    w[2] = huber(sqrt_w_depth * err[2]);
    
    // Add back in sigma
    const float w_const_pixel = sqrt_w_pixel * sqrt_w_pixel;
    const float w_const_depth = sqrt_w_depth * sqrt_w_depth;
    w[0] *= w_const_pixel;
    w[1] *= w_const_pixel;
    w[2] *= w_const_depth;

    // Jacobians    

    // x coordinate
    Ji[0] = fx * zj_inv;
    Ji[1] = 0.0;
    Ji[2] = -fx * x_div_z * zj_inv;
    Ji[3] = -fx * x_div_z * y_div_z;
    Ji[4] = fx * (1 + x_div_z*x_div_z);
    Ji[5] = -fx * y_div_z; 
    Ji[6] = 0.0;

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];


    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }

    // y coordinate
    Ji[0] = 0.0; 
    Ji[1] = fy * zj_inv;
    Ji[2] = -fy * y_div_z * zj_inv;
    Ji[3] = -fy * (1 + y_div_z*y_div_z);
    Ji[4] = fy * x_div_z * y_div_z;
    Ji[5] = fy * x_div_z; 
    Ji[6] = 0.0;

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }

    // z coordinate
    Ji[0] = 0.0; 
    Ji[1] = 0.0; 
    Ji[2] = zj_inv;
    Ji[3] = y_div_z; // y
    Ji[4] = -x_div_z; // -x
    Ji[5] = 0.0;
    Ji[6] = 1.0; // z

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }

  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}


std::vector<torch::Tensor> gauss_newton_calib_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    calib_proj_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      K.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}