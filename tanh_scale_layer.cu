#include <vector>
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/tanh_scale_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void ScaleFowardGpu(int nthreads,
          const Dtype* bottom_data, Dtype* top_data,
          Dtype* tanhx_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    tanhx_[index] = tanh(bottom_data[index]);
    top_data[index] = (2 + tanhx_[index]) * bottom_data[index];
  }
}



template <typename Dtype>
void TanhScaleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();  
  Dtype* top_data = top[0]->mutable_gpu_data();  
  Dtype* tanhx_ = tanh_x.mutable_gpu_data();

  /************************* normalize *************************/

  int nthreads = bottom[0]->count();
  ScaleFowardGpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads,                               
                                bottom_data, top_data,
                                tanhx_);
  
}

template <typename Dtype>
__global__ void ScaleBackwardGPU(int nthreads,
const Dtype* top_diff, 
const Dtype* bottom_data, Dtype* bottom_diff,
const Dtype* tanhx_){

  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = top_diff[index] * (2 + tanhx_[index] + bottom_data[index] * (1- powf(tanhx_[index], 2)));
  }
}

template <typename Dtype>
void TanhScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->cpu_diff();    
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* tanhx_ = tanh_x.mutable_gpu_data();


	if (propagate_down[0]) {    
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();    
		int nthreads = bottom[0]->count();
    ScaleBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, 
    top_diff,
    bottom_data, bottom_diff,
    tanhx_);

	}
  
}

INSTANTIATE_LAYER_GPU_FUNCS(TanhScaleLayer);

}