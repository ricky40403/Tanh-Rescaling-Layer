#include <vector>
#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/tanh_scale_layer.hpp"

namespace caffe {

template <typename Dtype>
void TanhScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

}


template <typename Dtype>
void TanhScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    tanh_x.ReshapeLike(*bottom[0]);
    top[0]->ReshapeLike(*bottom[0]);
    
}

template <typename Dtype>
void TanhScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* tanhx_ = tanh_x.mutable_cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
        tanhx_[i] = tanh(bottom_data[i]);
        top_data[i] = (2 + tanhx_[i]) * bottom_data[i];
    }
}

template <typename Dtype>
void TanhScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* tanhx_ = tanh_x.cpu_data();
    

    if (propagate_down[0]) {
        const int count = bottom[0]->count();
        for (int i = 0; i < count; ++i) {          
          bottom_diff[i] = top_diff[i] * (2 + tanhx_[i] + bottom_data[i]*(1- powf(tanhx_[i], 2)));
        }
    }
    

}
INSTANTIATE_CLASS(TanhScaleLayer);
REGISTER_LAYER_CLASS(TanhScale);

}