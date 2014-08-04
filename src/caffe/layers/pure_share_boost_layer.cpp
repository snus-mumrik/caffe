// Copyright 2014 BVLC and contributors.

#include <vector>
#include <float.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
void PureShareBoostLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  const int num_output = this->layer_param_.pure_share_boost_param().num_output();
  num_iterations_per_round_ = this->layer_param_.pure_share_boost_param().share_boost_param().num_iterations_per_round();
  // Figure out the dimensions
  M_ = bottom[0]->num(); // Number of batches
  K_ = bottom[0]->count() / bottom[0]->num(); // Number of inputs (per-batch)
  N_ = num_output; // Number of outputs (per batch) based on the parameters
  CHECK_GE(K_, N_) << "number of outputs higher than the number of inputs!";
  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
  //if (this->active_inputs_.count() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2); // FIXME 2 works and 1 doesn't, says "Check failed: blobs_lr_size == num_param_blobs || blobs_lr_size == 0"
    // Intialize the weight
    // We start with zero actual weights. When the column is activated they will be filled from pre-filled blob
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, K_)); // One weight per input
    memset(this->blobs_[0]->mutable_cpu_data(), 0, K_*sizeof(int));
    this->active_inputs_.Reshape(1, 1, 1, N_); // One index per output, initialization doesn't matter here
    // Init the number of active inputs to 0 accordingly
    num_active_inputs_ = 0;
    n_passes_ = 0;
  }  // parameter initialization
}

template <typename Dtype>
Dtype PureShareBoostLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int* active_inputs = this->active_inputs_.cpu_data();
  // Check we did not (somehow) set the number of selected features higher than the number of outputs
  CHECK_GE(num_active_inputs_, N_) << "number of selected features higher than the number of outputs!";
  // Assign the selected features to the top_data
  for (int idx = 0; idx < num_active_inputs_; idx ++) {
    memcpy(&top_data[idx*M_], &bottom_data[active_inputs[idx]*M_], M_*sizeof(Dtype));
  }
  // Set remaining outputs to 0
  memset(&top_data[num_active_inputs_*M_], 0, (N_ - num_active_inputs_)*M_*sizeof(Dtype));
  return Dtype(0);
}

template <typename Dtype>
void PureShareBoostLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  // If we want to add another feature, choose based on the differences
  Dtype highest_diff = -FLT_MAX;
  int highest_diff_idx = -1;
  if (n_passes_% num_iterations_per_round_ == 0 && N_ > num_active_inputs_) {
    for (int idx = 0; idx < K_; idx ++) { // Run over all inputs
      if (this->blobs_[0]->cpu_data()[idx] == 0){ // if feature not yet selected
        Dtype curr_abs_diff = 0;
        for (int bdx = 0; bdx < M_; bdx ++) // Run ovre all batches
          curr_abs_diff += fabs(top_diff[idx]*bottom_data[idx]);
        if (highest_diff > curr_abs_diff) {
          highest_diff = curr_abs_diff;
          highest_diff_idx = idx;
        }
      }
    }
    CHECK_NE(highest_diff_idx, -1);
    active_inputs_.mutable_cpu_data()[num_active_inputs_] = highest_diff_idx;
    this->blobs_[0]->mutable_cpu_data()[num_active_inputs_] = highest_diff_idx;
    num_active_inputs_ ++;
  }
  // Backpropagate the diff:
  // Set all the diff's to 0 at first
  memset((*bottom)[0]->mutable_cpu_diff(), 0, K_*M_*sizeof(Dtype));
  if (propagate_down[0]) {
    for (int idx = 0; idx < num_active_inputs_; idx ++) {
      memcpy(&(*bottom)[0]->mutable_cpu_diff()[active_inputs_.cpu_data()[idx]*M_], &top_diff[idx*M_], M_*sizeof(Dtype));
    }
  }
  // Count number of (backward) passes
  n_passes_ ++;
}
#ifdef CPU_ONLY
STUB_GPU(PureShareBoostLayer);
#endif

INSTANTIATE_CLASS(PureShareBoostLayer);

}  // namespace caffe
