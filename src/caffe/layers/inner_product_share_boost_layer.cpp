// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
void InnerProductShareBoostLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  const int num_output = this->layer_param_.inner_product_share_boost_param().inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_share_boost_param().inner_product_param().bias_term();
  num_iterations_per_round_ = this->layer_param_.inner_product_share_boost_param().share_boost_param().num_iterations_per_round();
  n_passes_ = 0;
  // Figure out the dimensions
  elements_per_feature_ = this->layer_param_.inner_product_share_boost_param().share_boost_param().num_elements_per_feature();
  if (elements_per_feature_ == 0) {
    elements_per_feature_ = bottom[0]->width() * bottom[0]->height();
  }
  M_ = bottom[0]->num();
  K_ = bottom[0]->count() / bottom[0]->num();
  N_ = num_output;
  num_input_features_ = K_ / elements_per_feature_;
  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    // We start with zero actual weights. When the column is activated they will be filled from pre-filled blob
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
    caffe_set(N_*K_, Dtype(0), this->blobs_[0]->mutable_cpu_data());
    weigth_fill_.Reshape(1, 1, N_, K_);
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_share_boost_param().inner_product_param().weight_filler()));
    weight_filler->Fill(&weigth_fill_);
    
    num_active_features_ = 0;
    active_features_.Reshape(1, 1, 1, num_input_features_);
    caffe_set(num_input_features_, 0, active_features_.mutable_cpu_data());
    // If necessary, initialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_share_boost_param().inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, M_);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
Dtype InnerProductShareBoostLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void InnerProductShareBoostLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());
  }
  if ((num_active_features_ == 0) || 
      ((num_active_features_ < num_input_features_) &&
      (n_passes_ % num_iterations_per_round_ == 0))) { // TODO Could be zero-gradient check
    // Find and activate a new column in matrix
    int best_k = -1;
    if (this->layer_param_.inner_product_share_boost_param().share_boost_param().choose_at_random()) {
      best_k = num_active_features_;
      CHECK_EQ(active_features_.cpu_data()[best_k], 0);
    } else {
      Dtype max_L1 = -1.0;
      // TODO use blas for faster calculation
      const Dtype* weights_diff = this->blobs_[0]->cpu_diff();
      for (int k = 0; k < num_input_features_; k++) {
	if (active_features_.cpu_data()[k])
	  continue;
	Dtype cur_L1 = 0.0;
	for (int n = 0; n < N_; n++) { // No need to iterate over images in batch because weigths_diff is calculated for the whole batch
	  for (int edx = 0; edx < elements_per_feature_; edx++) {
	    cur_L1 += fabs(weights_diff[n*K_ + k*elements_per_feature_ + edx]);
	  }
	}
	if (cur_L1 > max_L1) {
	  // This is the new candidate
	  max_L1 = cur_L1;
	  best_k = k;
	}
      }
    }

    CHECK_GE(best_k, 0); // FIXME sanity check, not really needed
    std::cerr << "activaiting k: " << best_k << '\n';
    num_active_features_++;
    active_features_.mutable_cpu_data()[best_k] = 1;
    Dtype* weights = this->blobs_[0]->mutable_cpu_data();
    const Dtype* weigth_fill = weigth_fill_.cpu_data();
    // Activate the column - fill weights that were zeroes till now
    for (int n = 0; n < N_; n++) {
      for (int edx = 0; edx < elements_per_feature_; edx++) {
	weights[n*K_ + best_k*elements_per_feature_ + edx] = weigth_fill[n*K_ + best_k*elements_per_feature_ + edx]; // TODO mem copy
      }
    }
  } // Find and activate a new column in matrix
  
  // Zero gradient of inactive columns
  Dtype* weights_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bias_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int k = 0; k < num_input_features_; k++) {
    if (!active_features_.cpu_data()[k]) {
      for (int n = 0; n < N_; n++) {
	for (int edx = 0; edx < elements_per_feature_; edx++) {
	  weights_diff[n*K_ + k*elements_per_feature_ + edx] = 0.0;
	  bias_diff[n*K_ + k*elements_per_feature_ + edx] = 0.0;
	}
      }
    }
  }
  
  n_passes_++;
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductShareBoostLayer);
#endif

INSTANTIATE_CLASS(InnerProductShareBoostLayer);

}  // namespace caffe
