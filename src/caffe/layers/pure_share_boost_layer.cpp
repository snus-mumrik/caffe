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
	num_iterations_per_round_ = this->layer_param_.pure_share_boost_param().share_boost_param().num_iterations_per_round();
	max_num_of_rounds_ = this->layer_param_.pure_share_boost_param().share_boost_param().max_num_of_rounds();
	weight_off_value_ = this->layer_param_.pure_share_boost_param().weight_off_value();
	random_test_ = this->layer_param_.pure_share_boost_param().share_boost_param().random_test();
	// Figure out the dimensions
	M_ = bottom[0]->num(); // Number of batches
	K_ = bottom[0]->count() / bottom[0]->num(); // Number of inputs (per-batch)
	N_ = K_; // Number of outputs (per batch) based on the parameters
	//  const int num_output = this->layer_param_.pure_share_boost_param().num_output();
	//  CHECK_EQ(K_, N_) << "number of outputs must be equal to number of inputs!";
	CHECK_GE(K_, max_num_of_rounds_) << "max_num_of_rounds can not be greater than number of inputs!";
	(*top)[0]->Reshape(M_, N_, 1, 1);
	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		//if (this->active_inputs_.count() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		this->blobs_.resize(1);
		// Intialize the weight
		// We start with zero actual weights. When the column is activated they will be filled from pre-filled blob
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, K_)); // One weight per input
		// Init the number of active inputs to 0 accordingly
		num_active_inputs_ = 0;
		n_passes_ = 0;
	}  // parameter initialization
}

template <typename Dtype>
Dtype PureShareBoostLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* weights = this->blobs_[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	if (n_passes_ == 0)
		memset(this->blobs_[0]->mutable_cpu_data(), 0, K_*sizeof(int));
	// Check we did not (somehow) set the number of selected features higher than the number of outputs
	CHECK_GE(N_, num_active_inputs_) << "number of selected features higher than the number of outputs!";
	// Assign the selected features to the top_data
#ifndef __STDC_IEC_559__
	LOG(FATAL) << "invalid use of memset, consider using for instead!";
	exit(0);
#endif
	for (int idx = 0; idx < K_; idx ++) {
		if (weights[idx] > 0) {
			for (int bdx = 0; bdx < M_; bdx ++)
				top_data[idx*M_ + bdx] = bottom_data[idx*M_ + bdx];
		} else {
			for (int bdx = 0; bdx < M_; bdx ++)
				top_data[idx*M_ + bdx] = weight_off_value_*bottom_data[idx*M_ + bdx];
		}
	}
	return Dtype(0);
}

template <typename Dtype>
void PureShareBoostLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	Dtype* weights = this->blobs_[0]->mutable_cpu_data();
	if (n_passes_ == 0)
		memset(weights, 0, K_*sizeof(int));
	if (random_test_) {
		if (n_passes_ == 0) {
			for (int idx = 0; idx < max_num_of_rounds_; idx ++)
				weights[idx] = 1.0f;
			for (int idx = max_num_of_rounds_; idx < K_; idx ++)
				weights[idx] = 0.0f;
		}
	} else {
		// If we want to add another feature, choose based on the differences
		Dtype highest_diff = -FLT_MAX;
		int highest_diff_idx = -1;
		if (n_passes_% num_iterations_per_round_ == 0 && max_num_of_rounds_ > num_active_inputs_) {
			LOG(INFO) << "adding the " <<  num_active_inputs_ << " feature";
			for (int idx = 0; idx < K_; idx ++) { // Run over all inputs
				if (weights[idx] == 0){ // if feature not yet selected
					Dtype curr_abs_diff = 0;
					for (int bdx = 0; bdx < M_; bdx ++){ // Run ovre all batches
						curr_abs_diff += fabs(top_diff[idx*M_ + bdx]*bottom_data[idx*M_ + bdx]);
					}
					if (curr_abs_diff > highest_diff) {
						highest_diff = curr_abs_diff;
						highest_diff_idx = idx;
					}
				}
			}
			CHECK_NE(highest_diff_idx, -1);
			weights[highest_diff_idx] = 1.0f;
			num_active_inputs_ ++;
		}
	}
	// Backpropagate the diff:
	// Set all the diff's to 0 at first
	//memset((*bottom)[0]->mutable_cpu_diff(), 0, K_*M_*sizeof(Dtype));
	if (propagate_down[0]) {
		// Back propagate the diffs
		for (int idx = 0; idx < K_; idx ++) {
			if (weights[idx] > 0) {
				for (int bdx = 0; bdx < M_; bdx ++)
					(*bottom)[0]->mutable_cpu_diff()[idx*M_ + bdx] = top_diff[idx*M_ + bdx];
			} else {
				for (int bdx = 0; bdx < M_; bdx ++)
					(*bottom)[0]->mutable_cpu_diff()[idx*M_ + bdx] = weight_off_value_ * top_diff[idx*M_ + bdx];
			}
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
