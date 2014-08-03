#!/usr/bin/env sh

TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe.bin train \
  --solver_proto_file=cifar10_quick_shareboost_solver.prototxt
