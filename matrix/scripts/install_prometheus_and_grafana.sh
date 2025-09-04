#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo "installing prometheus..."
wget https://github.com/prometheus/prometheus/releases/download/v3.1.0/prometheus-3.1.0.linux-amd64.tar.gz
tar -zxvf prometheus-3.1.0.linux-amd64.tar.gz
mv prometheus-3.1.0.linux-amd64 $CONDA_PREFIX/prometheus
rm prometheus-3.1.0.linux-amd64.tar.gz

echo "installing grafana..."
wget https://dl.grafana.com/enterprise/release/grafana-enterprise-11.5.0.linux-amd64.tar.gz
tar -zxvf grafana-enterprise-11.5.0.linux-amd64.tar.gz
mv grafana-v11.5.0 $CONDA_PREFIX/grafana
rm grafana-enterprise-11.5.0.linux-amd64.tar.gz

echo "installation succeed."

echo "create script adding prometheus and grafana to PATH when activate conda env $CONDA_PREFIX"
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
activate_script="$CONDA_PREFIX/etc/conda/activate.d/prometheus_and_grafana_activate.sh"
echo 'export PATH=$CONDA_PREFIX/prometheus:$CONDA_PREFIX/grafana/bin:$PATH' > $activate_script
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
deactivate_script="$CONDA_PREFIX/etc/conda/deactivate.d/prometheus_and_grafana_deactivate.sh"
echo 'export PATH=$(echo $PATH | sed -e "s|$CONDA_PREFIX/prometheus:$CONDA_PREFIX/grafana/bin:||g")' > $deactivate_script

if [ ! -f $activate_script ]; then
    echo "failed to create $activate_script, prometheus and grafana command won't work"
    exit 1
fi

if [ ! -f $deactivate_script ]; then
    echo "failed to create $deactivate_script, prometheus and grafana won't be removed from path when deactivate"
    exit 1
fi

echo "all succeed! Please re-activate your conda env."
