# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from matrix.utils.basics import sanitize_app_name


def test_sanitize_app_name():
    assert sanitize_app_name("meta-llama/Llama-3.1-8B") == "meta-llama-Llama-3.1-8B"
    assert sanitize_app_name("model") == "model"
    assert sanitize_app_name("a/b/c") == "a-b-c"
    assert sanitize_app_name("foo/bar/baz") == "foo-bar-baz"
    assert sanitize_app_name("/leading") == "leading"
    assert sanitize_app_name("trailing/") == "trailing"