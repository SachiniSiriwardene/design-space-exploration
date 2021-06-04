"""
 Copyright (c) 2019, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 
  WSO2 Inc. licenses this file to you under the Apache License,
  Version 2.0 (the "License"); you may not use this file except
  in compliance with the License.
  You may obtain a copy of the License at
 
  http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
"""

import numpy as np


class __RandomDist:
    def __init__(self, low, high):
        if low >= high:
            raise Exception("low should be lower than high")
        self.low = low
        self.high = high


class RandomInt(__RandomDist):

    def sample(self, size):
        return np.random.randint(self.low, self.high, (size, 1))

    def __init__(self, low, high, decimals=None):
        super().__init__(low, high)


class RandomFloat(__RandomDist):

    def __init__(self, low, high, decimals=None):
        super().__init__(low, high)
        self.decimals = decimals

    def sample(self, size):
        decimals = self.decimals
        sample = np.random.uniform(self.low, self.high, (size, 1))
        if decimals is not None:
            if decimals >= 0:
                sample = np.round(sample, decimals=decimals)
            else:
                raise Exception('decimals should be greater than or equals to 0')
        return sample
