# Wave-ETKF

[![Status](https://img.shields.io/badge/status-coming%20soon-yellow.svg)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Coming Soon!

We're excited to announce that our paper "From Points to Waves: Fast Ocean Wave Spatial-temporal Fields Estimation using Ensemble Transform Kalman Filter with Optical Measurement" has been accepted for publication in Coastal Engineering.

### Paper Information
- **Title**: From Points to Waves: Fast Ocean Wave Spatial-temporal Fields Estimation using Ensemble Transform Kalman Filter with Optical Measurement
- **Journal**: Coastal Engineering
- **Status**: Accepted

### Project Structure
This project consists of two main parts:
1. Vision-based measurements and **PARALLEL** wave prediction modules (adapted from WASSFAST)
2. Original ETKF implementation for wave field estimation (proprietary)

### Current Status
- The adapted WASSFAST code includes stereo vision processing pipeline and wave field prediction algorithms
- The Kalman Filter implementation is currently proprietary and pending software copyright registration
- Full open-source release will be considered after copyright registration

### Code Attribution
The sparse reconstruction and wave prediction modules are adapted from:
- Project: WASSFAST (Wave Acquisition Stereo System - Fast Analysis and Spatiotemporal Toolbox)
- Source: https://gitlab.com/fibe/wassfast
- Original Author: Filippo Bergamasco
- License: GNU General Public License v3.0

### Future Plans
- Complete software copyright registration for the Kalman Filter implementation
- Consider full open-source release
- Maintain modular structure for flexible integration of different prediction and measurement methods

### Stay Tuned
We will release:
- Implementation of the Wave-ETKF algorithm
- Testing datasets
- Example configurations

### Contact

Feng Wang - [wfeng@hrbeu.edu.cn]