# A MATLAB implementation of Ada-kNN, Ada-kNN2, Ada-kNN+GIHS and Ada-kNN2+GIHS.
## Written by: Sankha Subhra Mullick.

### Reference: S. S. Mullick, S. Datta and S. Das, "Adaptive Learning-Based k-Nearest Neighbor Classifiers With Resilience to Class Imbalance," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2018.2812279.
### Contact: mullicksankhasubhra@gmail.com (Sankha Subhra Mullick). 

### DESCRIPTION:
* The package contains 7 functions.
* adaKnn.m: Function implementing Ada-kNN algorithm.
* adaKnn2.m: Function implementing Ada-kNN2 algorithm.
* adaKnnGIHS.m: Function implementing Ada-kNN coupled with GIHS algorithm (for imbalanced classification).
* adaKnn2GIHS.m: Function implementing Ada-kNN coupled with GIHS algorithm (for imbalanced classification).
* kNNIMB.m: Function implementing weighted k-nearest neighbor algorithm.
* learningModel.m: Function implementing the proposed heuristic learning technique (to be used by adaKnn2 and adaKnn2GIHS).
* standardised.m: Supporting function.

### DEPENDENCIES:
* MATLAB 2014a and above.
* Neural Network toolbox (for adaKnn and adaKnnGIHS).


