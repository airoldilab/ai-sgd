## AI-SGD for Support Vector Machines

The files in this directory implement AI-SGD for linear SVMs in C++. We use
[Leon Bottou's code base](http://leon.bottou.org/projects/sgd), where more
instructions on compiling can be found there. To use the files in this
directory, copy the files from here to the `svm/` directory in Bottou's
repository.

After downloading and preprocessing the necessary dataset(s), you should then be able to compile and run the program:
```
make svmaisgd
./svmaisgd -lambda 5e-7 -epochs 8 -avgstart 1 rcv1.train.bin.gz rcv1.test.bin.gz
```

## RCV1 Benchmark
> The benchmark task is the recognition of RCV1 documents belonging to the class CCAT. Program “prep_rcv1” reads the RCV1-V2 token files from directory “data/rcv1” and computes the TF/IDF features on the basis of the our training set. This programs produces the two files “rcv1.train.bin.gz” and “rcv1.test.bin.gz” containing our training and testing sets. This short program also generates files in SvmLight format when compiled with option -DPREP_SVMLIGHT. [(Bottou)](http://leon.bottou.org/projects/sgd)

| Benchmark | Features | Training examples | Testing examples |
| :---- | :----: | :----: | :----: |
| RCV1 | 47152 | 781265 | 23149 |

For these benchmarks, `svmimplicit` uses the same learning rate procedure as `svmsgd`, and `svmaisgd` uses the same learning rate procedure as `svmasgd`.

| Algorithm (hinge loss, λ=1e-4) | Training Time\* | Primal cost | Test Error |
| :---- | ----: | ----: | ----: |
| SMO ([`SVMLight`](http://svmlight.joachims.org/)) | ≃ 16000 secs<sup>1</sup> | 0.2275 | 6.02% |
| Cutting Plane ([`SVMPerf`](http://www.cs.cornell.edu/People/tj/svm_light/svm_perf.html)) | ≃ 45 secs<sup>2</sup> | 0.2278 | 6.03% |
| Hinge Loss SDCA ([`LibLinear -s 3 -B 1`](http://www.csie.ntu.edu.tw/~cjlin/liblinear)) | 2.5 secs | - | 6.02% |
| SGD (`svmsgd`) | < 1 sec. | 0.2275 | 6.02% |
| ASGD (`svmasgd`) | < 1 sec. | 0.2275 | 6.02% |
| Implicit SGD ([`svmimplicit`](http://www.people.fas.harvard.edu/~ptoulis/harvard-homepage/implicit-sgd.html)) | < 1 sec. | 0.2275 | 6.04% |
| AISGD (`svmaisgd`) | < 1 sec. | 0.2276 | 6.03% |

<sup>1</sup> Extrapolated from a 23642 seconds run on a 30% slower machine.

<sup>2</sup> Extrapolated from a 66 seconds run on a 30% slower machine.

\* All timings exclude data loading time.

| Algorithm (log loss, λ=5e-7) | Training Time | Primal cost | Test Error |
| :---- | ----: | ----: | ----: |
| TRON ([`LibLinear -s 0 -B 1`](http://www.csie.ntu.edu.tw/~cjlin/liblinear)) | 33 secs | - | 5.14% |
| SDCA ([`LibLinear -s 7 -B 1`](http://www.csie.ntu.edu.tw/~cjlin/liblinear)) | 15 secs | - | 5.13% |
| SGD (`svmsgd`) | 4 secs | 0.1283 | 5.14% |
| ASGD (`svmasgd`) | 5 secs | 0.1281 | 5.13% |
| Implicit SGD ([`svmimplicit`](http://www.people.fas.harvard.edu/~ptoulis/harvard-homepage/implicit-sgd.html)) | 4 secs | 0.1375 | 5.20% |
| AISGD (`svmaisgd`) | 3 secs | 0.1307 | 5.24% |
