LAML (Linear Algebra and Machine Learning)

LAML is a stand-alone pure Java library for linear algebra and machine learning. The goal is to build efficient and easy-to-use linear algebra and machine learning libraries. The reason why linear algebra and machine learning are built together is that full control of the basic data structures for matrices and vectors is required to have fast implementation for machine learning methods. Additionally, LAML provides a lot of commonly used matrix functions in the same signature to MATLAB, thus can also be used to manually convert MATLAB code to Java code.

The built-in linear algebra library supports well-designed dense and sparse matrices and vectors. Standard compressed sparse column (CSC) and compressed sparse row (CSR) are used to design and implement sparse matrices. The matrix multiplication is competitive or even faster than well known linear algebra libraries in Java such as Commons-Math and Colt. Unlike other linear algebra libraries in Java, the built-in linear algebra library in LAML gives users full control of sparse matrices and vectors (e.g., the interior arrays in sparse matrices), which is crucial to make efficient high level implementations.

Carefully designed linear algebra library is the basis for machine learning library. ML library aims to provide fast implementation of mature machine learning methods. For instance, the LinearBinarySVM class re-implements the popular liblinear (in Java). For heart_scale data with C = 1.0 and eps = 1e-2, the average running time is 0.04 seconds using an Intel(R) Core(TM) i7 CPU M620 @ 2.67GHz with 4.00GB memory and 64-bit Windows 7 operating system, even a little faster than liblinear, which costs 0.06 seconds in average given the same parameter.

JML v.s. LAML
LAML is much faster than JML (more than 3 times faster) due to two implementation considerations. First, LAML allows full control of dense and sparse matrices and vectors. Second, LAML extensively uses in-place matrix and vector operations thus avoids too much memory allocation and garbage collection.

JML relies on third party linear algebra library, i.e. Apache Commons-math. Sparse matrices and vectors have been deprecated in Commons-math 3.0+, and will be ultimately eliminated. Whereas LAML has its own built-in linear algebra library.

Like JML, LAML also provides a lot of commonly used matrix functions in the same signature to MATLAB, thus can also be used to manually convert MATLAB code to Java code.

In short, JML has been replaced by LAML.

SourceForge
https://sourceforge.net/projects/lamal

Github:
https://github.com/MingjieQian/LAML

Documentation
http://web.engr.illinois.edu/~mqian2/upload/projects/java/LAML/doc/index.html

Features
Stand-alone Java library, completely cross-platform
Built-in Linear Algebra (LA) library
Full control of matrices and vectors
Many general-purpose optimization algorithms
Fast implementation of Machine Learning (ML) methods
Matrix functions with almost the same signature to MATLAB
Well documented source code and friendly API, very easy to use

Packages
la.decomposition
	LU, QR, eigenvalue decomposition, and SVD
la.matrix
	Sparse and dense matrix implementation
la.vector
	Sparse and dense vector implementation
la.io
	Functions of saving and loading a matrix, a vector, or a data set
ml.utils
	Efficient functions for array and matrix operations, Matlab-style functions, and Printer 
ml.classification
	Linear SVM, Linear multi-class SVM, regularized logistic regression, maximum entropy modeling, and AdaBoost
ml.clustering
	K-means, L1NMF, NMF, and spectral clustering
ml.optimization
	L-BFGS, BoundConstrainedPLBFGS, NonnegativePLBFGS, Projection, ProximalMapping, ShrinkageOperator, accelerated proximal gradient, accelerated gradient descent, general quadratic programming, nonlinear conjugate gradient, LBFGS on simplex, quadratic programming with bound constraint, primal-dual interior-point method
ml.sequence
	Hidden Markov Models (HMM) and Conditional Random Fields (CRF)
ml.kernel
	Commonly used kernel functions ('linear' | 'poly' | 'rbf' | 'cosine')
ml.manifold
	Commonly used manifold learning functions such as computing adjacency matrix, Laplacian matrix, and local learning regularization matrix
ml.subspace
	PCA, kernel PCA, Multi-dimensional Scaling (MDS), Isomap, and Locally Linear Embedding (LLE)
ml.regression
	LASSO and linear regression
ml.random
	Multivariate Gaussian distribution
ml.recovery
	Matrix completion and robust PCA
ml.topics
	LDA
ml.graph
	Minimum spanning tree using Prim's algorithm, shortest path using Dijkstra's algorithm, topological order, all-pairs shortest path using Floyd-Warshall algorithm, Huffman codes, and maximum flow using Ford-Fulkerson algorithm.
ml.recommendation
	Factorization machines and structured sparse regression (STSR)
	
Code Examples

# Eigenvalue Decomposition (For Real Symmetric Matrices)

int m = 4;
int n = 4;
Matrix A = hilb(m, n);

fprintf("A:%n");
disp(A);
long start = 0;
start = System.currentTimeMillis();
Matrix[] VD = EigenValueDecomposition.decompose(A);
System.out.format("Elapsed time: %.4f seconds.%n", (System.currentTimeMillis() - start) / 1000.0);
fprintf("*****************************************%n");

Matrix V = VD[0];
Matrix D = VD[1];

fprintf("V:%n");
printMatrix(V);

fprintf("D:%n");
printMatrix(D);

fprintf("VDV':%n");
disp(V.mtimes(D).mtimes(V.transpose()));

fprintf("A:%n");
printMatrix(A);

fprintf("V'V:%n");
printMatrix(V.transpose().mtimes((V)));

# Output

A:
         1    0.5000    0.3333    0.2500  
    0.5000    0.3333    0.2500    0.2000  
    0.3333    0.2500    0.2000    0.1667  
    0.2500    0.2000    0.1667    0.1429  

Elapsed time: 0.0100 seconds.
*****************************************
V:
    0.7926    0.5821   -0.1792   -0.0292  
    0.4519   -0.3705    0.7419    0.3287  
    0.3224   -0.5096   -0.1002   -0.7914  
    0.2522   -0.5140   -0.6383    0.5146  

D:
    1.5002  
              0.1691  
                        0.0067  
                                  0.0001  

VDV':
         1    0.5000    0.3333    0.2500  
    0.5000    0.3333    0.2500    0.2000  
    0.3333    0.2500    0.2000    0.1667  
    0.2500    0.2000    0.1667    0.1429  

A:
         1    0.5000    0.3333    0.2500  
    0.5000    0.3333    0.2500    0.2000  
    0.3333    0.2500    0.2000    0.1667  
    0.2500    0.2000    0.1667    0.1429  

V'V:
    1.0000   -0.0000   -0.0000   -0.0000  
   -0.0000    1.0000    0.0000         0  
   -0.0000    0.0000    1.0000   -0.0000  
   -0.0000         0   -0.0000    1.0000

# -------------------------------------------------------------------------- #
   
# LU Decomposition

double[][] data = new double[][] {
		{1, -2, 3},
		{2, -5, 12},
		{0, 2, -10}
};
Matrix A = new DenseMatrix(data);
fprintf("A:%n");
printMatrix(A);

Matrix[] LUP = LUDecomposition.decompose(A);
Matrix L = LUP[0];
Matrix U = LUP[1];
Matrix P = LUP[2];

fprintf("L:%n");
printMatrix(L);

fprintf("U:%n");
printMatrix(U);

fprintf("P:%n");
printMatrix(P);

fprintf("PA:%n");
printMatrix(P.mtimes(A));

fprintf("LU:%n");
printMatrix(L.mtimes(U));

long start = 0;
start = System.currentTimeMillis();

LUDecomposition LUDecomp = new LUDecomposition(A);
Vector b = new DenseVector(new double[] {2, 3, 4});
Vector x = LUDecomp.solve(b);
fprintf("Solution for Ax = b:%n");
printVector(x);
fprintf("b = %n");
printVector(b);
fprintf("Ax = %n");
printVector(A.operate(x));

fprintf("A^{-1}:%n");
printMatrix(LUDecomp.inverse());

fprintf("det(A) = %.2f%n", LUDecomp.det());
System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);
fprintf("**********************************%n");

A = sparse(A);
fprintf("A:%n");
printMatrix(A);

LUP = LUDecomposition.decompose(A);
L = LUP[0];
U = LUP[1];
P = LUP[2];

fprintf("L:%n");
printMatrix(L);

fprintf("U:%n");
printMatrix(U);

fprintf("P:%n");
printMatrix(P);

fprintf("PA:%n");
printMatrix(P.mtimes(A));

fprintf("LU:%n");
printMatrix(L.mtimes(U));

start = System.currentTimeMillis();

LUDecomp = new LUDecomposition(sparse(A));
b = new DenseVector(new double[] {2, 3, 4});
x = LUDecomp.solve(b);
fprintf("Solution for Ax = b:%n");
printVector(x);
fprintf("Ax = %n");
printVector(A.operate(x));
fprintf("b = %n");
printVector(b);

Matrix B = new DenseMatrix(new double[][] {
		{2, 4},
		{3, 3}, 
		{4, 2} }
		);
Matrix X = LUDecomp.solve(B);
fprintf("Solution for AX = B:%n");
printMatrix(X);
fprintf("AX = %n");
printMatrix(A.mtimes(X));
fprintf("B = %n");
printMatrix(B);

fprintf("A^{-1}:%n");
printMatrix(LUDecomp.inverse());

fprintf("det(A) = %.2f%n", LUDecomp.det());
System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);

# Output

A:
         1        -2         3  
         2        -5        12  
         0         2       -10  

L:
         1         0         0  
         0         1         0  
    0.5000    0.2500         1  

U:
         2        -5        12  
         0         2       -10  
         0         0   -0.5000  

P:
         0         1         0  
         0         0         1  
         1         0         0  

PA:
         2        -5        12  
         0         2       -10  
         1        -2         3  

LU:
         2        -5        12  
         0         2       -10  
         1        -2         3  

Solution for Ax = b:
        13
         7
         1

b = 
         2
         3
         4

Ax = 
         2
         3
         4

A^{-1}:
       -13         7    4.5000  
       -10         5         3  
        -2         1    0.5000  

det(A) = -2.00
Elapsed time: 0.02 seconds.
**********************************
A:
         1        -2         3  
         2        -5        12  
                   2       -10  

L:
         1  
                   1  
    0.5000    0.2500         1  

U:
         2        -5        12  
                   2       -10  
                       -0.5000  

P:
                   1  
                             1  
         1  

PA:
         2        -5        12  
                   2       -10  
         1        -2         3  

LU:
         2        -5        12  
                   2       -10  
         1        -2         3  

Solution for Ax = b:
        13
         7
         1

Ax = 
         2
         3
         4

b = 
         2
         3
         4

Solution for AX = B:
        13       -22  
         7       -19  
         1        -4  

AX = 
         2         4  
         3         3  
         4         2  

B = 
         2         4  
         3         3  
         4         2  

A^{-1}:
       -13         7    4.5000  
       -10         5         3  
        -2         1    0.5000  

det(A) = -2.00
Elapsed time: 0.02 seconds.

# -------------------------------------------------------------------------- #

# QR Decomposition

int m = 4;
int n = 3;
Matrix A = hilb(m, n);

fprintf("When A is full:%n");

fprintf("A:%n");
printMatrix(A);

long start = 0;
start = System.currentTimeMillis();

Matrix[] QRP = QRDecomposition.decompose(A);
Matrix Q = QRP[0];
Matrix R = QRP[1];
Matrix P = QRP[2];

fprintf("Q:%n");
printMatrix(Q);

fprintf("R:%n");
printMatrix(R);

fprintf("P:%n");
printMatrix(P);

fprintf("AP:%n");
printMatrix(A.mtimes(P));

fprintf("QR:%n");
printMatrix(Q.mtimes(R));

fprintf("Q'Q:%n");
printMatrix(Q.transpose().mtimes(Q));

System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);
fprintf("**********************************%n");

// fprintf("|AP - QR| = ");

A = sparse(hilb(m, n));

fprintf("When A is sparse:%n");

fprintf("A:%n");
printMatrix(A);

start = System.currentTimeMillis();

QRP = QRDecomposition.decompose(A);
Q = QRP[0];
R = QRP[1];
P = QRP[2];

fprintf("Q:%n");
printMatrix(Q);

fprintf("R:%n");
printMatrix(R);

fprintf("P:%n");
printMatrix(P);

fprintf("AP:%n");
printMatrix(A.mtimes(P));

fprintf("QR:%n");
printMatrix(Q.mtimes(R));

fprintf("Q'Q:%n");
printMatrix(Q.transpose().mtimes(Q));

System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);

QRDecomposition QRDecomp = new QRDecomposition((A));
Vector b = new DenseVector(new double[] {2, 3, 4, 9});
Vector x = QRDecomp.solve(b);
fprintf("Solution for Ax = b:%n");
printVector(x);
fprintf("b = %n");
printVector(b);
fprintf("Ax = %n");
printVector(A.operate(x));

# Output

When A is full:
A:
         1    0.5000    0.3333  
    0.5000    0.3333    0.2500  
    0.3333    0.2500    0.2000  
    0.2500    0.2000    0.1667  

Q:
   -0.8381    0.5144   -0.1796   -0.0263  
   -0.4191   -0.4052    0.7487    0.3157  
   -0.2794   -0.5351   -0.1132   -0.7892  
   -0.2095   -0.5338   -0.6280    0.5261  

R:
   -1.1932   -0.4749   -0.6705  
         0   -0.1258   -0.1184  
         0         0    0.0059  
         0         0         0  

P:
         1  
                             1  
                   1  

AP:
         1    0.3333    0.5000  
    0.5000    0.2500    0.3333  
    0.3333    0.2000    0.2500  
    0.2500    0.1667    0.2000  

QR:
    1.0000    0.3333    0.5000  
    0.5000    0.2500    0.3333  
    0.3333    0.2000    0.2500  
    0.2500    0.1667    0.2000  

Q'Q:
    1.0000    0.0000   -0.0000    0.0000  
    0.0000         1   -0.0000         0  
   -0.0000   -0.0000    1.0000         0  
    0.0000         0         0         1  

Elapsed time: 0.05 seconds.
**********************************
When A is sparse:
A:
         1    0.5000    0.3333  
    0.5000    0.3333    0.2500  
    0.3333    0.2500    0.2000  
    0.2500    0.2000    0.1667  

Q:
   -0.8381    0.5144   -0.1796   -0.0263  
   -0.4191   -0.4052    0.7487    0.3157  
   -0.2794   -0.5351   -0.1132   -0.7892  
   -0.2095   -0.5338   -0.6280    0.5261  

R:
   -1.1932   -0.4749   -0.6705  
             -0.1258   -0.1184  
                        0.0059  
  

P:
         1  
                             1  
                   1  

AP:
         1    0.3333    0.5000  
    0.5000    0.2500    0.3333  
    0.3333    0.2000    0.2500  
    0.2500    0.1667    0.2000  

QR:
    1.0000    0.3333    0.5000  
    0.5000    0.2500    0.3333  
    0.3333    0.2000    0.2500  
    0.2500    0.1667    0.2000  

Q'Q:
    1.0000    0.0000   -0.0000    0.0000  
    0.0000         1   -0.0000         0  
   -0.0000   -0.0000    1.0000         0  
    0.0000         0         0         1  

Elapsed time: 0.04 seconds.
Solution for Ax = b:
  117.2346
  -719.5017
  733.7439

b = 
         2
         3
         4
         9

Ax = 
    2.0651
    2.2194
    5.9516
    7.6990

# -------------------------------------------------------------------------- #

# Singular Value Decomposition

Matrix A = new DenseMatrix(new double[][] { {1d, 2d}, {2d, 0d}, {1d, 7d}});

/*A = new DenseMatrix(new double[][] {
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12}
});*/
// printMatrix(SingularValueDecomposition.bidiagonalize(A)[1]);

// A = IO.loadMatrix("SVDInput");

/*fprintf("When A is full:%n%n");

fprintf("A:%n");
printMatrix(A);*/

long start = 0;
start = System.currentTimeMillis();

boolean computeUV = !false;
Matrix[] USV = SingularValueDecomposition.decompose(A, computeUV);

System.out.format("Elapsed time: %.4f seconds.%n", (System.currentTimeMillis() - start) / 1000.0);
fprintf("*****************************************%n");

Matrix U = USV[0];
Matrix S = USV[1];
Matrix V = USV[2];

if (computeUV) {
	fprintf("USV':%n");
	disp(U.mtimes(S).mtimes(V.transpose()));

	fprintf("A:%n");
	printMatrix(A);

	fprintf("U'U:%n");
	printMatrix(U.transpose().mtimes((U)));

	fprintf("V'V:%n");
	printMatrix(V.transpose().mtimes((V)));
	
	fprintf("U:%n");
	printMatrix(U);
	
	fprintf("V:%n");
	printMatrix(V);

}

fprintf("S:%n");
printMatrix(S);

fprintf("rank(A): %d%n", rank(A));

# Output

Elapsed time: 0.0100 seconds.
*****************************************
USV':
    1.0000    2.0000  
    2.0000   -0.0000  
    1.0000    7.0000  

A:
         1         2  
         2         0  
         1         7  

U'U:
    1.0000    0.0000    0.0000  
    0.0000    1.0000    0.0000  
    0.0000    0.0000    1.0000  

V'V:
    1.0000   -0.0000  
   -0.0000    1.0000  

U:
   -0.2906   -0.2976   -0.9094  
   -0.0492   -0.9445    0.3248  
   -0.9556    0.1391    0.2598  

V:
   -0.1819   -0.9833  
   -0.9833    0.1819  

S:
    7.3935  
              2.0822  
  

rank(A): 2

# -------------------------------------------------------------------------- #

# Linear Binary SVM

double C = 1.0;
double eps = 1e-4;
Classifier linearBinarySVM = new LinearBinarySVM(C, eps);

int[] pred_labels = null;
double[][] data = { 
		{3.5, 4.4, 1.3, 2.3},
		{5.3, 2.2, 0.5, 4.5},
		{0.2, 0.3, 4.1, -3.1},
		{-1.2, 0.4, 3.2, 1.6}
		};

int[] labels = new int[] {1, 1, -1, -1};

linearBinarySVM.feedData(data);
linearBinarySVM.feedLabels(labels);
linearBinarySVM.train();
fprintf("W:%n");
printMatrix(linearBinarySVM.W);
fprintf("b:%n");
printVector(linearBinarySVM.b);
pred_labels = linearBinarySVM.predict(data);
getAccuracy(pred_labels, labels);

# Output

W:
    0.2143  
    0.1312  
   -0.2407  
    0.0150  

b:
   -0.0490

Accuracy: 100.00%

# -------------------------------------------------------------------------- #

# Linear Multi-Class SVM

double C = 1.0;
double eps = 1e-4;
Classifier linearMCSVM = new LinearMCSVM(C, eps);

double[][] data = { 
		{3.5, 4.4, 1.3, 2.3},
		{5.3, 2.2, 0.5, 4.5},
		{0.2, 0.3, 4.1, -3.1},
		{-1.2, 0.4, 3.2, 1.6}
		};

int[] labels = new int[] {1, 2, 3, 4};

linearMCSVM.feedData(data);
linearMCSVM.feedLabels(labels);
linearMCSVM.train();
fprintf("W:%n");
printMatrix(linearMCSVM.W);
fprintf("b:%n");
printVector(linearMCSVM.b);
int[] pred_labels = linearMCSVM.predict(data);
getAccuracy(pred_labels, labels);

# Output

..
W:
   -0.0482    0.1314    0.0630   -0.1462  
    0.2825   -0.2241   -0.0053   -0.0531  
   -0.0819   -0.0836    0.0599    0.1056  
   -0.0838    0.1711   -0.2301    0.1428  

b:
   -0.0248   -0.0089    0.0043    0.0295

Accuracy: 100.00%

# -------------------------------------------------------------------------- #

# Multi-Class Logistic Regression with Multiple Choices of Regularization

double[][] data = {
		{3.5, 5.3, 0.2, -1.2},
		{4.4, 2.2, 0.3, 0.4},
		{1.3, 0.5, 4.1, 3.2}
		};

int[] labels = new int[] {1, 2, 3};

/*
 * Regularization type.
 * 0:  No regularization
 * 1:  L1 regularization
 * 2:  L2^2 regularization
 * 3:  L2 regularization
 * 4:  Infinity norm regularization
 */
int regularizationType = 1;
double lambda = 0.1;
Classifier logReg = new LogisticRegression(regularizationType, lambda);
logReg.epsilon = 1e-5;
logReg.feedData(data);
logReg.feedLabels(labels);

// Get elapsed time in seconds
tic();
logReg.train();
fprintf("Elapsed time: %.3f seconds.%n", toc());

fprintf("W:%n");
printMatrix(logReg.W);
fprintf("b:%n");
printVector(logReg.b);

double[][] dataTest = data;

fprintf("Ground truth:%n");
printMatrix(logReg.Y);
fprintf("Predicted probability matrix:%n");
Matrix Prob_pred = logReg.predictLabelScoreMatrix(dataTest);
disp(Prob_pred);
fprintf("Predicted label matrix:%n");
Matrix Y_pred = logReg.predictLabelMatrix(dataTest);
printMatrix(Y_pred);
int[] pred_labels = logReg.predict(dataTest);
getAccuracy(pred_labels, labels);

# Output

# Without regularization

L-BFGS converges with norm(Grad) 0.000006
Elapsed time: 0.060 seconds.
W:
   -1.8522    3.1339   -1.2817  
    3.4138   -1.7282   -1.6856  
   -1.2455   -1.2630    2.5084  
   -2.8311    0.5558    2.2753  

b:
   -0.3616    0.2534    0.1082

Ground truth:
         1  
                   1  
                             1  

Predicted probability matrix:
    1.0000    0.0000    0.0000  
    0.0000    1.0000    0.0000  
    0.0000    0.0000    1.0000  

Predicted label matrix:
         1  
                   1  
                             1  

Accuracy: 100.00%

# L1-norm regularization

Accelerated proximal gradient method converges with norm(G_Y_k) 0.000004
Elapsed time: 0.430 seconds.
W:
         0    0.8678         0  
    0.9579         0         0  
         0         0    0.9894  
         0         0         0  

b:
         0         0         0

Ground truth:
         1  
                   1  
                             1  

Predicted probability matrix:
    0.8790    0.1143    0.0067  
    0.1493    0.8263    0.0244  
    0.0258    0.0494    0.9247  

Predicted label matrix:
         1  
                   1  
                             1  

Accuracy: 100.00%

# squared Frobenius norm regularization

Accelerated proximal gradient method converges with norm(G_Y_k) 0.000009
Elapsed time: 0.242 seconds.
W:
   -0.2817    0.5368   -0.2551  
    0.5298   -0.3269   -0.2029  
   -0.1181   -0.3004    0.4185  
   -0.3814    0.0449    0.3364  

b:
   -0.0417    0.0266    0.0151

Ground truth:
         1  
                   1  
                             1  

Predicted probability matrix:
    0.8872    0.1028    0.0100  
    0.1241    0.8299    0.0461  
    0.0137    0.0514    0.9349  

Predicted label matrix:
         1  
                   1  
                             1  

Accuracy: 100.00%

# Frobenius norm regularization

Accelerated proximal gradient method converges with norm(G_Y_k) 0.000008
Elapsed time: 0.358 seconds.
W:
   -0.4333    0.7669   -0.3336  
    0.7402   -0.4956   -0.2447  
   -0.1262   -0.4227    0.5489  
   -0.5135    0.0810    0.4326  

b:
   -0.0587    0.0375    0.0212

Ground truth:
         1  
                   1  
                             1  

Predicted probability matrix:
    0.9509    0.0461    0.0029  
    0.0558    0.9249    0.0192  
    0.0039    0.0221    0.9740  

Predicted label matrix:
         1  
                   1  
                             1  

Accuracy: 100.00%

# Infinity norm regularization

Accelerated proximal gradient method converges with norm(G_Y_k) 0.000010
Elapsed time: 0.631 seconds.
W:
   -0.7162    0.7162   -0.7162  
    0.7162   -0.7162   -0.7162  
   -0.7162   -0.7162    0.7162  
   -0.7162    0.7162    0.7162  

b:
   -0.6125   -0.3405    0.5728

Ground truth:
         1  
                   1  
                             1  

Predicted probability matrix:
    0.9821    0.0175    0.0004  
    0.0179    0.9752    0.0068  
    0.0000    0.0072    0.9928  

Predicted label matrix:
         1  
                   1  
                             1  

Accuracy: 100.00%

# -------------------------------------------------------------------------- #

# Maximum Entropy Modeling

long start = System.currentTimeMillis();

/*
 * a 3D {@code double} array, where data[n][i][k]
 * is the i-th feature value on the k-th class 
 * for the n-th sample
 */
double[][][] data = new double[][][] {
		{{1, 0, 0}, {2, 1, -1}, {0, 1, 2}, {-1, 2, 1}},
		{{0, 2, 0}, {1, 0, -1}, {0, 1, 1}, {-1, 3, 0.5}},
		{{0, 0, 0.8}, {2, 1, -1}, {1, 3, 0}, {-0.5, -1, 2}},
		{{0.5, 0, 0}, {1, 1, -1}, {0, 0.5, 1.5}, {-2, 1.5, 1}},
};

/*double [][] labels = new double[][] { 
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{1, 0, 0}
};*/
int[] labels = new int[] {1, 2, 3, 1};

MaxEnt maxEnt = new MaxEnt();
maxEnt.feedData(data);
maxEnt.feedLabels(labels);
maxEnt.train();
double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);

fprintf("MaxEnt parameters:\n");
display(maxEnt.W);
String modelFilePath = "MaxEnt-Model.dat";
maxEnt.saveModel(modelFilePath);

maxEnt = new MaxEnt();
maxEnt.loadModel(modelFilePath);
fprintf("Predicted probability matrix:\n");
display(maxEnt.predictLabelScoreMatrix(data));
fprintf("Predicted label matrix:\n");
display(full(maxEnt.predictLabelMatrix(data)));
fprintf("Predicted labels:\n");
display(maxEnt.predict(data));

# Output

L-BFGS converges with norm(Grad) 0.000072
Elapsed time: 0.060 seconds
MaxEnt parameters:
    7.4697  
   -1.1159  
   -2.8338  
   -1.0426  

Model saved.
Loading model...
Model loaded.
Predicted probability matrix:
    1.0000    0.0000    0.0000  
    0.0001    0.9999    0.0000  
    0.0001    0.0000    0.9999  
    0.9997    0.0002    0.0001  

Predicted label matrix:
         1         0         0  
         0         1         0  
         0         0         1  
         1         0         0  

Predicted labels:
        1          2          3          1  

# -------------------------------------------------------------------------- #
		
# AdaBoost

double[][] data = { {3.5, 4.4, 1.3},
					{5.3, 2.2, 0.5},
					{0.2, 0.3, 4.1},
					{5.3, 2.2, -1.5},
					{-1.2, 0.4, 3.2} };
int[] labels = {1, 1, -1, -1, -1};

Matrix X = new DenseMatrix(data);

int T = 10;
Classifier[] weakClassifiers = new Classifier[T];
for (int t = 0; t < 10; t++) {
	weakClassifiers[t] = new LogisticRegression(epsilon); 
}
Classifier adaBoost = new AdaBoost(weakClassifiers);

adaBoost.feedData(X);
adaBoost.feedLabels(labels);
tic();
adaBoost.train();
System.out.format("Elapsed time: %.2f seconds.%n", toc());

Xt = X.copy();
display(full(adaBoost.predictLabelMatrix(Xt)));
display(adaBoost.predict(Xt));
accuracy = Classifier.getAccuracy(labels, adaBoost.predict(Xt));
fprintf("Accuracy for AdaBoost with logistic regression: %.2f%%\n", 100 * accuracy);

# Output

Elapsed time: 0.18 seconds.
         1         0  
         1         0  
         0         1  
         0         1  
         0         1  

        1          1         -1         -1         -1  
Accuracy: 100.00%

# -------------------------------------------------------------------------- #

# K-means

double[][] data = {
		{3.5, 5.3, 0.2, -1.2},
		{4.4, 2.2, 0.3, 0.4},
		{1.3, 0.5, 4.1, 3.2}
		};

KMeansOptions options = new KMeansOptions();
options.nClus = 2;
options.verbose = true;
options.maxIter = 100;

KMeans KMeans= new KMeans(options);

KMeans.feedData(data);
// KMeans.initialize(null);
Matrix initializer = null;
initializer = new SparseMatrix(3, 2);
initializer.setEntry(0, 0, 1);
initializer.setEntry(1, 1, 1);
initializer.setEntry(2, 0, 1);
KMeans.clustering(initializer); // Use null for random initialization

System.out.println("Indicator Matrix:");
printMatrix(full(KMeans.getIndicatorMatrix()));

# Output

Iter 1: mse = 9.534 (0.000 secs)
KMeans complete.
Indicator Matrix:
         0         1  
         0         1  
         1         0

# -------------------------------------------------------------------------- #
		 
# NMF

double[][] data = { 
		{3.5, 4.4, 1.3},
		{5.3, 2.2, 0.5},
		{0.2, 0.3, 4.1},
		{1.2, 0.4, 3.2} 
		};

KMeansOptions options = new KMeansOptions();
options.nClus = 2;
options.verbose = true;
options.maxIter = 100;

KMeans KMeans= new KMeans(options);

KMeans.feedData(data);
KMeans.initialize(null);
KMeans.clustering();
Matrix G0 = KMeans.getIndicatorMatrix();

NMFOptions NMFOptions = new NMFOptions();
NMFOptions.nClus = 2;
NMFOptions.maxIter = 50;
NMFOptions.verbose = true;
NMFOptions.calc_OV = false;
NMFOptions.epsilon = 1e-5;
Clustering NMF = new NMF(NMFOptions);

NMF.feedData(data);
// NMF.initialize(null);
NMF.clustering(G0); // If null, KMeans will be used for initialization

System.out.println("Basis Matrix:");
printMatrix(full(NMF.getCenters()));

System.out.println("Indicator Matrix:");
printMatrix(full(NMF.getIndicatorMatrix()));

# Output

Iter 1: mse = 13.060 (0.000 secs)
Iter 2: mse = 2.875 (0.000 secs)
KMeans complete.
Iteration 10, delta G: 0.001012
Converge successfully!
Basis Matrix:
    5.0577    3.6774    0.5237  
    0.2326    0.3013    4.4328  

Indicator Matrix:
    0.8543    0.2004  
    0.8928         0  
    0.0000    0.9251  
    0.1520    0.7014

# -------------------------------------------------------------------------- #
	
# L1NMF

String dataMatrixFilePath = "CNN - DocTermCount.txt";

tic();
Matrix X = loadMatrixFromDocTermCountFile(dataMatrixFilePath);
X = Matlab.getTFIDF(X);
X = Matlab.normalizeByColumns(X);
X = X.transpose();

KMeansOptions kMeansOptions = new KMeansOptions();
kMeansOptions.nClus = 10;
kMeansOptions.maxIter = 50;
kMeansOptions.verbose = true;

KMeans KMeans = new KMeans(kMeansOptions);
KMeans.feedData(X);
// KMeans.initialize(null);
KMeans.clustering();

Matrix G0 = KMeans.getIndicatorMatrix();

// Matrix X = Data.loadSparseMatrix("X.txt");
G0 = loadDenseMatrix("G0.txt");
L1NMFOptions L1NMFOptions = new L1NMFOptions();
L1NMFOptions.nClus = 10;
L1NMFOptions.gamma = 1 * 0.0001;
L1NMFOptions.mu = 1 * 0.1;
L1NMFOptions.maxIter = 50;
L1NMFOptions.verbose = true;
L1NMFOptions.calc_OV = !true;
L1NMFOptions.epsilon = 1e-5;
Clustering L1NMF = new L1NMF(L1NMFOptions);
L1NMF.feedData(X);
// L1NMF.initialize(G0);

L1NMF.clustering(G0); // Use null for random initialization

System.out.format("Elapsed time: %.3f seconds\n", toc());

# Output

Iter 1: mse = 1.524 (0.030 secs)
Iter 2: mse = 0.816 (0.030 secs)
Iter 3: mse = 0.806 (0.030 secs)
Iter 4: mse = 0.805 (0.040 secs)
KMeans complete.
Iteration 10, delta G: 0.046591
Iteration 20, delta G: 0.047140
Iteration 30, delta G: 0.020651
Iteration 40, delta G: 0.010017
Iteration 50, delta G: 0.007973
Maximal iterations
Elapsed time: 3.933 seconds

# -------------------------------------------------------------------------- #

# Spectral Clustering

tic();

int nClus = 2;
boolean verbose = false;
int maxIter = 100;
String graphType = "nn";
double graphParam = 2;
String graphDistanceFunction = "euclidean";
String graphWeightType = "heat";
double graphWeightParam = 1;
ClusteringOptions options = new SpectralClusteringOptions(
		nClus,
		verbose,
		maxIter,
		graphType,
		graphParam,
		graphDistanceFunction,
		graphWeightType,
		graphWeightParam);

Clustering spectralClustering = new SpectralClustering(options);

double[][] data = {
		{3.5, 5.3, 0.2, -1.2},
		{4.4, 2.2, 0.3, 0.4},
		{1.3, 0.5, 4.1, 3.2}
		};

spectralClustering.feedData(data);
spectralClustering.clustering(null);
display(full(spectralClustering.getIndicatorMatrix()));

System.out.format("Elapsed time: %.3f seconds\n", toc());

# Output

Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 2.
KMeans complete.
Spectral clustering complete.
         0         1  
         0         1  
         1         0  

Elapsed time: 0.070 seconds

# -------------------------------------------------------------------------- #

# Computing Adjacency Matrix, Graph Laplacian and Local Learning Regularization

String filePath = "CNN - DocTermCount.txt";
Matrix X = loadMatrixFromDocTermCountFile(filePath);
int NSample = Math.min(20, X.getColumnDimension());
X = X.getSubMatrix(0, X.getRowDimension() - 1, 0, NSample - 1);
System.out.println(String.format("%d samples loaded", X.getColumnDimension()));
X = X.transpose();
GraphOptions options = new GraphOptions();
options.graphType = "nn";
String type = options.graphType;
double NN = options.graphParam;
System.out.println(String.format("Graph type: %s with NN: %d", type, (int)NN));

// Parameter setting for text data
options.kernelType = "cosine";
options.graphDistanceFunction = "cosine";

// Parameter setting for image data
/*options.kernelType = "rbf";
options.graphDistanceFunction = "euclidean";*/

options.graphNormalize = true;
options.graphWeightType = "heat";

boolean show = true && !false;

// Test adjacency function - pass
tic();
String DISTANCEFUNCTION = options.graphDistanceFunction;
Matrix A = adjacency(X, type, NN, DISTANCEFUNCTION);
System.out.format("Elapsed time: %.2f seconds.%n", toc());
String adjacencyFilePath = "adjacency.txt";
saveMatrix(adjacencyFilePath, A);
if (show)
	disp(A.getSubMatrix(0, 4, 0, 4));

// Test laplacian function - pass
tic();
Matrix L = laplacian(X, type, options);
System.out.format("Elapsed time: %.2f seconds.%n", toc());
String LaplacianFilePath = "Laplacian.txt";
saveMatrix(LaplacianFilePath, L);
if (show)
	disp(L.getSubMatrix(0, 4, 0, 4));

// Test local learning regularization - pass
NN = options.graphParam;
String DISTFUNC = options.graphDistanceFunction;
String KernelType = options.kernelType;
double KernelParam = options.kernelParam;
double lambda = 0.001;
tic();
Matrix LLR_text = calcLLR(X, NN, DISTFUNC, KernelType, KernelParam, lambda);
System.out.format("Elapsed time: %.2f seconds.%n", toc());
String LLRFilePath = "localLearningRegularization.txt";
saveMatrix(LLRFilePath, LLR_text);
if (show)
	display(LLR_text.getSubMatrix(0, 4, 0, 4));
	
# Output

20 samples loaded
Graph type: nn with NN: 6
Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 6.
Elapsed time: 0.28 seconds.
Data matrix file written: adjacency.txt

    (2, 1)   0.8162
    (3, 1)   0.8841
    (1, 2)   0.8162
    (3, 2)   0.9041
    (4, 2)   0.9074
    (1, 3)   0.8841
    (2, 3)   0.9041
    (2, 4)   0.9074
    (5, 4)   0.9558
    (4, 5)   0.9558

Computing Graph Laplacian...
Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 6.
Elapsed time: 0.22 seconds.
Data matrix file written: Laplacian.txt

    (1, 1)        1
    (2, 1)  -0.1522
    (3, 1)  -0.1118
    (1, 2)  -0.1522
    (2, 2)        1
    (3, 2)  -0.1276
    (4, 2)  -0.1521
    (1, 3)  -0.1118
    (2, 3)  -0.1276
    (3, 3)        1
    (2, 4)  -0.1521
    (4, 4)        1
    (5, 4)  -0.1499
    (4, 5)  -0.1499
    (5, 5)        1

Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 6.
Elapsed time: 0.26 seconds.
Data matrix file written: localLearningRegularization.txt

    (1, 1)   1.0562
    (2, 1)  -0.2926
    (3, 1)  -0.1244
    (4, 1)   0.0132
    (1, 2)  -0.2926
    (2, 2)   1.0353
    (3, 2)  -0.1167
    (4, 2)  -0.1202
    (5, 2)   0.0026
    (1, 3)  -0.1244
    (2, 3)  -0.1167
    (3, 3)   1.0305
    (4, 3)   0.0045
    (1, 4)   0.0132
    (2, 4)  -0.1202
    (3, 4)   0.0045
    (4, 4)   1.0583
    (5, 4)  -0.0052
    (2, 5)   0.0026
    (4, 5)  -0.0052
    (5, 5)   1.0042

# -------------------------------------------------------------------------- #

# Matrix Completion

int m = 6;
int r = 1;
int p = (int) Math.round(m * m * 0.3);

Matrix L = randn(m, r);
Matrix R = randn(m, r);
Matrix A_star = mtimes(L, R.transpose());

int[] indices = randperm(m * m);
minusAssign(indices, 1);
indices = linearIndexing(indices, colon(0, p - 1));

Matrix Omega = zeros(size(A_star));
linearIndexingAssignment(Omega, indices, 1);

Matrix D = zeros(size(A_star));
linearIndexingAssignment(D, indices, linearIndexing(A_star, indices));
		
Matrix E_star = D.minus(A_star);
logicalIndexingAssignment(E_star, Omega, 0);

// Run matrix completion
MatrixCompletion matrixCompletion = new MatrixCompletion();
matrixCompletion.feedData(D);
matrixCompletion.feedIndices(Omega);
tic();
matrixCompletion.run();
fprintf("Elapsed time: %.2f seconds.%n", toc());

// Output
Matrix A_hat = matrixCompletion.GetLowRankEstimation();

fprintf("A*:\n");
disp(A_star, 4);
fprintf("A^:\n");
disp(A_hat, 4);
fprintf("D:\n");
disp(D, 4);
fprintf("rank(A*): %d\n", rank(A_star));
fprintf("rank(A^): %d\n", rank(A_hat));
fprintf("||A* - A^||_F: %.4f\n", norm(A_star.minus(A_hat), "fro"));

# Output

Elapsed time: 0.07 seconds.
A*:
    1.1683   -0.4309    1.7763   -0.3742   -0.4096   -0.6490  
   -2.9765    1.0978   -4.5257    0.9533    1.0437    1.6536  
   -0.4342    0.1601   -0.6602    0.1391    0.1522    0.2412  
    1.6696   -0.6158    2.5386   -0.5347   -0.5854   -0.9276  
   -0.6291    0.2320   -0.9566    0.2015    0.2206    0.3495  
    0.4162   -0.1535    0.6329   -0.1333   -0.1459   -0.2312  

A^:
    1.1683   -0.4309   -0.0000   -0.3742   -0.4096   -0.6490  
   -2.9765    1.0978    0.0000    0.9533    1.0437    1.6536  
   -0.4342    0.1601    0.0000    0.1391    0.1522    0.2412  
    1.6696   -0.6158   -0.0000   -0.5347   -0.5854   -0.9276  
   -0.6291    0.2320    0.0000    0.2015    0.2206    0.3495  
    0.4162   -0.1535   -0.0000   -0.1333   -0.1459   -0.2312  

D:
         0         0         0         0         0   -0.6490  
   -2.9765         0         0         0    1.0437    1.6536  
         0         0         0    0.1391    0.1522         0  
    1.6696   -0.6158         0   -0.5347         0         0  
         0         0         0         0    0.2206         0  
         0         0         0   -0.1333         0         0  

rank(A*): 1
rank(A^): 1
||A* - A^||_F: 5.6420

# -------------------------------------------------------------------------- #

# Robust PCA

int m = 8;
int r = m / 4;

Matrix L = randn(m, r);
Matrix R = randn(m, r);

Matrix A_star = mtimes(L, R.transpose());
Matrix E_star = zeros(size(A_star));
int[] indices = randperm(m * m);
int nz = m * m / 20;
int[] nz_indices = new int[nz];
for (int i = 0; i < nz; i++) {
	nz_indices[i] = indices[i] - 1;
}
Matrix E_vec = vec(E_star);
setSubMatrix(E_vec, nz_indices, new int[] {0}, (minus(rand(nz, 1), 0.5).times(100)));
E_star = reshape(E_vec, size(E_star));

// Input
Matrix D = A_star.plus(E_star);
double lambda = 1 * Math.pow(m, -0.5);

// Run Robust PCA
RobustPCA robustPCA = new RobustPCA(lambda);
robustPCA.feedData(D);
tic();
robustPCA.run();
fprintf("Elapsed time: %.2f seconds.%n", toc());

// Output
Matrix A_hat = robustPCA.GetLowRankEstimation();
Matrix E_hat = robustPCA.GetErrorMatrix();

fprintf("A*:\n");
disp(A_star, 4);
fprintf("A^:\n");
disp(A_hat, 4);
fprintf("E*:\n");
disp(E_star, 4);
fprintf("E^:\n");
disp(E_hat, 4);
fprintf("rank(A*): %d\n", rank(A_star));
fprintf("rank(A^): %d\n", rank(A_hat));
fprintf("||A* - A^||_F: %.4f\n", norm(A_star.minus(A_hat), "fro"));
fprintf("||E* - E^||_F: %.4f\n", norm(E_star.minus(E_hat), "fro"));

# Output

Elapsed time: 0.05 seconds.
A*:
    0.4542    1.4239   -0.7884    0.9424   -1.9555    0.3413   -0.9643    1.9974  
    3.2620   -0.5927   -1.7637    0.6375    1.4598   -1.2037    1.9060   -1.8263  
   -1.9222   -0.4894    1.3415   -0.8509    0.3416    0.4260   -0.4386   -0.1774  
   -1.3192    0.7056    0.5454    0.0062   -1.2580    0.6442   -1.1511    1.4349  
   -2.2590    0.3961    1.2266   -0.4496   -0.9903    0.8288   -1.3082    1.2432  
    3.2288    1.0411   -2.3324    1.5534   -0.8878   -0.6416    0.5579    0.6255  
    1.7459   -0.1703   -0.9969    0.4244    0.5708   -0.5946    0.9002   -0.7579  
    1.0295   -1.0952   -0.2294   -0.3135    1.7622   -0.6867    1.3429   -1.9339  

A^:
    0.4296    1.4239   -0.7884    0.9422   -1.9322    0.3282   -0.9459    1.9804  
    3.1412   -0.5927   -1.7637    0.6375    1.4596   -1.2037    1.9060   -1.8263  
   -1.8337   -0.4894    1.3369   -0.8509    0.3325    0.4261   -0.4385   -0.1774  
   -1.3192   -0.4311    0.5454   -0.1480    0.2049    0.6442   -0.7934    0.4356  
   -2.1752    0.3960    1.2266   -0.4496   -0.9903    0.8287   -1.3082    1.2432  
    2.6885    1.0320   -1.9370    1.2690   -0.8878   -0.6340    0.5579    0.5161  
    1.6787   -0.1704   -0.9966    0.4247    0.5725   -0.5947    0.9002   -0.7579  
    1.0103   -1.0939   -0.2342   -0.3135    1.7505   -0.6867    1.3428   -1.9330  

E*:
         0         0         0         0         0         0         0         0  
         0         0         0         0         0         0         0         0  
         0         0         0         0         0         0         0         0  
         0  -20.5412         0         0   34.4119         0         0         0  
         0         0         0         0         0         0         0         0  
         0         0         0         0         0         0         0         0  
         0         0         0         0   20.5941         0         0         0  
         0         0         0         0         0         0         0         0  

E^:
    0.0245         0         0    0.0002   -0.0233    0.0131   -0.0184    0.0170  
    0.1208         0         0         0    0.0001   -0.0001    0.0000         0  
   -0.0885         0    0.0046   -0.0000    0.0091   -0.0001   -0.0001         0  
         0  -19.4046         0    0.1542   32.9490         0   -0.3577    0.9993  
   -0.0839    0.0001    0.0000         0         0    0.0000    0.0000    0.0000  
    0.5403    0.0091   -0.3953    0.2844         0   -0.0076         0    0.1094  
    0.0672    0.0001   -0.0004   -0.0003   20.5923    0.0000         0         0  
    0.0193   -0.0014    0.0048         0    0.0117         0    0.0001   -0.0009  

rank(A*): 2
rank(A^): 4
||A* - A^||_F: 2.2716
||E* - E^||_F: 2.2716

# -------------------------------------------------------------------------- #

# LASSO

double[][] data = {{1, 2, 3, 2},
				   {4, 2, 3, 6},
				   {5, 1, 2, 1}};

double[][] depVars = {{3, 2},
					  {2, 3},
					  {1, 4}};

Options options = new Options();
options.maxIter = 600;
options.lambda = 0.05;
options.verbose = !true;
options.calc_OV = !true;
options.epsilon = 1e-5;

Regression LASSO = new LASSO(options);
LASSO.feedData(data);
LASSO.feedDependentVariables(depVars);

tic();
LASSO.train();
fprintf("Elapsed time: %.3f seconds\n\n", toc());

fprintf("Projection matrix:\n");
display(LASSO.W);

Matrix Yt = LASSO.predict(data);
fprintf("Predicted dependent variables:\n");
display(Yt);

# Output

Elapsed time: 0.060 seconds

Projection matrix:
   -0.2295    0.5994  
         0         0  
    1.1058    0.5858  
   -0.0631   -0.1893  

Predicted dependent variables:
    2.9618    1.9782  
    2.0209    3.0191  
    1.0009    3.9791

# -------------------------------------------------------------------------- #
	
# Linear Regression

double[][] data = {
		{1, 2, 3, 2},
		{4, 2, 3, 6},
		{5, 1, 4, 1}
		}; 

double[][] depVars = {
		{3, 2},
		{2, 3},
		{1, 4}
		};

Options options = new Options();
options.maxIter = 600;
options.lambda = 0.1;
options.verbose = !true;
options.calc_OV = !true;
options.epsilon = 1e-5;

Regression LR = new LinearRegression(options);
LR.feedData(data);
LR.feedDependentVariables(depVars);

tic();
LR.train();
fprintf("Elapsed time: %.3f seconds\n\n", toc());

fprintf("Projection matrix:\n");
display(LR.W);

fprintf("Bias vector:\n");
display(((LinearRegression)LR).B);

Matrix Yt = LR.predict(data);
fprintf("Predicted dependent variables:\n");
display(Yt);

# Output

Elapsed time: 0.025 seconds

Projection matrix:
   -0.4700    0.4049  
    0.5621    0.1216  
    0.6117    0.4454  
    0.1163   -0.0513  

Bias vector:
    0.2345
    0.1131

Predicted dependent variables:
    2.9561    1.9949  
    2.0111    3.0043  
    1.0093    3.9895

# -------------------------------------------------------------------------- #

# Basic Conditional Random Field (CRF)

// Number of data sequences
int D = 1000;
// Minimal length for the randomly generated data sequences
int n_min = 4;
// Maximal length for the randomly generated data sequences
int n_max = 6;
// Number of feature functions
int d = 10;
// Number of states
int N = 2;
// Sparseness for the feature matrices
double sparseness = 0.2;

// Randomly generate labeled sequential data for CRF 
Object[] dataSequences = CRF.generateDataSequences(D, n_min, n_max, d, N, sparseness);
Matrix[][][] Fs = (Matrix[][][]) dataSequences[0];
int[][] Ys = (int[][]) dataSequences[1];

// Train a CRF model for the randomly generated sequential data with labels
double epsilon = 1e-4;
CRF CRF = new CRF(epsilon);
CRF.feedData(Fs);
CRF.feedLabels(Ys);
CRF.train();

// Save the CRF model
String modelFilePath = "CRF-Model.dat";
CRF.saveModel(modelFilePath);
fprintf("CRF Parameters:\n");
display(CRF.W);

// Prediction
CRF = new CRF();
CRF.loadModel(modelFilePath);
int ID = new Random().nextInt(D);
int[] Yt = Ys[ID];
Matrix[][] Fst = Fs[ID];

fprintf("True label sequence:\n");
display(Yt);
fprintf("Predicted label sequence:\n");
display(CRF.predict(Fst));

# Output

Initial ofv: 46.6525
Iter 1, ofv: 45.9559, norm(Grad): 0.475827
Iter 2, ofv: 33.8123, norm(Grad): 0.472433
Iter 3, ofv: 25.6766, norm(Grad): 0.335332
Objective function value doesn't decrease, iteration stopped!
Iter 4, ofv: 25.6766, norm(Grad): 0.0901032
Model saved.
CRF Parameters:
   -5.7255
    6.0560
    5.8877
   -7.7452
   11.2735
   -4.5474
   -0.6763
    3.6997
   -4.3791
    0.6701

Loading model...
Model loaded.
True label sequence:
        1          0          1          0  
Predicted label sequence:
P*(YPred|x) = 0.624515
        1          1          1          0  

# -------------------------------------------------------------------------- #
		
# Hidden Markov Model (HMM)

int numStates = 3;
int numObservations = 2;
double epsilon = 1e-8;
int maxIter = 10;

double[] pi = new double[] {0.33, 0.33, 0.34};

double[][] A = new double[][] {
		{0.5, 0.3, 0.2},
		{0.3, 0.5, 0.2},
		{0.2, 0.4, 0.4}
};

double[][] B = new double[][] {
		{0.7, 0.3},
		{0.5, 0.5},
		{0.4, 0.6}
};

// Generate the data sequences for training
int D = 10000;
int T_min = 5;
int T_max = 10;
int[][][] data = HMM.generateDataSequences(D, T_min, T_max, pi, A, B);
int[][] Os = data[0];
int[][] Qs = data[1];

boolean trainHMM = !false;
if (trainHMM){
	HMM HMM = new HMM(numStates, numObservations, epsilon, maxIter);
	HMM.feedData(Os);
	HMM.feedLabels(Qs);
	HMM.train();

	fprintf("True Model Parameters: \n");
	fprintf("Initial State Distribution: \n");
	display(pi);
	fprintf("State Transition Probability Matrix: \n");
	display(A);
	fprintf("Observation Probability Matrix: \n");
	display(B);

	fprintf("Trained Model Parameters: \n");
	fprintf("Initial State Distribution: \n");
	display(HMM.pi);
	fprintf("State Transition Probability Matrix: \n");
	display(HMM.A);
	fprintf("Observation Probability Matrix: \n");
	display(HMM.B);

	String HMMModelFilePath = "HMMModel.dat";
	HMM.saveModel(HMMModelFilePath);
}

// Predict the single best state path

int ID = new Random().nextInt(D);
int[] O = Os[ID];
		
HMM HMMt = new HMM();
HMMt.loadModel("HMMModel.dat");
int[] Q = HMMt.predict(O);

fprintf("Observation sequence: \n");
HMMt.showObservationSequence(O);
fprintf("True state sequence: \n");
HMMt.showStateSequence(Qs[ID]);
fprintf("Predicted state sequence: \n");
HMMt.showStateSequence(Q);
double p = HMMt.evaluate(O);
System.out.format("P(O|Theta) = %f\n", p);

# Output

Iter: 1, log[P(O|Theta)]: -51857.728975
Iter: 2, log[P(O|Theta)]: -51857.679272
Iter: 3, log[P(O|Theta)]: -51857.633951
Iter: 4, log[P(O|Theta)]: -51857.592261
Iter: 5, log[P(O|Theta)]: -51857.553794
Iter: 6, log[P(O|Theta)]: -51857.518197
Iter: 7, log[P(O|Theta)]: -51857.485156
Iter: 8, log[P(O|Theta)]: -51857.454400
Iter: 9, log[P(O|Theta)]: -51857.425690
Iter: 10, log[P(O|Theta)]: -51857.398814
True Model Parameters: 
Initial State Distribution: 
    0.3300
    0.3300
    0.3400

State Transition Probability Matrix: 
    0.5000    0.3000    0.2000  
    0.3000    0.5000    0.2000  
    0.2000    0.4000    0.4000  

Observation Probability Matrix: 
    0.7000    0.3000  
    0.5000    0.5000  
    0.4000    0.6000  

Trained Model Parameters: 
Initial State Distribution: 
    0.3309
    0.3269
    0.3422

State Transition Probability Matrix: 
    0.5000    0.3067    0.1934  
    0.2959    0.5031    0.2010  
    0.2037    0.3940    0.4023  

Observation Probability Matrix: 
    0.7030    0.2970  
    0.4957    0.5043  
    0.3956    0.6044  

Model saved.
Loading model...
Model loaded.
Observation sequence: 
1 0 1 0 1 0 0 
True state sequence: 
1 1 1 1 1 0 2 
Predicted state sequence: 
1 1 1 1 1 0 0 
P(O|Theta) = 0.007996

# -------------------------------------------------------------------------- #

# Isomap

double[][] data = {{0, 2, 3, 4}, {2, 0, 4, 5}, {3, 4.1, 5, 6}, {2, 7, 1, 6}};
Matrix X = new DenseMatrix(data);
X = X.transpose();

int K = 3;
int r = 3;
Matrix R = Isomap.run(X, K, r);
disp("Original Data:");
disp(X);
disp("Reduced Data:");
disp(R);

# Output

Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 3.
Original Data:
         0         2         3         2  
         2         0    4.1000         7  
         3         4         5         1  
         4         5         6         6  

Reduced Data:
    2.2473    2.5240   -0.7279  
   -3.4592    2.1429    0.4972  
    2.9742   -1.3689    0.7943  
   -1.7622   -3.2981   -0.5637

# -------------------------------------------------------------------------- #
   
# Kernel PCA

double[][] data = {
		{0, 2, 3, 4}, 
		{2, 0, 4, 5}, 
		{3, 4.1, 5, 6}, 
		{2, 7, 1, 6}
		};
Matrix X = new DenseMatrix(data).transpose();

int r = 3;
Matrix R = KernelPCA.run(X, r);
disp("Original Data:");
disp(X);
disp("Reduced Data:");
disp(R);

# Output

Original Data:
         0         2         3         2  
         2         0    4.1000         7  
         3         4         5         1  
         4         5         6         6  

Reduced Data:
    0.5011    0.0000    0.7063  
   -0.4987   -0.7080   -0.0001  
    0.4990    0.0018   -0.7078  
   -0.5013    0.7062    0.0017

# -------------------------------------------------------------------------- #
   
# Locally Linear Embedding (LLE)

double[][] data = {
		{0, 2, 3, 4}, 
		{2, 0, 4, 5}, 
		{3, 4.1, 5, 6}, 
		{2, 7, 1, 6}
		};
Matrix X = new DenseMatrix(data).transpose();

int K = 3;
int r = 3;
Matrix R = LLE.run(X, K, r);
disp("Original Data:");
disp(X);
disp("Reduced Data:");
disp(R);

# Output

Computing directed adjacency graph...
Creating the adjacency matrix. Nearest neighbors, N = 3.
Original Data:
         0         2         3         2  
         2         0    4.1000         7  
         3         4         5         1  
         4         5         6         6  

Reduced Data:
   -1.0360   -0.8728    1.0793  
    1.0282   -1.1127   -0.8395  
   -0.9630    0.8727   -1.1450  
    0.9707    1.1128    0.9051

# -------------------------------------------------------------------------- #

# Multi-dimensional Scaling (MDS)

double[][] data = {
		{0, 2, 3, 4}, 
		{2, 0, 4, 5}, 
		{3, 4.1, 5, 6}, 
		{2, 7, 1, 6} 
		};
Matrix O = new DenseMatrix(data).transpose();

Matrix D = l2Distance(O, O);
Matrix X = MDS.run(D, 3);
disp("Reduced X:");
disp(X);

# Output

Reduced X:
    2.2473    2.5240   -0.7279  
   -3.4592    2.1429    0.4972  
    2.9742   -1.3689    0.7943  
   -1.7622   -3.2981   -0.5637

# -------------------------------------------------------------------------- #
   
# Principal Component Analysis (PCA)

double[][] data = {
		{0, 2, 3, 4}, 
		{2, 0, 4, 5}, 
		{3, 4.1, 5, 6}, 
		{2, 7, 1, 6}
		};
Matrix X = new DenseMatrix(data).transpose();

int r = 3;
Matrix R = PCA.run(X, r);
disp("Original Data:");
disp(X);
disp("Reduced Data:");
disp(R);

# Output

Original Data:
         0         2         3         2  
         2         0    4.1000         7  
         3         4         5         1  
         4         5         6         6  

Reduced Data:
   -2.2473    2.5240    0.7279  
    3.4592    2.1429   -0.4972  
   -2.9742   -1.3689   -0.7943  
    1.7622   -3.2981    0.5637

# -------------------------------------------------------------------------- #

# General Quadratic Programming

/*
 * min  2 \ x' * Q * x + c' * x
 * s.t. A * x = b
 *      B * x <= d
 */

/*
 * Number of unknown variables
 */
int n = 5;

/*
 * Number of inequality constraints
 */
int m = 6;

/*
 * Number of equality constraints
 */
int p = 3;

Matrix x = rand(n, n);
Matrix Q = x.mtimes(x.transpose()).plus(times(rand(1), eye(n)));
Matrix c = rand(n, 1);

double HasEquality = 1;
Matrix A = times(HasEquality, rand(p, n));
x = rand(n, 1);
Matrix b = A.mtimes(x);
Matrix B = rand(m, n);
double rou = -2;
Matrix d = plus(B.mtimes(x), times(rou, ones(m, 1)));
QPSolution solution = GeneralQP.solve(Q, c, A, b, B, d);

# Output

Phase I:

Terminate successfully.

x_opt:
  640.1439  6991.6488  -2975.2211  -6851.4134  -6405.0501  

s_opt:
    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000  

lambda for the inequalities s_i >= 0:
    1.0000    1.0000    1.0000    1.0000    1.0000    1.0000  

B * x - d:
  -2123.5480  -2842.1394  -4496.4653  -1640.3370  -3613.3843  -14582.9937  

lambda for the inequalities fi(x) <= s_i:
    0.0000    0.0000    0.0000    0.0000    0.0000    0.0000  

nu for the equalities A * x = b:
   -0.0000   -0.0000   -0.0000  

residual: 1.07222e-11

A * x - b:
    0.0000    0.0000   -0.0000  

norm(A * x - b, "fro"): 0.000000

fval_opt: 3.65174e-11

The problem is feasible.

Computation time: 0.590000 seconds

halt execution temporarily in 1 seconds...

Phase II:

Terminate successfully.

residual: 9.83544e-12

Optimal objective function value: 173.050

Optimizer:
    1.4528    7.8845   -3.0459   -7.3020   -6.5998  

B * x - d:
   -0.0000   -1.4506   -3.0662   -0.0000   -2.0318  -14.1904  

lambda:
   52.4785    0.0000    0.0000  137.3581    0.0000    0.0000  

nu:
  -19.7244  -38.6926  -118.0042  

norm(A * x - b, "fro"): 0.000000

Computation time: 0.061000 seconds

# -------------------------------------------------------------------------- #

# Quadratic Programming with Bound Constraints

/*
 * min  2 \ x' * Q * x + c' * x
 * s.t. l <= x <= u
 */

int n = 5;
Matrix x = rand(n);
Matrix Q = minus(x.mtimes(x.transpose()), times(rand(1).getEntry(0, 0), eye(n)));
Matrix c = plus(-2, times(2, rand(n, 1)));
double l = 0;
double u = 1;
double epsilon = 1e-6;

QPSolution S = QPWithBoundConstraints.solve(Q, c, l, u, epsilon);

disp("Q:");
disp(Q);
disp("c:");
disp(c);
fprintf("Optimum: %g\n", S.optimum);
fprintf("Optimizer:\n");
display(S.optimizer.transpose());

# Output

Initial ofv: -0.678917
Iter 1, ofv: -1.28937, norm(PGrad): 3.08886
PLBFGS converges with norm(PGrad) 0.000000
Q:
    1.6699    1.4719    0.9729    0.9943    1.6691  
    1.4719    0.5587    0.5777    0.6273    1.1480  
    0.9729    0.5777    1.4139    0.6709    1.0215  
    0.9943    0.6273    0.6709    0.4050    0.6021  
    1.6691    1.1480    1.0215    0.6021    1.0643  

c:
   -0.3002  
   -1.7387  
   -1.7054  
   -1.8646  
   -1.9046  

Optimum: -1.28937
Optimizer:
         0    0.0469         0    0.7148         0

**************************************************		 
For other general-purpose optimization algorithms such as L-BFGS and accelerated proximal gradient, please refer to the documentation.

# -------------------------------------------------------------------------- #

# LDA

int[][] documents = { {1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6},
                                 {2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2},
                                 {1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0},
                                 {5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0},
                                 {2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0},
                                 {5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2} };
        
LDAOptions LDAOptions = new LDAOptions();
LDAOptions.nTopic = 2;
LDAOptions.iterations = 5000;
LDAOptions.burnIn = 1500;
LDAOptions.thinInterval = 200;
LDAOptions.sampleLag = 10;
LDAOptions.alpha = 2;
LDAOptions.beta = 0.5;

LDA LDA = new LDA(LDAOptions);
LDA.readCorpus(documents);
LDA.train();

fprintf("Topic--term associations: \n");
display(LDA.topicMatrix);

fprintf("Document--topic associations: \n");
display(LDA.indicatorMatrix);

# Output

Topic--term associations: 
   0.1258   0.0176  
   0.1531   0.0846  
   0.0327   0.3830  
   0.0418   0.1835  
   0.0360   0.2514  
   0.2713   0.0505  
   0.3393   0.0294  

Document--topic associations: 
   0.2559   0.7441  
   0.1427   0.8573  
   0.8573   0.1427  
   0.6804   0.3196  
   0.5491   0.4509  
   0.4420   0.5580

# -------------------------------------------------------------------------- #

# Factorization Machines (FM)

String trainFilePath = "Train.txt";
String testFilePath = "Test.txt";
String outputFilePath = "FM-YijPredOnTest.txt";

// Load training data
int idxStart = 0;
FM.feedTrainingData(trainFilePath, idxStart);

// Initialization
FM.allocateResource(k);
FM.feedParams(maxIter, lambda);
FM.initialize();

// Train FM model parameters by training data
FM.train();

// Prediction: generate and save FM-YijPredOnTest.txt
DataSet testData = FM.loadData(testFilePath, 0);
double[] Yij_pred = FM.predict(testData.X);
ml.utils.IO.save(Yij_pred, outputFilePath);

# -------------------------------------------------------------------------- #

# STructured Sparse Regression (STSR)

double lambda = 0.01;
double nu = 0.00001;
int maxIter = 30;

String trainFilePath = "Train.txt";
String testFilePath = "Test.txt";
String outputFilePath = "STSR-YijPredOnTest.txt";

// Load training data
int idxStart = 0;
STSR.feedTrainingData(trainFilePath, idxStart);

// Build tree structured pair groups
/*
 * featureSize format:
 * User[\t]383
 * Item[\t]1175
 * Event[\t]1
 */
String featureSizeFilePath = "FeatureSize.txt";
/*
 * Each line is a group feature index pair (idx1, idx2) separated by 
 * a tab character, e.g.
 * (157, 158)
 * (157, 236)
 * (24, 157)[\t](157, 158)[\t](157, 236)
 */
String userFeatureGroupListFilePath = "UserTreeStructuredPairGroupList.txt";
String itemFeatureGroupListFilePath = "ItemTreeStructuredPairGroupList.txt";
STSR.buildTreeStructuredPairGroupList(
		featureSizeFilePath,
		userFeatureGroupListFilePath,
		itemFeatureGroupListFilePath
		);

// Initialization
STSR.allocateResource();

STSR.feedParams(maxIter, lambda, nu);
STSR.initialize();
// Train STSR model parameters by training data
STSR.train();

// Prediction: generate and save STSR-YijPredOnTest.txt
DataSet testData = STSR.loadData(testFilePath, idxStart);
double[] Yij_pred = STSR.predict(testData.X);
ml.utils.IO.save(Yij_pred, outputFilePath);

# -------------------------------------------------------------------------- #

-----------------------------------
Author: Mingjie Qian
Version: 1.6.5
Date: April 25th, 2016
