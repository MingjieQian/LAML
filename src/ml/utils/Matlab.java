package ml.utils;

import static ml.utils.ArrayOperator.allocate1DArray;
import static ml.utils.ArrayOperator.allocate2DArray;
import static ml.utils.ArrayOperator.allocateVector;
import static ml.utils.ArrayOperator.divideAssign;
import static ml.utils.ArrayOperator.quickSort;
import static ml.utils.InPlaceOperator.assign;
import static ml.utils.InPlaceOperator.clear;
import static ml.utils.InPlaceOperator.minusAssign;
import static ml.utils.InPlaceOperator.timesAssign;
import static ml.utils.Printer.disp;
import static ml.utils.Printer.err;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Time.tic;
import static ml.utils.Time.toc;
import static ml.utils.Utility.exit;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import la.decomposition.EigenValueDecomposition;
import la.decomposition.LUDecomposition;
import la.decomposition.QRDecomposition;
import la.decomposition.SingularValueDecomposition;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import ml.random.MultivariateGaussianDistribution;

public class Matlab {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double[] Vec = new double[] {4, 2, 3, 6, 1, 8, 5, 9, 7};
		double start = 0;
		start = tic();
		disp(max(Vec));
		System.out.format("Elapsed time: %.9f seconds.%n", toc(start));
		start = tic();
		double max = Vec[0];
		for (int i = 1; i < Vec.length; i++) {
			if (max < Vec[i])
				max = Vec[i];
		}
		disp(max);
		System.out.format("Elapsed time: %.9f seconds.%n", toc(start));
		
		double[][] data = { {10d,-5d,0d,3d}, {2d,0d,1d,2d}, {1d,6d,0d,5d}};
		Matrix A = new DenseMatrix(data);
		disp(sigmoid(A));
		
		tic();
		/*
		 * Allocate 1000 x 1000 matrix costs 0.02 seconds!
		 * In effect, during the iteration of an algorithm,
		 * memory allocation for temporary matrix variables
		 * will cost considerable time. OpenCV is right. Never
		 * allocate memory in the inner part of a function.
		 * 
		 * Avoid implicitly use the garbage collection from
		 * within the Java virtual machine.
		 */
		for (int i = 0; i < 0; i++) {
			A = new DenseMatrix(1000, 1000);
		}
		System.out.format("Elapsed time: %.9f seconds.%n", toc());
		
		int m = 4;
		int n = 3;
		A = hilb(m, n);
		
		int[] rIndices = new int[] {0, 1, 3, 1, 2, 2, 3, 2, 3};
		int[] cIndices = new int[] {0, 0, 0, 1, 1, 2, 2, 3, 3};
		double[] values = new double[] {10, 3.2, 3, 9, 7, 8, 8, 7, 7};
		int numRows = 4;
		int numColumns = 4;
		int nzmax = rIndices.length;

		A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		
		fprintf("A:%n");
		printMatrix(A);
		fprintf("sum(A):%n");
		disp(sum(A));
		fprintf("sum(A, 2):%n");
		disp(sum(A, 2));
		
		disp("mean(A):");
		disp(mean(A, 1));
		
		disp("std(A):");
		disp(std(A, 0, 1));
		
		fprintf("max(A):%n");
		disp(max(A)[0]);
		fprintf("max(A, 2):%n");
		disp(max(A, 2)[0]);
		
		fprintf("min(A):%n");
		disp(min(A)[0]);
		fprintf("min(A, 2):%n");
		disp(min(A, 2)[0]);
		
		A = full(A);
		fprintf("A:%n");
		disp(A);
		fprintf("sum(A):%n");
		disp(sum(A));
		fprintf("sum(A, 2):%n");
		disp(sum(A, 2));
		
		fprintf("max(A):%n");
		disp(max(A)[0]);
		fprintf("max(A, 2):%n");
		disp(max(A, 2)[0]);
		
		fprintf("min(A):%n");
		disp(min(A)[0]);
		fprintf("min(A, 2):%n");
		disp(min(A, 2)[0]);
		
		fprintf("A'A:%n");
		disp(A.transpose().mtimes(A));
		Matrix[] VD = EigenValueDecomposition.decompose(A.transpose().mtimes(A));
		
		Matrix V = VD[0];
		Matrix D = VD[1];
		
		fprintf("V:%n");
		printMatrix(V);
		
		fprintf("D:%n");
		printMatrix(D);
		
		fprintf("VDV':%n");
		disp(V.mtimes(D).mtimes(V.transpose()));

		fprintf("A'A:%n");
		printMatrix(A.transpose().mtimes(A));

		fprintf("V'V:%n");
		printMatrix(V.transpose().mtimes((V)));
		
		fprintf("norm(A, 2):%n");
		disp(norm(A, 2));
		
		fprintf("rank(A):%n");
		disp(rank(A));
		
		Vector V1 = new SparseVector(3);
		V1.set(1, 1);
		V1.set(2, -1);
		disp("V1:");
		disp(V1);
		
		Vector V2 = new SparseVector(3);
		V2.set(0, -1);
		V2.set(2, 1);
		disp("V2:");
		disp(V2);
		
		fprintf("max(V1, -1)%n");
		disp(max(V1, -1));
		
		fprintf("max(V1, 1)%n");
		disp(max(V1, 1));
		
		fprintf("max(V1, 0)%n");
		disp(max(V1, 0));
		
		fprintf("max(V1, V2)%n");
		disp(max(V1, V2));
		
		fprintf("min(V1, -1)%n");
		disp(min(V1, -1));
		
		fprintf("min(V1, 1)%n");
		disp(min(V1, 1));
		
		fprintf("min(V1, 0)%n");
		disp(min(V1, 0));
		
		fprintf("min(V1, V2)%n");
		disp(min(V1, V2));
		
		A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		
		disp("A:");
		printMatrix(A);
		
		Vector[] Vs = Matlab.sparseMatrix2SparseRowVectors(A);
		Matrix S = Matlab.sparseRowVectors2SparseMatrix(Vs);
		disp("S:");
		printMatrix(S);
		
		Vs = Matlab.sparseMatrix2SparseColumnVectors(S);
		S = Matlab.sparseColumnVectors2SparseMatrix(Vs);
		disp("S:");
		printMatrix(S);
		
		Matrix B = sparse(new DenseMatrix(size(A), 2).times(A.transpose()));
		disp("B:");
		printMatrix(B);
		
		disp("max(A, 5)");
		printMatrix(max(A, 5d));
		
		disp("max(A, -2)");
		printMatrix(max(A, -2d));
		
		disp("max(A, B)");
		printMatrix(max(A, B));
		
		disp("A:");
		printMatrix(A);
		
		disp("B:");
		printMatrix(B);
		
		disp("min(A, 5)");
		printMatrix(min(A, 5d));
		
		disp("min(A, -2)");
		printMatrix(min(A, -2d));
		
		disp("min(A, B)");
		printMatrix(min(A, B));
		
		disp("A:");
		printMatrix(A);
		
		Matrix[] sortRes = sort((A), 1, "ascend");
		disp("Sorted values:");
		printMatrix(sortRes[0]);
		disp("Sorted indices:");
		disp(sortRes[1]);
		
		Vector V3 = new SparseVector(8);
		V3.set(1, 6);
		V3.set(2, -2);
		V3.set(4, 9);
		V3.set(6, 8);
		disp("V3:");
		disp(V3);
		double[] IX = sort(V3, "ascend");
		disp("Sorted V3:");
		disp(V3);
		disp("Sorted indices:");
		disp(IX);
		
		/*disp("A:");
		A.setEntry(2, 3, 0);
		A.setEntry(3, 3, 0);
		printMatrix(A);
		printMatrix(A.getSubMatrix(1, 3, 1, 3));
		printMatrix(getRows(A, 1, 3, 2));
		printMatrix(getColumns(A, 1, 3));
		printMatrix(getColumns(A, new int[] {3, 2}));*/
		
		disp("A:");
		printMatrix(A);
		disp("repmat(A, 2, 3):");
		printMatrix(repmat(A, 2, 3));
		disp("vec(A)");
		disp(vec(A));
		disp("reshape(vec(A), 4, 4)");
		printMatrix(reshape(vec(A), 4, 4));
		
		A = full(A);
		disp("full(A)");
		printMatrix(A);
		disp("repmat(A, 2, 3):");
		disp(repmat(A, 2, 3));
		disp("vec(A)");
		disp(vec(A));
		disp("reshape(vec(A), 4, 4)");
		disp(reshape(vec(A), 4, 4));
		
		B = new DenseMatrix(new double[][] {{3, 2}, {0, 2}});
		
		disp("sparse(A)");
		printMatrix(sparse(A));
		disp("sparse(B)");
		printMatrix(sparse(B));
		
		printMatrix(kron(full(A), full(B)));
		printMatrix(kron(full(A), sparse(B)));
		printMatrix(kron(sparse(A), full(B)));
		printMatrix(kron(sparse(A), sparse(B)));
		
		/*printMatrix(A);
		printMatrix(A.getSubMatrix(1, 3, 1, 3));
		printMatrix(getRows(A, 1, 3, 2));
		printMatrix(getColumns(A, 1, 3));
		printMatrix(getColumns(A, new int[] {3, 2}));*/
		
	}
	
	/**
	 * A constant holding the smallest positive
	 * nonzero value of type double, 2-1074.
	 */
	public static double eps = Double.MIN_VALUE;
	
	/**
	 * A constant holding the positive infinity of type double.
	 */
	public static double inf = Double.POSITIVE_INFINITY;
	
	/**
	 * Compute the mean for rows or columns of a real matrix.
	 * 
	 * @param A a real matrix
	 * 
	 * @param dim direction, 1 for column-wise, 2 for row-wise
	 *
	 * @return mean(A, dim)
	 */
	/*public static Vector mean(Matrix A, int dim) {
		if (dim != 1 && dim != 2) {
			err("dim should be 1 or 2.");
			exit(1);
		}
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (M == 1) { // A is a row matrix
			if (dim == 1) {
				return A.getRowVector(0);
			} else if (dim == 2) {
				double mean = sumAll(A) / N;
				return new DenseVector(new double[] {mean});
			}
		} else if (N == 1) { // A is a column matrix
			if (dim == 2) {
				return A.getColumnVector(0);
			} else if (dim == 1) {
				double mean = sumAll(A) / M;
				return new DenseVector(new double[] {mean});
			}
		} else { // A is a normal matrix
			Vector res = sum(A, dim);
			if (dim == 1)
				timesAssign(res, 1.0 / M);
			else if (dim == 2)
				timesAssign(res, 1.0 / N);
		}
	}*/
	
	/**
	 * Compute standard deviation.
	 * 
	 * @param X a real matrix
	 * 
	 * @param flag 0: n - 1 in the divisor, 1: n in the divisor
	 * 
	 * @param dim direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @return a real dense vector
	 */
	public static DenseVector std(Matrix X, int flag, int dim) {
		if (dim != 1 && dim != 2) {
			err("dim should be 1 or 2.");
			exit(1);
		}
		int M = X.getRowDimension();
		int N = X.getColumnDimension();
		DenseVector mean = mean(X, dim);
		Matrix meanMat = null;
		Matrix temp = null;
		if (dim == 1) {
			meanMat = rowVector2RowMatrix(mean);
			temp = repmat(meanMat, M, 1);
		} else {
			meanMat = columnVector2ColumnMatrix(mean);
			temp = repmat(meanMat, 1, N);
		}
		minusAssign(temp, X);
		timesAssign(temp, temp);
		int num = size(X, dim);
		double[] res = sum(temp, dim).getPr();
		if (flag == 0) { // flag = 0
			if (num == 1)
				clear(res);
			else
				divideAssign(res, num - 1);
		} else if (flag == 1) { // flag = 1
			if (num != 1)
				divideAssign(res, num);
		}
		for (int k = 0; k < res.length; k++)
			res[k] = Math.sqrt(res[k]);
		return new DenseVector(res);
	}
	
	/**
	 * Compute standard deviation for columns of a real matrix.
	 * 
	 * @param X a real matrix
	 * 
	 * @param flag 0: n - 1 in the divisor, 1: n in the divisor
	 * 
	 * @return a real dense vector
	 */
	public static DenseVector std(Matrix X, int flag) {
		return std(X, flag, 1);
	}
	
	/**
	 * Compute standard deviation for columns of a real matrix with
	 * n - 1 in the divisor.
	 * 
	 * @param X a real matrix
	 * 
	 * @return a real dense vector
	 */
	public static DenseVector std(Matrix X) {
		return std(X, 0);
	}
	
	/**
	 * Set submatrix of A with selected rows and selected columns by elements of B.
	 * B should have the same shape to the submatrix of A to be set. It is equivalent
	 * to the syntax A(selectedRows, selectedColumns) = B.
	 * 
	 * @param A a matrix whose submatrix is to be set
	 * 
	 * @param selectedRows {@code int[]} holding indices of selected rows
	 * 
	 * @param selectedColumns {@code int[]} holding indices of selected columns
	 * 
	 * @param B a matrix to set the submatrix of A
	 * 
	 */
	public static void setSubMatrix(Matrix A, int[] selectedRows, 
			int[] selectedColumns, Matrix B) {

		int r, c;
		for (int i = 0; i < selectedRows.length; i++) {
			for (int j = 0; j < selectedColumns.length; j++) {
				r = selectedRows[i];
				c = selectedColumns[j];
				A.setEntry(r, c, B.getEntry(i, j));
			}
		}

	}
	
	/**
	 * Reshape a matrix to a new shape specified by a two dimensional
	 * integer array.
	 * 
	 * @param A a matrix
	 * 
	 * @param size a two dimensional integer array describing a new shape
	 * 
	 * @return a new matrix with a shape specified by size 
	 * 
	 */
	public static Matrix reshape(Matrix A, int[] size) {

		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
		}

		int M = size[0];
		int N = size[1];

		if (M * N != A.getRowDimension() * A.getColumnDimension()) {
			System.err.println("Wrong shape!");
			exit(1);
		}
		
		Matrix res = null;
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[][] resData = new double[M][];
			double[] resRow = null;
			int r, c;
			for (int i = 0, shiftI = 0; i < M; i++, shiftI++) {
				resData[i] = new double[N];
				resRow = resData[i];
				for (int j = 0, shiftJ = shiftI; j < N; j++, shiftJ += M) {
					r = shiftJ % A.getRowDimension();
					c = shiftJ / A.getRowDimension();
					resRow[j] = AData[r][c];
				}
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			int nnz = ((SparseMatrix) A).getNNZ();
			int[] resIr = new int[nnz];
			int[] resJc = new int[N + 1];
			double[] resPr = new double[nnz];
			System.arraycopy(pr, 0, resPr, 0, nnz);
			int lastColIdx = -1;
			int currentColIdx = 0;
			int idx = 0;
			for (int j = 0, shiftJ = 0; j < A.getColumnDimension(); j++, shiftJ += A.getRowDimension()) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					idx = ir[k] + shiftJ;
					currentColIdx = idx / M;
					resIr[k] = idx % M;
					while (lastColIdx < currentColIdx) {
						resJc[lastColIdx + 1] = k;
						lastColIdx++;
					}
				}
			}
			resJc[N] = nnz;
			res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, M, N, nnz);
		}

		return res;

	}
	
	/**
	 * Reshape a matrix to a new shape specified number of rows and
	 * columns.
	 * 
	 * @param A a matrix
	 * 
	 * @param M number of rows of the new shape
	 * 
	 * @param N number of columns of the new shape
	 * 
	 * @return a new M-by-N matrix whose elements are taken columnwise
	 *         from A
	 * 
	 */
	public static Matrix reshape(Matrix A, int M, int N) {
		return reshape(A, new int[]{M, N});
	}
	
	/**
	 * Reshape a vector to a matrix with a shape specified by a two dimensional
	 * integer array.
	 * 
	 * @param V a vector
	 * 
	 * @param size a two dimensional integer array describing a new shape
	 * 
	 * @return a new matrix with a shape specified by size 
	 * 
	 */
	public static Matrix reshape(Vector V, int[] size) {
		
		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
			exit(1);
		}

		int dim = V.getDim();

		if (size[0] * size[1] != dim) {
			System.err.println("Wrong shape!");
		}
		
		Matrix res = null;
		
		int M = size[0];
		int N = size[1];
		
		if (V instanceof DenseVector) {
			double[][] resData = new double[M][];
			double[] resRow = null;
			double[] pr = ((DenseVector) V).getPr();
			for (int i = 0, shiftI = 0; i < M; i++, shiftI++) {
				resData[i] = new double[N];
				resRow = resData[i];
				for (int j = 0, shiftJ = shiftI; j < N; j++, shiftJ += M) {
					resRow[j] = pr[shiftJ];
				}
			}
			res = new DenseMatrix(resData);
		} else if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			int[] resIr = new int[nnz];
			int[] resJc = new int[N + 1];
			double[] resPr = new double[nnz];
			System.arraycopy(pr, 0, resPr, 0, nnz);
			int lastColIdx = -1;
			int currentColIdx = 0;
			int idx = 0;
			for (int k = 0; k < nnz; k++) {
				idx = ir[k];
				currentColIdx = idx / M;
				resIr[k] = idx % M;
				while (lastColIdx < currentColIdx) {
					resJc[lastColIdx + 1] = k;
					lastColIdx++;
				}
			}
			resJc[N] = nnz;
			res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, M, N, nnz);
		}
		
		return res;
		
	}
	
	/**
	 * Reshape a vector to a new shape specified number of rows and
	 * columns.
	 * 
	 * @param V a vector
	 * 
	 * @param M number of rows of the new shape
	 * 
	 * @param N number of columns of the new shape
	 * 
	 * @return a new M-by-N matrix whose elements are taken from V
	 * 
	 */
	public static Matrix reshape(Vector V, int M, int N) {
		
		int dim = V.getDim();

		if (M * N != dim) {
			System.err.println("Wrong shape!");
		}
		
		Matrix res = null;
		
		if (V instanceof DenseVector) {
			double[][] resData = new double[M][];
			double[] resRow = null;
			double[] pr = ((DenseVector) V).getPr();
			for (int i = 0, shiftI = 0; i < M; i++, shiftI++) {
				resData[i] = new double[N];
				resRow = resData[i];
				for (int j = 0, shiftJ = shiftI; j < N; j++, shiftJ += M) {
					resRow[j] = pr[shiftJ];
				}
			}
			res = new DenseMatrix(resData);
		} else if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			int[] resIr = new int[nnz];
			int[] resJc = new int[N + 1];
			double[] resPr = new double[nnz];
			System.arraycopy(pr, 0, resPr, 0, nnz);
			int lastColIdx = -1;
			int currentColIdx = 0;
			int idx = 0;
			for (int k = 0; k < nnz; k++) {
				idx = ir[k];
				currentColIdx = idx / M;
				resIr[k] = idx % M;
				while (lastColIdx < currentColIdx) {
					resJc[lastColIdx + 1] = k;
					lastColIdx++;
				}
			}
			resJc[N] = nnz;
			res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, M, N, nnz);
		}
		
		return res;
		
	}
	
	/**
	 * Vectorize a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return Vectorization of a matrix A
	 */
	public static Matrix vec(Matrix A) {
		
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		
		if (N == 1) {
			return A;
		}
		
		Matrix res = null;
		int dim = M * N;
		if (A instanceof DenseMatrix) {
			double[][] resData = new double[dim][];
			double[][] AData = ((DenseMatrix) A).getData();
			for (int j = 0, shift = 0; j < N; j++, shift += M) {
				for (int i = 0, shiftI = shift; i < M; i++, shiftI++) {
					resData[shiftI] = new double[] {AData[i][j]};
				}
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			int nnz = ((SparseMatrix) A).getNNZ();
			int[] resIr = new int[nnz];
			int[] resJc = new int[] {0, nnz};
			double[] resPr = new double[nnz];
			System.arraycopy(pr, 0, resPr, 0, nnz);
			int cnt = 0;
			for (int j = 0, shift = 0; j < N; j++, shift += M) {
				for (int k = jc[j]; k <jc[j + 1]; k++) {
					resIr[cnt++] = ir[k] + shift;
				}
			}
			res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, dim, 1, nnz);
		}
		
		return res;
		
	}
	
	/**
	 * Compute the "economy size" matrix singular value decomposition.
	 * 
	 * @param A a real matrix
	 * 
	 * @return a matrix array [U, S, V] where U is left orthonormal matrix, S is a 
	 * 		   a diagonal matrix, and V is the right orthonormal matrix such that 
	 *         A = U * S * V'
	 * 
	 */
	public static Matrix[] svd(Matrix A) {
		
		SingularValueDecomposition svdImpl = new SingularValueDecomposition(A);
		Matrix U = svdImpl.getU();
		Matrix S = svdImpl.getS();
		Matrix V = svdImpl.getV();
		
		Matrix[] res = new Matrix[3];
		res[0] = U;
		res[1] = S;
		res[2] = V;
		
		return res;
		
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and covariance SIGMA.
	 * 
	 * @param MU 1 x d mean vector
	 * 
	 * @param SIGMA covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static Matrix mvnrnd(Matrix MU, Matrix SIGMA, int cases) {
		return MultivariateGaussianDistribution.mvnrnd(MU, SIGMA, cases);
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and covariance SIGMA.
	 * 
	 * @param MU a 1D {@code double} array holding the mean vector
	 * 
	 * @param SIGMA a 2D {@code double} array holding the covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static Matrix mvnrnd(double[] MU, double[][] SIGMA, int cases) {
		return mvnrnd(new DenseMatrix(MU, 2), new DenseMatrix(SIGMA), cases);
	}
	
	/**
	 * Generate random samples chosen from the multivariate Gaussian 
	 * distribution with mean MU and a diagonal covariance SIGMA.
	 * 
	 * @param MU a 1D {@code double} array holding the mean vector
	 * 
	 * @param SIGMA a 1D {@code double} array holding the diagonal elements
	 *        of the covariance matrix
	 * 
	 * @param cases number of d dimensional random samples
	 * 
	 * @return cases-by-d sample matrix subject to the multivariate 
	 *         Gaussian distribution N(MU, SIGMA)
	 *         
	 */
	public static Matrix mvnrnd(double[] MU, double[] SIGMA, int cases) {
		return mvnrnd(new DenseMatrix(MU, 2), diag(SIGMA), cases);
	}
	
	
	/**
	 * Replicate and tile an array. 
	 * 
	 * @param A a matrix
	 * 
	 * @param M number of rows to replicate
	 * 
	 * @param N number of columns to replicate
	 * 
	 * @return repmat(A, M, N)
	 * 
	 */
	public static Matrix repmat(Matrix A, int M, int N) {

		Matrix res = null;
		int nRow = M * A.getRowDimension();
		int nCol = N * A.getColumnDimension();
		if (A instanceof DenseMatrix) {
			double[][] resData = allocate2DArray(nRow, nCol);
			double[] resRow = null;
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			int r;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				r = i % A.getRowDimension();
				ARow = AData[r];
				for (int k = 0, shift = 0; k < N; k++, shift += A.getColumnDimension()) {
					System.arraycopy(ARow, 0, resRow, shift, A.getColumnDimension());
				}
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			int nnz = ((SparseMatrix) A).getNNZ();
			
			int resNNZ = nnz * N * M;
			int[] resIr = new int[resNNZ];
			int[] resJc = new int[nCol + 1];
			double[] resPr = new double[resNNZ];
			int[] nnzPerColumn = new int[A.getColumnDimension()];
			for (int j = 0; j < A.getColumnDimension(); j++) {
				nnzPerColumn[j] = M * (jc[j + 1] - jc[j]);
			}
			resJc[0] = 0;
			for (int c = 0; c < nCol; c++) {
				int j = c % A.getColumnDimension();
				resJc[c + 1] = resJc[c] + nnzPerColumn[j];
			}
			for (int j = 0, shiftA = 0; j < A.getColumnDimension(); j++) {
				int numNNZACol_j = (jc[j + 1] - jc[j]);
				int[] irACol_j = new int[numNNZACol_j];
				for (int k = 0, shift = shiftA * M; k < N; k++, shift += nnz * M) {
					System.arraycopy(ir, shiftA, irACol_j, 0, numNNZACol_j);
					for (int i = 0, shift2 = shift; i < M; i++, shift2 += numNNZACol_j) {
						System.arraycopy(irACol_j, 0, resIr, shift2, numNNZACol_j);
						if (i < M - 1)
							for (int t = 0; t < numNNZACol_j; t++)
								irACol_j[t] += A.getRowDimension();
						System.arraycopy(pr, shiftA, resPr, shift2, numNNZACol_j);
					}
				}
				shiftA += numNNZACol_j;
			}
			res = SparseMatrix.createSparseMatrixByCSCArrays(resIr, resJc, resPr, nRow, nCol, resNNZ);
		}
		return res;

	}
	
	/**
	 * Replicate and tile an array. 
	 * 
	 * @param A a matrix
	 * 
	 * @param size a int[2] vector [M N]
	 * 
	 * @return repmat(A, size)
	 * 
	 */
	public static Matrix repmat(Matrix A, int[] size) {
		return repmat(A, size[0], size[1]);
	}
	
	/**
	 * M = mean(A, dim) returns the mean values for elements 
	 * along the dimension of A specified by scalar dim. 
	 * For matrices, mean(A, 2) is a column vector containing 
	 * the mean value of each row.
	 *
	 * @param X a real matrix
	 * 
	 * @param dim dimension order
	 * 
	 * @return mean(A, dim)
	 * 
	 */
	public static DenseVector mean(Matrix X, int dim) {
		int N = size(X, dim);
		double[] S = sum(X, dim).getPr();
		divideAssign(S, N);
		return new DenseVector(S);
	}
	
	/**
	 * Compute eigenvalues and eigenvectors of a symmetric real matrix.
	 * 
	 * @param A a symmetric real matrix
	 * 
	 * @param K number of eigenvalues selected
	 * 
	 * @param sigma either "lm" (largest magnitude) or "sm" (smallest magnitude)
	 * 
	 * @return a matrix array [V, D], V is the selected K eigenvectors (normalized 
	 *         to 1), and D is a diagonal matrix holding selected K eigenvalues.
	 *         
	 */
	public static Matrix[] eigs(Matrix A, int K, String sigma) {
		
		EigenValueDecomposition eigImpl = new EigenValueDecomposition(A, 1e-6);
		Matrix eigV = eigImpl.getV();
		Matrix eigD = eigImpl.getD();
		
		/*disp(eigV);
		disp(eigD);*/
		
		int N = A.getRowDimension();
		Matrix[] res = new Matrix[2];
		
		Vector eigenValueVector = new DenseVector(K);
		Matrix eigenVectors = null;
		if (sigma.equals("lm")) {
			for (int k = 0; k < K; k++)
				eigenValueVector.set(k, eigD.getEntry(k, k));
			eigenVectors = eigV.getSubMatrix(0, N - 1, 0, K - 1);
		} else if (sigma.equals("sm")) {
			for (int k = 0; k < K; k++)
				eigenValueVector.set(k, eigD.getEntry(N - 1 - k, N - 1 - k));
			// eigenVectors = new DenseMatrix(N, K);
			double[][] eigenVectorsData = allocate2DArray(N, K);
			double[][] eigVData = ((DenseMatrix) eigV).getData();
			double[] eigenVectorsRow = null;
			double[] eigVRow = null;
			// eigenVectors.setColumnVector(k, eigV.getColumnVector(j));
			for (int i = 0; i < N; i++) {
				eigenVectorsRow = eigenVectorsData[i];
				eigVRow = eigVData[i];
				for(int j = N - 1, k = 0; k < K ; j--, k++) {
					eigenVectorsRow[k] = eigVRow[j];
				}
			}
			eigenVectors = new DenseMatrix(eigenVectorsData);
		} else {
			System.err.println("sigma should be either \"lm\" or \"sm\"");
			System.exit(-1);
		}
		
		res[0] = eigenVectors;
		res[1] = diag(eigenValueVector);
		
		return res;
		
	}
	
	/**
	 * Generate a diagonal matrix with its elements of a 1D {@code double} 
	 * array on its main diagonal.
	 *   
	 * @param V a 1D {@code double} array holding the diagonal elements
	 * 
	 * @return diag(V)
	 * 
	 */
	public static Matrix diag(double[] V) {

		int d = V.length;
		Matrix res = new SparseMatrix(d, d);

		for (int i = 0; i < d; i++) {
			res.setEntry(i, i, V[i]);
		}

		return res;

	}
	
	/**
	 * Calculate the element-wise logarithm of a matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return log(A)
	 * 
	 */
	public static DenseMatrix log(Matrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		double[][] resData = allocate2DArray(nRow, nCol);
		double[] resRow = null;
			
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				ARow = AData[i];
				for (int j = 0; j < nCol; j++) {
					resRow[j] = Math.log(ARow[j]);
				}
			}
		} else if (A instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) A).getIc();
			int[] jr = ((SparseMatrix) A).getJr();
			int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
			double[] pr = ((SparseMatrix) A).getPr();
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				if (jr[i + 1] == jr[i]) {
					assign(resRow, Double.NEGATIVE_INFINITY);
					continue;
				}
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					currentIdx = ic[k];
					for (int j = lastIdx + 1; j < currentIdx; j++) {
						resRow[j] = Double.NEGATIVE_INFINITY;
					}
					resRow[currentIdx] = Math.log(pr[valCSRIndices[k]]);
					lastIdx = currentIdx;
				}
				for (int j = lastIdx + 1; j < nCol; j++) {
					resRow[j] = Double.NEGATIVE_INFINITY;
				}
			}
		}

		/*for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, Math.log(A.getEntry(i, j)));
			}
		}*/

		return new DenseMatrix(resData);

	}
	
	/**
	 * Calculate TFIDF of a doc-term-count matrix, each column
	 * is a data sample.
	 * 
	 * @param docTermCountMatrix a matrix, each column is a data example
	 * 
	 * @return TFIDF of docTermCountMatrix
	 * 
	 */
	public static Matrix getTFIDF(Matrix docTermCountMatrix) {

		final int NTerm = docTermCountMatrix.getRowDimension();
		final int NDoc = docTermCountMatrix.getColumnDimension();

		// Get TF vector
		double[] tfVector = new double[NTerm];
		for (int i = 0; i < docTermCountMatrix.getRowDimension(); i++) {
			tfVector[i] = 0;
			for (int j = 0; j < docTermCountMatrix.getColumnDimension(); j++) {
				tfVector[i] += docTermCountMatrix.getEntry(i, j) > 0 ? 1 : 0;
			}
		}

		Matrix res = docTermCountMatrix.copy();
		for (int i = 0; i < docTermCountMatrix.getRowDimension(); i++) {
			for (int j = 0; j < docTermCountMatrix.getColumnDimension(); j++) {
				if (res.getEntry(i, j) > 0) {
					res.setEntry(i, j, res.getEntry(i, j) * (tfVector[i] > 0 ? Math.log(NDoc / tfVector[i]) : 0));
				}
			}
		}

		return res;

	}
	
	/**
	 * Normalize A by columns.
	 * 
	 * @param A a matrix
	 * 
	 * @return a column-wise normalized matrix
	 */
	public static Matrix normalizeByColumns(Matrix A) {
		double[] AA = full(sqrt(sum(A.times(A)))).getPr();
		Matrix res = A.copy();
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] /= AA[j];
				}
			}
		} else if (res instanceof SparseMatrix) {
			// int[] ir = ((SparseMatrix) res).getIr();
			int[] jc = ((SparseMatrix) res).getJc();
			double[] pr = ((SparseMatrix) res).getPr();
			double v = 0;
			for (int j = 0; j < N; j++) {
				v = AA[j];
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					pr[k] /= v;
				}
			}
		}
		return res;
	}
	
	/**
	 * Random permutation. 
	 * </br>
	 * randperm(n) returns a random permutation of the integers 1:n.
	 * 
	 * @param n an integer
	 * 
	 * @return randperm(n)
	 */
	public static int[] randperm(int n) {
		
		int[] res = new int[n];
		
		Set<Integer> leftSet = new TreeSet<Integer>();
		for (int i = 0; i < n; i++) {
			leftSet.add(i);
		}
		
		Random generator = new Random();
		for (int i = 0; i < n; i++) {
			double[] uniformDist = allocateVector(n - i, 1.0 / (n - i));
			
			double rndRealScalor = generator.nextDouble();
			double sum = 0;
			for (int j = 0, k = 0; j < n; j++) {
				if (!leftSet.contains(j))
					continue;
				sum += uniformDist[k];
				if (rndRealScalor <= sum) {
					res[i] = j + 1;
					leftSet.remove(j);
					break;
				} else {
					k++;
				}
			}
		}
		
		return res;

	}
	
	/**
	 * Find nonzero elements and return their indices.
	 * 
	 * @param V a real vector
	 * 
	 * @return an integer array of indices of nonzero elements of V
	 * 
	 */
	public static int[] find(Vector V) {
		
		int[] indices = null;
		if (V instanceof DenseVector) {
			ArrayList<Integer> idxList = new ArrayList<Integer>();
			double[] pr = ((DenseVector) V).getPr();
			double v = 0;
			for (int k = 0; k < V.getDim(); k++) {
				v = pr[k];
				if (v != 0) {
					idxList.add(k);
				}
			}
			int nnz = idxList.size();
			indices = new int[nnz];
			Iterator<Integer> idxIter = idxList.iterator();
			int cnt = 0;
			while (idxIter.hasNext()) {
				indices[cnt++] = idxIter.next();
			}
		} else if (V instanceof SparseVector) {
			((SparseVector) V).clean();
			int nnz = ((SparseVector) V).getNNZ();
			int[] ir = ((SparseVector) V).getIr();
			indices = new int[nnz];
			System.arraycopy(ir, 0, indices, 0, nnz);
		}

		return indices;

	}
	
	/**
	 * Find nonzero elements and return their value, row and column indices.
	 * 
	 * @param A a matrix
	 * 
	 * @return a {@code FindResult} data structure which has three instance
	 * data members:<br/>
	 * rows: row indices array for non-zero elements of a matrix<br/>
	 * cols: column indices array for non-zero elements of a matrix<br/>
	 * vals: values array for non-zero elements of a matrix<br/>
	 *         
	 */
	public static FindResult find(Matrix A) {
		int[] rows = null;
		int[] cols = null;
		double[] vals = null;
		if (A instanceof SparseMatrix) {
			((SparseMatrix) A).clean();
			int nnz = ((SparseMatrix) A).getNNZ();
			rows = new int[nnz];
			cols = new int[nnz];
			vals = new double[nnz];
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			double[] pr = ((SparseMatrix) A).getPr();
			int cnt = 0;
			for (int j = 0; j < A.getColumnDimension(); j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					rows[cnt] = ir[k];
					cols[cnt] = j;
					vals[cnt] = pr[k];
					cnt++;
				}
			}
		} else if (A instanceof DenseMatrix) {
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			ArrayList<Integer> rowIdxList = new ArrayList<Integer>();
			ArrayList<Integer> colIdxList = new ArrayList<Integer>();
			ArrayList<Double> valList = new ArrayList<Double>();
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double v = 0;
			for (int i = 0; i < M; i++) {
				ARow = AData[i];
				for (int j = 0; j < N; j++) {
					v = ARow[j];
					if (v != 0) {
						rowIdxList.add(i);
						colIdxList.add(j);
						valList.add(v);
					}
				}
			}
			int nnz = valList.size();
			rows = new int[nnz];
			cols = new int[nnz];
			vals = new double[nnz];
			Iterator<Integer> rowIdxIter = rowIdxList.iterator();
			Iterator<Integer> colIdxIter = colIdxList.iterator();
			Iterator<Double> valIter = valList.iterator();
			int cnt = 0;
			while (valIter.hasNext()) {
				rows[cnt] = rowIdxIter.next();
				cols[cnt] = colIdxIter.next();
				vals[cnt] = valIter.next();
				cnt++;
			}
		}
		return new FindResult(rows, cols, vals);
		
	}
	
	/**
	 * Compute the element-wise exponential of a matrix
	 * 
	 * @param A a matrix
	 * 
	 * @return exp(A)
	 * 
	 */
	public static Matrix exp(Matrix A) {

		int M = A.getRowDimension();
		int N = A.getColumnDimension();

		Matrix res = new DenseMatrix(M, N, 1);
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		
		if (A instanceof DenseMatrix) {
			double[][] data = A.getData();
			double[] row = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				row = data[i];
				for (int j = 0; j < N; j++) {
					resRow[j] = Math.exp(row[j]);
				}
			}
		} else if (A instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) A).getPr();
			int[] ir = ((SparseMatrix) A).getIr();
			int[] jc = ((SparseMatrix) A).getJc();
			for (int j = 0; j < N; j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					resData[ir[k]][j] = Math.exp(pr[k]);
				}
			}
		}

		return res;

	}
	
	/**
	 * Get the subMatrix containing the elements of the specified rows.
	 * Rows are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param startRow initial row index (inclusive)
	 * 
	 * @param endRow final row index (inclusive)
	 * 
	 * @return the subMatrix of A containing the data of the specified rows
	 * 
	 */
	public static Matrix getRows(Matrix A, int startRow, int endRow) {
		/*Matrix res = null;
		if (A instanceof DenseMatrix) {
			double[][] resData = new double[endRow - startRow + 1][];
			double[][] AData = ((DenseMatrix) A).getData();
			for (int r = startRow, i = 0; r <= endRow; r++, i++) {
				resData[i] = AData[r].clone();
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			Vector[] vectors = sparseMatrix2SparseRowVectors(A);
			Vector[] resVectors = new Vector[endRow - startRow + 1];
			for (int r = startRow, i = 0; r <= endRow; r++, i++) {
				resVectors[i] = vectors[r];
			}
			res = sparseRowVectors2SparseMatrix(resVectors);
		}
		return res;*/
		return A.getRows(startRow, endRow);
	}
	
	/**
	 * Get the subMatrix containing the elements of the specified rows.
	 * Rows are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param selectedRows indices of selected rows
	 * 
	 * @return the subMatrix of A containing the data of the specified rows
	 * 
	 */
	public static Matrix getRows(Matrix A, int... selectedRows) {	
		/*Matrix res = null;
		if (A instanceof DenseMatrix) {
			double[][] resData = new double[selectedRows.length][];
			double[][] AData = ((DenseMatrix) A).getData();
			for (int i = 0; i < selectedRows.length; i++) {
				resData[i] = AData[selectedRows[i]].clone();
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			Vector[] vectors = sparseMatrix2SparseRowVectors(A);
			Vector[] resVectors = new Vector[selectedRows.length];
			for (int i = 0; i < selectedRows.length; i++) {
				resVectors[i] = vectors[selectedRows[i]];
			}
			res = sparseRowVectors2SparseMatrix(resVectors);
		}
		return res;*/
		return A.getRows(selectedRows);
	}
	
	/**
	 * Get the subMatrix containing the elements of the specified columns.
	 * Columns are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param startColumn initial column index (inclusive)
	 * 
	 * @param endColumn final column index (inclusive)
	 * 
	 * @return the subMatrix of A containing the data of the specified rows
	 * 
	 */
	public static Matrix getColumns(Matrix A, int startColumn, int endColumn) {
		Matrix res = null;
		int nRow = A.getRowDimension();
		int nCol = endColumn - startColumn + 1;
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double[][] resData = new double[nRow][nCol];
			double[] resRow = null;
			for (int r = 0; r < nRow; r++) {
				ARow = AData[r];
				resRow = new double[nCol];
				for (int c = startColumn, j = 0; c <= endColumn; c++, j++) {
					resRow[j] = ARow[c];
				}
				resData[r] = resRow;
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			Vector[] vectors = sparseMatrix2SparseColumnVectors(A);
			Vector[] resVectors = new Vector[nCol];
			for (int c = startColumn, j = 0; c <= endColumn; c++, j++) {
				resVectors[j] = vectors[c];
			}
			res = sparseColumnVectors2SparseMatrix(resVectors);
		}
		return res;
	}
	
	/**
	 * Get the subMatrix containing the elements of the specified columns.
	 * Columns are indicated counting from 0.
	 * 
	 * @param A a real matrix
	 * 
	 * @param selectedColumns indices of selected columns
	 * 
	 * @return the subMatrix of A containing the data of the specified columns
	 * 
	 */
	public static Matrix getColumns(Matrix A, int... selectedColumns) {
		Matrix res = null;
		int nRow = A.getRowDimension();
		int nCol = selectedColumns.length;
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			double[][] resData = new double[nRow][nCol];
			double[] resRow = null;
			for (int r = 0; r < nRow; r++) {
				ARow = AData[r];
				resRow = new double[nCol];
				for (int j = 0; j < nCol; j++) {
					resRow[j] = ARow[selectedColumns[j]];
				}
				resData[r] = resRow;
			}
			res = new DenseMatrix(resData);
		} else if (A instanceof SparseMatrix) {
			Vector[] vectors = sparseMatrix2SparseColumnVectors(A);
			Vector[] resVectors = new Vector[nCol];
			for (int j = 0; j < nCol; j++) {
				resVectors[j] = vectors[selectedColumns[j]];
			}
			res = sparseColumnVectors2SparseMatrix(resVectors);
		}
		return res;
	}
	
	/**
	 * Concatenate matrices vertically. All matrices in the argument
	 * list must have the same number of columns.
	 * 
	 * @param As matrices to be concatenated vertically
	 * 
	 * @return [A1; A2; ...]
	 * 
	 */
	public static Matrix vertcat(final Matrix ... As) {
		int nM = As.length;
		int nRow = 0;
		int nCol = 0;
		for (int i = 0; i < nM; i++) {
			if (As[i] == null)
				continue;
			nRow += As[i].getRowDimension();
			nCol = As[i].getColumnDimension();
		}
		
		for (int i = 1; i < nM; i++) {
			if (As[i] != null && nCol != As[i].getColumnDimension())
				System.err.println("Any matrix in the argument list should either " +
						"be empty matrix or have the same number of columns to the others!");
		}
		
		if (nRow == 0 || nCol == 0) {
			return null;
		}
		
		Matrix res = null;
		double[][] resData = new double[nRow][];
		double[] resRow = null;
		int idx = 0;
		for (int i = 0; i < nM; i++) {
			if (i > 0 && As[i - 1] != null)
				idx += As[i - 1].getRowDimension();
			if (As[i] == null)
				continue;
			if (As[i] instanceof DenseMatrix) {
				DenseMatrix A = (DenseMatrix) As[i];
				double[][] AData = A.getData();
				for (int r = 0; r < A.getRowDimension(); r++) {
					// res.setRow(idx + r, As[i].getRow(r));
					resData[idx + r] = AData[r].clone();
				}
			} else if (As[i] instanceof SparseMatrix) {
				SparseMatrix A = (SparseMatrix) As[i];
				double[] pr = A.getPr();
				int[] ic = A.getIc();
				int[] jr = A.getJr();
				int[] valCSRIndices = A.getValCSRIndices();
				for (int r = 0; r < A.getRowDimension(); r++) {
					resRow = allocate1DArray(nCol, 0);
					for (int k = jr[r]; k < jr[r + 1]; k++) {
						resRow[ic[k]] = pr[valCSRIndices[k]];
					}
					resData[idx + r] = resRow;
				}
			}
		}
		res = new DenseMatrix(resData);
		return res;
	}
	
	/**
	 * Concatenate matrices horizontally. All matrices in the argument
	 * list must have the same number of rows.
	 * 
	 * @param As matrices to be concatenated horizontally
	 * 
	 * @return [A1 A2 ...]
	 * 
	 */
	public static Matrix horzcat(final Matrix ... As) {
		int nM = As.length;
		int nCol = 0;
		int nRow = 0;
		for (int i = 0; i < nM; i++) {
			if (As[i] != null) {
				nCol += As[i].getColumnDimension();
				nRow = As[i].getRowDimension();
			}
		}
		
		for (int i = 1; i < nM; i++) {
			if (As[i] != null && nRow != As[i].getRowDimension())
				System.err.println("Any matrix in the argument list should either " +
						"be empty matrix or have the same number of rows to the others!");
		}
		
		if (nRow == 0 || nCol == 0) {
			return null;
		}
		
		Matrix res = new DenseMatrix(nRow, nCol, 0);
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		int idx = 0;
		
		for (int r = 0; r < nRow; r++) {
			resRow = resData[r];
			idx = 0;
			for (int i = 0; i < nM; i++) {
				if (i > 0 && As[i - 1] != null)
					idx += As[i - 1].getColumnDimension();
				if (As[i] == null)
					continue;
				if (As[i] instanceof DenseMatrix) {
					DenseMatrix A = (DenseMatrix) As[i];
					System.arraycopy(A.getData()[r], 0, resRow, idx, A.getColumnDimension());
				} else if (As[i] instanceof SparseMatrix) {
					SparseMatrix A = (SparseMatrix) As[i];
					double[] pr = A.getPr();
					int[] ic = A.getIc();
					int[] jr = A.getJr();
					int[] valCSRIndices = A.getValCSRIndices();
					for (int k = jr[r]; k < jr[r + 1]; k++) {
						resRow[idx + ic[k]] = pr[valCSRIndices[k]];
					}
				}
			}
		}
		
		return res;
		
	}
	
	/**
	 * Concatenate matrices along specified dimension.
	 * 
	 * @param dim specified dimension, can only be either 1 or 2 currently
	 * 
	 * @param As matrices to be concatenated
	 * 
	 * @return a concatenation of all the matrices in the argument list
	 * 
	 */
	public static Matrix cat(int dim, final Matrix... As) {
		Matrix res = null;
		if (dim == 1)
			res = vertcat(As);
		else if (dim == 2)
			res = horzcat(As);
		else
			System.err.println("Specified dimension can only be either 1 or 2 currently!");
		
		return res;
	}
	
	/**
	 * Compute the Kronecker tensor product of A and B
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return Kronecker product of A and B
	 * 
	 */
	public static Matrix kron(Matrix A, Matrix B) {
		Matrix res = null;
		int nRowLeft = A.getRowDimension();
		int nColLeft = A.getColumnDimension();
		int nRowRight = B.getRowDimension();
		int nColRight = B.getColumnDimension();
		if (A instanceof DenseMatrix && B instanceof DenseMatrix) {
			res = new DenseMatrix(nRowLeft * nRowRight, nColLeft * nColRight, 0);
			double[][] resData = res.getData();
			double[][] BData = B.getData();
			for (int i = 0, rShift = 0; i < nRowLeft; i++, rShift += nRowRight) {
				for (int j = 0, cShift = 0; j < nColLeft; j++, cShift += nColRight) {
					double A_ij = A.getEntry(i, j);
					if (A_ij == 0) {
						continue;
					}
					for (int p = 0; p < nRowRight; p++) {
						int r = rShift + p;
						double[] BRow = BData[p];
						double[] resRow = resData[r];
						for (int q = 0; q < nColRight; q++) {
							if (BRow[q] == 0) {
								continue;
							}
							int c = cShift + q;
							resRow[c] = A_ij * BRow[q];
						}
					}
				}
			}
		} else if (A instanceof DenseMatrix && B instanceof SparseMatrix) {
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
			
			int[] ir2 = ((SparseMatrix) B).getIr();
			int[] jc2 = ((SparseMatrix) B).getJc();
			double[] pr2 = ((SparseMatrix) B).getPr();
			
			for (int i = 0, rShift = 0; i < nRowLeft; i++, rShift += nRowRight) {
				for (int j = 0, cShift = 0; j < nColLeft; j++, cShift += nColRight) {
					double A_ij = A.getEntry(i, j);
					if (A_ij == 0) {
						continue;
					}
					for (int j2 = 0, c = cShift; j2 < nColRight; j2++, c++) {
						for (int k2 = jc2[j2]; k2 < jc2[j2 + 1]; k2++) {
							if (pr2[k2] == 0) {
								continue;
							}
							int r = rShift + ir2[k2];
							map.put(Pair.of(r, c), A_ij * pr2[k2]);
						}
					}
				}
			}
			res = SparseMatrix.createSparseMatrix(map, nRowLeft * nRowRight, nColLeft * nColRight);
		} else if (A instanceof SparseMatrix && B instanceof DenseMatrix) {
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
			
			int[] ir1 = ((SparseMatrix) A).getIr();
			int[] jc1 = ((SparseMatrix) A).getJc();
			double[] pr1 = ((SparseMatrix) A).getPr();
			
			double[][] BData = B.getData();
			for (int j1 = 0, cShift = 0; j1 < nColLeft; j1++, cShift += nColRight) {
				for (int k1 = jc1[j1]; k1 < jc1[j1 + 1]; k1++) {
					if (pr1[k1] == 0) {
						continue;
					}
					int rShift = ir1[k1] * nRowRight;
					for (int i = 0; i < nRowRight; i++) {
						double[] BRow = BData[i];
						int r = rShift + i;
						for (int j = 0; j < nColRight; j++) {
							if (BRow[j] == 0) {
								continue;
							}
							int c = cShift + j;
							map.put(Pair.of(r, c), pr1[k1] * BRow[j]);
						}
					}
				}
			}
			res = SparseMatrix.createSparseMatrix(map, nRowLeft * nRowRight, nColLeft * nColRight);
		} else if (A instanceof SparseMatrix && B instanceof SparseMatrix) {
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
			
			int[] ir1 = ((SparseMatrix) A).getIr();
			int[] jc1 = ((SparseMatrix) A).getJc();
			double[] pr1 = ((SparseMatrix) A).getPr();
			
			int[] ir2 = ((SparseMatrix) B).getIr();
			int[] jc2 = ((SparseMatrix) B).getJc();
			double[] pr2 = ((SparseMatrix) B).getPr();
			
			for (int j1 = 0, cShift = 0; j1 < nColLeft; j1++, cShift += nColRight) {
				for (int k1 = jc1[j1]; k1 < jc1[j1 + 1]; k1++) {
					if (pr1[k1] == 0) {
						continue;
					}
					int rShift = ir1[k1] * nRowRight;
					for (int j2 = 0, c = cShift; j2 < nColRight; j2++, c++) {
						for (int k2 = jc2[j2]; k2 < jc2[j2 + 1]; k2++) {
							if (pr2[k2] == 0) {
								continue;
							}
							int r = rShift + ir2[k2];
							map.put(Pair.of(r, c), pr1[k1] * pr2[k2]);
						}
					}
				}
			}
			res = SparseMatrix.createSparseMatrix(map, nRowLeft * nRowRight, nColLeft * nColRight);
		}
		return res;
	}
	
	/**
	 * Compute the sum of all elements of a matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return sum(sum(A))
	 * 
	 */
	public static double sumAll(Matrix A) {
		return sum(sum(A));
	}
	
	/**
	 * If A is a 1-row or 1-column matrix, then diag(A) is a
	 * sparse diagonal matrix with elements of A as its main diagonal,
	 * else diag(A) is a column matrix holding A's diagonal elements.
	 * 
	 * @param A a matrix
	 * 
	 * @return diag(A)
	 * 
	 */
	public static Matrix diag(Matrix A) {
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		Matrix res = null;

		if (nRow == 1) {
			res = new SparseMatrix(nCol, nCol);
			for (int i = 0; i < nCol; i++) {
				res.setEntry(i, i, A.getEntry(0, i));
			} 
		} else if (nCol == 1) {
			res = new SparseMatrix(nRow, nRow);
			for (int i = 0; i < nRow; i++) {
				res.setEntry(i, i, A.getEntry(i, 0));
			}
		} else if (nRow == nCol) {
			res = new DenseMatrix(nRow, 1);
			for (int i = 0; i < nRow; i++) {
				res.setEntry(i, 0, A.getEntry(i, i));
			}
		}

		return res;
	}
	
	/**
	 * Construct a sparse diagonal matrix from a vector.
	 * 
	 * @param V a real vector
	 * 
	 * @return diag(V)
	 */
	public static SparseMatrix diag(Vector V) {
		int dim = V.getDim();
		SparseMatrix res = new SparseMatrix(dim, dim);
		for (int i = 0; i < dim; i++) {
			res.setEntry(i, i, V.get(i));
		}
		return res;
	}
	
	/**
	 * Right array division.
	 * 
	 * @param A a matrix
	 * 
	 * @param v a scalar
	 * 
	 * @return A ./ v
	 */
	public static Matrix rdivide(Matrix A, double v) {
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		Matrix res = A.copy();
		
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				for (int j = 0; j < nCol; j++) {
					resRow[j] /= v;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			for (int k = 0; k < pr.length; k++) {
				pr[k] /= v;
			}
		}

		return res;
	}
	
	/**
	 * Generate an nRow-by-nCol matrix containing pseudo-random values drawn 
	 * from the standard uniform distribution on the open interval (0,1).
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return rand(nRow, nCol)
	 * 
	 */
	public static Matrix rand(int nRow, int nCol) {
		Random generator = new Random();
		Matrix res = new DenseMatrix(nRow, nCol);
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, generator.nextDouble());
			}
		}
		return res;
	}
	
	/**
	 * Generate an n-by-n matrix containing pseudo-random values drawn 
	 * from the standard uniform distribution on the open interval (0,1).
	 * 
	 * @param n number of rows or columns
	 * 
	 * @return rand(n, n)
	 * 
	 */
	public static Matrix rand(int n) {
		return rand(n, n);
	}
	
	/**
	 * Generate an nRow-by-nCol matrix containing pseudo-random values drawn 
	 * from the standard normal distribution.
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return randn(nRow, nCol)
	 * 
	 */
	public static Matrix randn(int nRow, int nCol) {
		Random generator = new Random();
		Matrix res = new DenseMatrix(nRow, nCol);
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				res.setEntry(i, j, generator.nextGaussian());
			}
		}
		return res;
	}
	
	/**
	 * Generate an n-by-n matrix containing pseudo-random values drawn 
	 * from the standard normal distribution.
	 * 
	 * @param n number of rows or columns
	 * 
	 * @return randn(n, n)
	 * 
	 */
	public static Matrix randn(int n) {
		return randn(n, n);
	}
	
	/**
	 * Signum function.
	 * <p>
     * For each element of X, SIGN(X) returns 1 if the element
     * is greater than zero, 0 if it equals zero and -1 if it is
     * less than zero.
     * </p>
     * 
	 * @param A a real matrix
	 * 
	 * @return sign(A)
	 * 
	 */
	public static Matrix sign(Matrix A) {
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		Matrix res = A.copy();
		double v = 0;
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				for (int j = 0; j < nCol; j++) {
					v = resRow[j];
					if (v > 0) {
						resRow[j] = 1;
					} else if (v < 0) {
						resRow[j] = -1;
					}
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			for (int k = 0; k < pr.length; k++) {
				v = pr[k];
				if (v > 0) {
					pr[k] = 1;
				} else if (v < 0) {
					pr[k] = -1;
				}
			}
		}

		return res;
	}
	
	/**
	 * Compute the squared l2 distance matrix between column vectors in matrix X
	 * and column vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j) entry being the squared l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X(:, i) - Y(:, j) ||_2^2
	 * 
	 */
	@Deprecated
	public static Matrix l2DistanceSquare0(Matrix X, Matrix Y) {

		int nX = X.getColumnDimension();
		int nY = Y.getColumnDimension();

		Matrix dist = null;
		
		Matrix part1 = columnVector2ColumnMatrix(sum(times(X, X), 1)).mtimes(ones(1, nY));
		
		Matrix part2 = ones(nX, 1).mtimes(rowVector2RowMatrix(sum(times(Y, Y), 1)));
		
		Matrix part3 = X.transpose().mtimes(Y).times(2);

		dist = part1.plus(part2).minus(part3);
		
		Matrix I = lt(dist, 0);
		logicalIndexingAssignment(dist, I, 0);

		return dist;

	}
	
	/**
	 * Compute the squared l2 distance matrix between row vectors in matrix X
	 * and row vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each row being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each row being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j) entry being the squared l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X(i, :) - Y(j, :) ||_2^2
	 * 
	 */
	public static Matrix l2DistanceSquare(Matrix X, Matrix Y) {

		int nX = X.getRowDimension();
		int nY = Y.getRowDimension();

		Matrix dist = null;
		
		double[] XX = sum(times(X, X), 2).getPr();
		double[] YY = sum(times(Y, Y), 2).getPr();
		
		dist = full(X.mtimes(Y.transpose()).times(-2));
		double[][] resData = ((DenseMatrix) dist).getData();
		double[] resRow = null;
		double s = 0;
		double v = 0;
		for (int i = 0; i < nX; i++) {
			resRow = resData[i];
			s = XX[i];
			for (int j = 0; j < nY; j++) {
				v = resRow[j] + s + YY[j];
				// resRow[j] += s + YY[j];
				if (v >= 0)
					resRow[j] = v;
				else
					resRow[j] = 0;
			}
		}

		/*Matrix I = lt(dist, 0);
		logicalIndexingAssignment(dist, I, 0);*/

		return dist;

	}
	
	/**
	 * Compute the squared l2 distance vector between a vector V
	 * and row vectors in matrix Y.
	 * 
	 * @param V
	 *        a feature vector
	 *        
	 * @param Y
	 *        Data matrix with each row being a feature vector
	 *        
	 * @return an n_y dimensional vector with its i-th entry being the squared l2
	 * distance between V and the i-th feature vector in Y, i.e., || V - Y(i, :) ||_2^2
	 * 
	 */
	public static Vector l2DistanceSquare(Vector V, Matrix Y) {

		// int nX = 1;
		int nY = Y.getRowDimension();

		Vector dist = null;
		
		double XX = sum(times(V, V));
		double[] YY = sum(times(Y, Y), 2).getPr();
		
		dist = full(Y.operate(V).times(-2));
		double[] pr = ((DenseVector) dist).getPr();
		double v = 0;
		for (int j = 0; j < nY; j++) {
			v = pr[j] + XX + YY[j];
			if (v >= 0)
				pr[j] = v;
			else
				pr[j] = 0;
			// pr[j] += XX + YY[j];
		}

		/*Matrix I = lt(dist, 0);
		logicalIndexingAssignment(dist, I, 0);*/

		return dist;
		
	}
	
	/**
	 * Compute the squared l2 distance matrix between column vectors in matrix X
	 * and column vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j) entry being the squared l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X(:, i) - Y(:, j) ||_2^2
	 * 
	 */
	public static Matrix l2DistanceSquareByColumns(Matrix X, Matrix Y) {

		int nX = X.getColumnDimension();
		int nY = Y.getColumnDimension();

		Matrix dist = null;
		
		double[] XX = sum(times(X, X)).getPr();
		double[] YY = sum(times(Y, Y)).getPr();
		
		dist = full(X.transpose().mtimes(Y).times(-2));
		double[][] resData = ((DenseMatrix) dist).getData();
		double[] resRow = null;
		double s = 0;
		double v = 0;
		for (int i = 0; i < nX; i++) {
			resRow = resData[i];
			s = XX[i];
			for (int j = 0; j < nY; j++) {
				v = resRow[j] + s + YY[j];
				// resRow[j] += XX[i] + YY[j];
				if (v >= 0)
					resRow[j] = v;
				else
					resRow[j] = 0;
			}
		}

		/*Matrix I = lt(dist, 0);
		logicalIndexingAssignment(dist, I, 0);*/
		
		return dist;

	}
	
	/**
	 * Compute the squared l2 distance matrix between two sets of vectors X and Y.
	 * 
	 * @param X
	 *        Data vectors
	 *        
	 * @param Y
	 *        Data vectors
	 *        
	 * @return an n_x X n_y matrix with its (i, j) entry being the squared l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X[i] - Y[j] ||_2^2
	 * 
	 */
	public static Matrix l2DistanceSquare(Vector[] X, Vector[] Y) {

		int nX = X.length;
		int nY = Y.length;

		Matrix dist = null;
		
		double[] XX = new double[nX];
		Vector V = null;
		for (int i = 0; i < nX; i++) {
			V = X[i];
			XX[i] = sum(V.times(V));
		}
		double[] YY = new double[nY];
		for (int i = 0; i < nY; i++) {
			V = Y[i];
			YY[i] = sum(V.times(V));
		}
		
		double[][] resData = allocate2DArray(nX, nY, 0);
		double[] resRow = null;
		double s = 0;
		double v = 0;
		for (int i = 0; i < nX; i++) {
			resRow = resData[i];
			V = X[i];
			s = XX[i];
			for (int j = 0; j < nY; j++) {
				v = s + YY[j] - 2 * innerProduct(V, Y[j]);;
				// resRow[j] = s + YY[j] - 2 * innerProduct(V, Y[j]);
				if (v >= 0)
					resRow[j] = v;
				else
					resRow[j] = 0;
			}
		}

		/*Matrix I = lt(dist, 0);
		logicalIndexingAssignment(dist, I, 0);*/
		dist = new DenseMatrix(resData);
		return dist;

	}
	
	/**
	 * Compute the l2 distance matrix between column vectors in matrix X
	 * and column vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each column being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j)th entry being the l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X(:, i) - Y(:, j) ||_2
	 * 
	 */
	public static Matrix l2DistanceByColumns(Matrix X, Matrix Y) {
		return sqrt(l2DistanceSquareByColumns(X, Y));
	}
	
	/**
	 * Compute the l2 distance matrix between row vectors in matrix X
	 * and row vectors in matrix Y.
	 * 
	 * @param X
	 *        Data matrix with each row being a feature vector.
	 *        
	 * @param Y
	 *        Data matrix with each row being a feature vector.
	 *        
	 * @return an n_x X n_y matrix with its (i, j)th entry being the l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X(i, :) - Y(j, :) ||_2
	 * 
	 */
	public static Matrix l2Distance(Matrix X, Matrix Y) {
		return sqrt(l2DistanceSquare(X, Y));
	}
	
	/**
	 * Compute the l2 distance vector between a vector V
	 * and row vectors in matrix Y.
	 * 
	 * @param V
	 *        a feature vector
	 *        
	 * @param Y
	 *        Data matrix with each row being a feature vector
	 *        
	 * @return an n_y dimensional vector with its i-th entry being the l2
	 * distance between V and the i-th feature vector in Y, i.e., || V - Y(i, :) ||_2
	 * 
	 */
	public static Vector l2Distance(Vector V, Matrix Y) {
		return sqrt(l2DistanceSquare(V, Y));
	}
	
	/**
	 * Compute the l2 distance matrix between two sets of vectors X and Y.
	 * 
	 * @param X
	 *        Data vectors
	 *        
	 * @param Y
	 *        Data vectors
	 *        
	 * @return an n_x X n_y matrix with its (i, j)th entry being the l2
	 * distance between i-th feature vector in X and j-th feature
	 * vector in Y, i.e., || X[i] - Y[j] ||_2
	 * 
	 */
	public static Matrix l2Distance(Vector[] X, Vector[] Y) {
		return sqrt(l2DistanceSquare(X, Y));
	}
	
	/**
	 * Calculate element by element division between a scalar and a vector.
	 * 
	 * @param v a real scalar
	 * 
	 * @param V a real vector
	 * 
	 * @return v ./ V
	 */
	public static Vector dotDivide(double v, Vector V) {
		int dim = V.getDim();
		DenseVector res = (DenseVector) full(V).copy();
		double pr[] = ((DenseVector) res).getPr();
		for (int k = 0; k < dim; k++) {
			pr[k] = v / pr[k];
		}
		return res;
	}
	
	/**
	 * Calculate square root for all elements of a vector V.
	 * 
	 * @param V a real vector
	 * 
	 * @return sqrt(V)
	 */
	public static Vector sqrt(Vector V) {
		int dim = V.getDim();
		Vector res = V.copy();
		if (res instanceof DenseVector) {
			double pr[] = ((DenseVector) res).getPr();
			for (int k = 0; k < dim; k++) {
				pr[k] = Math.sqrt(pr[k]);
			}
		} else if (res instanceof SparseVector) {
			double pr[] = ((SparseVector) res).getPr();
			int nnz = ((SparseVector) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] = Math.sqrt(pr[k]);
			}
		}
		return res;
	}
	
	/**
	 * Calculate square root for all elements of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return sqrt(A)
	 * 
	 */
	public static Matrix sqrt(Matrix A) {
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		Matrix res = A.copy();
		
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				for (int j = 0; j < nCol; j++) {
					resRow[j] = Math.sqrt(resRow[j]);
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			for (int k = 0; k < pr.length; k++) {
				pr[k] = Math.sqrt(pr[k]);
			}
		}

		return res;
	}
	
	// public static Matrix repmat(Vector)
	
	public static Matrix rowVector2RowMatrix(Vector V) {
		Matrix res = null;
		if (V instanceof DenseVector) {
			res = denseRowVectors2DenseMatrix(new Vector[] {V});
		} else if (V instanceof SparseVector) {
			res = sparseRowVectors2SparseMatrix(new Vector[] {V});
		}
		return res;
	}
	
	public static Matrix columnVector2ColumnMatrix(Vector V) {
		Matrix res = null;
		if (V instanceof DenseVector) {
			res = denseColumnVectors2DenseMatrix(new Vector[] {V});
		} else if (V instanceof SparseVector) {
			res = sparseColumnVectors2SparseMatrix(new Vector[] {V});
		}
		return res;
	}
	
	public static Matrix denseRowVectors2DenseMatrix(Vector[] Vs) {
		int M = Vs.length;
		// int N = Vs[0].getDim();
		double[][] resData = new double[M][];
		for (int i = 0; i < M; i++) {
			resData[i] = ((DenseVector) Vs[i]).getPr().clone();
		}
		return new DenseMatrix(resData);
	}
	
	public static Matrix denseColumnVectors2DenseMatrix(Vector[] Vs) {
		int N = Vs.length;
		int M = Vs[0].getDim();
		double[][] resData = new double[M][];
		for (int i = 0; i < M; i++) {
			resData[i] = new double[N];
		}
		for (int j = 0; j < N; j++) {
			double[] column = ((DenseVector) Vs[j]).getPr().clone();
			for (int i = 0; i < M; i++) {
				resData[i][j] = column[i];
			}
		}
		return new DenseMatrix(resData);
	}
	
	public static Vector[] denseMatrix2DenseRowVectors(Matrix A) {
		int M = A.getRowDimension();
		Vector[] res = new Vector[M];
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			for (int i = 0; i < M; i++) {
				res[i] = new DenseVector(AData[i]);
			}
		} else {
			System.err.println("The input matrix should be a dense matrix.");
			exit(1);
		}
		return res;
	}
	
	public static Vector[] denseMatrix2DenseColumnVectors(Matrix A) {
		int N = A.getColumnDimension();
		int M = A.getRowDimension();
		Vector[] res = new Vector[N];
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			for (int j = 0; j < N; j++) {
				double[] column = new double[M];
				for (int i = 0; i < M; i++) {
					column[i] = AData[i][j];
				}
				res[j] = new DenseVector(column);
			}
		} else {
			System.err.println("The input matrix should be a dense matrix.");
			exit(1);
		}
		return res;
	}
	
	/**
	 * Generate nRow by nCol all zero matrix.
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return zeros(nRow, nCol)
	 * 
	 */
	public static Matrix zeros(int nRow, int nCol) {
		if (nRow == 0 || nCol == 0) {
			return null;
		}
		return new DenseMatrix(nRow, nCol, 0);
	}
	
	/**
	 * Generate an all zero matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return an all zero matrix with its shape specified by size 
	 * 
	 */
	public static Matrix zeros(int[] size) {
		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
		}
		return zeros(size[0], size[1]);
	}

	/**
	 * Generate an n by n all zero matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return ones(n)
	 * 
	 */
	public static Matrix zeros(int n) {
		return zeros(n, n);
	}
	
	/**
	 * Generate an all one matrix with nRow rows and nCol columns.
	 * 
	 * @param nRow number of rows
	 * 
	 * @param nCol number of columns
	 * 
	 * @return ones(nRow, nCol)
	 * 
	 */
	public static Matrix ones(int nRow, int nCol) {
		if (nRow == 0 || nCol == 0) {
			return null;
		}
		return new DenseMatrix(nRow, nCol, 1);
	}
	
	/**
	 * Generate an all one matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return an all one matrix with its shape specified by size 
	 * 
	 */
	public static Matrix ones(int[] size) {
		if (size.length != 2) {
			System.err.println("Input vector should have two elements!");
		}
		return ones(size[0], size[1]);
	}

	/**
	 * Generate an n by n all one matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return ones(n)
	 * 
	 */
	public static Matrix ones(int n) {
		return ones(n, n);
	}
	
	/**
	 * Compute the determinant of a real square matrix.
	 * 
	 * @param A a real square matrix
	 * 
	 * @return det(A)
	 */
	public static double det(Matrix A) {
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (M != N) {
			System.err.println("Input should be a square matrix.");
			exit(1);
		}
		return new LUDecomposition(A).det();
	}
	
	/**
	 * Compute the inverse of a real square matrix.
	 * 
	 * @param A a real square matrix
	 * 
	 * @return inv(A)
	 */
	public static Matrix inv(Matrix A) {
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (M != N) {
			System.err.println("Input should be a square matrix.");
			exit(1);
		}
		LUDecomposition LUDecomp = new LUDecomposition(A);
		if (LUDecomp.det() == 0) {
			System.err.println("The input matrix is not invertible.");
			exit(1);
		}
		return LUDecomp.inverse();
	}
	
	/**
	 * Sort elements of a vector V in place with an increasing order.
	 * 
	 * @param V a real vector, it will be sorted in place.
	 * 
	 * @return sorted indices represented by a 1D {@code double} array
	 */
	public static double[] sort(Vector V) {
		return sort(V, "ascend");
	}
	
	/**
	 * Sort elements of a vector V in place with a specified order.
	 * 
	 * @param V a real vector, it will be sorted in place.
	 * 
	 * @param order a {@code String} variable either "ascend" or "descend"
	 * 
	 * @return sorted indices represented by a 1D {@code double} array
	 */
	public static double[] sort(Vector V, String order) {
		double[] indices = null;
		int dim = V.getDim();
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			indices = new double[dim];
			for (int k = 0; k < dim; k++) {
				indices[k] = k;
			}
			quickSort(pr, indices, 0, dim - 1, order);
		} else if (V instanceof SparseVector) {
			double[] pr = ((SparseVector) V).getPr();
			int[] ir = ((SparseVector) V).getIr();
			int nnz = ((SparseVector) V).getNNZ();
			indices = new double[dim];
			
			int insertionPos = nnz;
			if (order.equalsIgnoreCase("ascend")) {
				for (int k = 0; k < nnz; k++) {
					if (pr[k] >= 0) {
						insertionPos--;
					}
				}
			} else if (order.equalsIgnoreCase("descend")) {
				for (int k = 0; k < nnz; k++) {
					if (pr[k] <= 0) {
						insertionPos--;
					}
				}
			}
			
			int lastIdx = -1;
			int currentIdx = 0;
			int cnt = insertionPos;
			for (int k = 0; k < nnz; k++) {
				currentIdx = ir[k];
				for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
					indices[cnt++] = idx;
				}
				lastIdx = currentIdx;
			}
			for (int idx = lastIdx + 1; idx < dim; idx++) {
				indices[cnt++] = idx;
			}
			
			quickSort(pr, ir, 0, nnz - 1, order);
			
			for (int k = 0; k < insertionPos; k++) {
				indices[k] = ir[k];
			}
			for (int k = insertionPos; k < nnz; k++) {
				indices[k + dim - nnz] = ir[k];
			}
			
			for (int k = 0; k < nnz; k++) {
				if (k < insertionPos) {
					ir[k] = k;
				} else {
					ir[k] = k + dim - nnz;
				}
			}
		}
		return indices;
	}
	
	/**
	 * Sort elements of a vector V in place with a specified order.
	 * 
	 * @param V a real vector, it will be sorted in place.
	 * 
	 * @param order a {@code String} variable either "ascend" or "descend"
	 * 
	 * @return sorted indices represented by a 1D {@code double} array
	 */
	@Deprecated
	public static double[] sort1(Vector V, String order) {
		double[] indices = null;
		int dim = V.getDim();
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			indices = new double[dim];
			for (int k = 0; k < dim; k++) {
				indices[k] = k;
			}
			quickSort(pr, indices, 0, dim - 1, order);
		} else if (V instanceof SparseVector) {
			double[] pr = ((SparseVector) V).getPr();
			int[] ir = ((SparseVector) V).getIr();
			int nnz = ((SparseVector) V).getNNZ();
			int[] ir_ori = ir.clone();
			quickSort(pr, ir, 0, nnz - 1, order);
			int insertionPos = nnz;
			if (order.equalsIgnoreCase("ascend")) {
				for (int k = 0; k < nnz; k++) {
					if (pr[k] >= 0) {
						insertionPos = k;
						break;
					}
				}
			} else if (order.equalsIgnoreCase("descend")) {
				for (int k = 0; k < nnz; k++) {
					if (pr[k] <= 0) {
						insertionPos = k;
						break;
					}
				}
			}
			indices = new double[dim];
			for (int k = 0; k < insertionPos; k++) {
				indices[k] = ir[k];
			}
			int lastIdx = -1;
			int currentIdx = 0;
			int cnt = insertionPos;
			for (int k = 0; k < nnz; k++) {
				currentIdx = ir_ori[k];
				for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
					indices[cnt++] = idx;
				}
				lastIdx = currentIdx;
			}
			for (int idx = lastIdx + 1; idx < dim; idx++) {
				indices[cnt++] = idx;
			}
			for (int k = insertionPos; k < nnz; k++) {
				indices[k + dim - nnz] = ir[k];
			}
			for (int k = 0; k < nnz; k++) {
				if (k < insertionPos) {
					ir[k] = k;
				} else {
					ir[k] = k + dim - nnz;
				}
			}
		}
		return indices;
	}
	
	/**
	 * Sort elements of a vector V in place with a specified order.
	 * 
	 * @param V a real vector, it will be sorted in place.
	 * 
	 * @param order a {@code String} variable either "ascend" or "descend"
	 * 
	 * @return sorted indices represented by a 1D {@code double} array
	 */
	@Deprecated
	public static double[] sort0(Vector V, String order) {
		double[] indices = null;
		int dim = V.getDim();
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			indices = new double[dim];
			for (int k = 0; k < dim; k++) {
				indices[k] = k;
			}
			quickSort(pr, indices, 0, dim - 1, order);
		} else if (V instanceof SparseVector) {
			double[] pr = ((SparseVector) V).getPr();
			int[] ir = ((SparseVector) V).getIr();
			int nnz = ((SparseVector) V).getNNZ();
			int[] ir_ori = ir.clone();
			if (order.equalsIgnoreCase("ascend")) {
				quickSort(pr, ir, 0, nnz - 1, order);
				int numNegatives = nnz;
				for (int k = 0; k < nnz; k++) {
					if (pr[k] >= 0) {
						numNegatives = k;
						break;
					}
				}
				indices = new double[dim];
				for (int k = 0; k < numNegatives; k++) {
					indices[k] = ir[k];
				}
				int lastIdx = -1;
				int currentIdx = 0;
				int cnt = numNegatives;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir_ori[k];
					for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
						indices[cnt++] = idx;
					}
					lastIdx = currentIdx;
				}
				for (int idx = lastIdx + 1; idx < dim; idx++) {
					indices[cnt++] = idx;
				}
				/*for (int k = numNegatives; k < numNegatives + dim - nnz; k++) {
			}*/
				for (int k = numNegatives; k < nnz; k++) {
					indices[k + dim - nnz] = ir[k];
				}

				for (int k = 0; k < nnz; k++) {
					if (k < numNegatives) {
						ir[k] = k;
					} else {
						ir[k] = k + dim - nnz;
					}
				}
			} else if (order.equalsIgnoreCase("descend")) {
				quickSort(pr, ir, 0, nnz - 1, order);
				int numPositives = nnz;
				for (int k = 0; k < nnz; k++) {
					if (pr[k] <= 0) {
						numPositives = k;
						break;
					}
				}
				indices = new double[dim];
				for (int k = 0; k < numPositives; k++) {
					indices[k] = ir[k];
				}
				int lastIdx = -1;
				int currentIdx = 0;
				int cnt = numPositives;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir_ori[k];
					for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
						indices[cnt++] = idx;
					}
					lastIdx = currentIdx;
				}
				for (int idx = lastIdx + 1; idx < dim; idx++) {
					indices[cnt++] = idx;
				}
				/*for (int k = numNegatives; k < numNegatives + dim - nnz; k++) {
			}*/
				for (int k = numPositives; k < nnz; k++) {
					indices[k + dim - nnz] = ir[k];
				}

				for (int k = 0; k < nnz; k++) {
					if (k < numPositives) {
						ir[k] = k;
					} else {
						ir[k] = k + dim - nnz;
					}
				}
			}
		}
		return indices;
	}
	
	/**
	 * Sort elements of a matrix A on a direction in a specified order.
	 * A will not be modified.
	 * 
	 * @param A a matrix to be sorted
	 * 
	 * @param dim sorting direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @param order sorting order, either "ascend" or "descend"
	 * 
	 * @return a {@code Matrix} array: 
	 *         res[0] is the sorted matrix
	 *         res[1] is the sorted indices
	 *         
	 */
	public static Matrix[] sort(Matrix A, int dim, String order) {
		
		if (A == null) {
			return null;
		}
		
		if (dim != 1 && dim != 2) {
			System.err.println("Dimension should be either 1 or 2.");
			exit(1);
		}
		
		if (!order.equalsIgnoreCase("ascend") && !order.equalsIgnoreCase("descend")) {
			System.err.println("Order should be either \"ascend\" or \"descend\".");
			exit(1);
		}
		
		Matrix[] res = new Matrix[2];
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		Matrix sortedValues = null;
		Matrix sortedIndices = null;
		double[][] sortedIndexData = null;
		if (A instanceof DenseMatrix) {
			sortedValues = A.copy();
			sortedIndices = null;
			double[][] data = ((DenseMatrix) sortedValues).getData();
			if (dim == 2) {
				sortedIndexData = new double[M][];
				double[] values = null;
				double[] indices = null;
				for (int i = 0; i < M; i++) {
					values = data[i];
					indices = new double[N];
					for (int j = 0; j < N; j++) {
						indices[j] = j;
					}
					quickSort(values, indices, 0, N - 1, order);
					sortedIndexData[i] = indices;
				}
				sortedIndices = new DenseMatrix(sortedIndexData);
			} else if (dim == 1) {
				Matrix[] res2 = sort(A.transpose(), 2, order);
				sortedValues = res2[0].transpose();
				sortedIndices = res2[1].transpose();
			}
		} else if (A instanceof SparseMatrix) {
			if (dim == 2) {
				Vector[] rowVectors = sparseMatrix2SparseRowVectors(A);
				sortedIndexData = new double[M][];
				for (int i = 0; i < M; i++) {
					sortedIndexData[i] = sort(rowVectors[i], order);
				}
				sortedIndices = new DenseMatrix(sortedIndexData);
				sortedValues = sparseRowVectors2SparseMatrix(rowVectors);
			} else if (dim == 1) {
				Matrix[] res2 = sort(A.transpose(), 2, order);
				sortedValues = res2[0].transpose();
				sortedIndices = res2[1].transpose();
			}
		}
		
		res[0] = sortedValues;
		res[1] = sortedIndices;
		return res;
		
	}
	
	/**
	 * Sort elements of a matrix A on a direction in an increasing order.
	 * 
	 * @param A a matrix to be sorted
	 * 
	 * @param dim sorting direction, 1 for column-wise, 2 for row-wise
	 * 
	 * @return a {@code Matrix} array: 
	 *         res[0] is the sorted matrix
	 *         res[1] is the sorted indices
	 *         
	 */
	public static Matrix[] sort(Matrix A, int dim) {
		return sort(A, dim, "ascend");
	}
	
	/**
	 * Sort elements of a matrix A by columns in a specified order.
	 * A will not be modified.
	 * 
	 * @param A a matrix to be sorted
	 * 
	 * @param order sorting order, either "ascend" or "descend"
	 * 
	 * @return a {@code Matrix} array: 
	 *         res[0] is the sorted matrix
	 *         res[1] is the sorted indices
	 *         
	 */
	public static Matrix[] sort(Matrix A, String order) {
		return sort(A, 1, order);
	}
	
	/**
	 * Sort elements of a matrix A by columns in an increasing order.
	 * A will not be modified.
	 * 
	 * @param A a matrix to be sorted
	 * 
	 * @return a {@code Matrix} array: 
	 *         res[0] is the sorted matrix
	 *         res[1] is the sorted indices
	 *         
	 */
	public static Matrix[] sort(Matrix A) {
		return sort(A, "ascend");
	}
	
	/**
	 * Get a two dimensional integer array for size of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return size(A)
	 * 
	 */
	public static int[] size(Matrix A) {
		return new int[] { A.getRowDimension(), A.getColumnDimension()};
	}

	/**
	 * Get the dimensionality on dim-th dimension for a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @param dim dimension order
	 * 
	 * @return size(A, dim)
	 * 
	 */
	public static int size(Matrix A, int dim) {
		if (dim == 1) {
			return A.getRowDimension();
		} else if (dim == 2) {
			return A.getColumnDimension();
		} else {
			System.err.println("Dim error!");
			return 0;
		}
	}
	
	/**
	 * Compute maximum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param A a real matrix
	 * 
	 * @param v a real number
	 * 
	 * @return max(A, v)
	 * 
	 */
	public static Matrix max(Matrix A, double v) {
		
		if (A == null)
			return null;
		
		Matrix res = null;
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (A instanceof DenseMatrix) {
			res = A.copy();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (resRow[j] < v) {
						resRow[j] = v;
					}
				}
			}
		} else if (A instanceof SparseMatrix) {
			if (v != 0) {
				res = new DenseMatrix(M, N, v);
				double[][] resData = ((DenseMatrix) res).getData();
				double[] resRow = null;
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						if (v < 0)
							ArrayOperator.assignVector(resRow, 0);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						if (v < 0)
							for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
								resRow[c] = 0;
							}
						resRow[currentColumnIdx] = Math.max(pr[valCSRIndices[k]], v);
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						if (v < 0)
							resRow[c] = 0;
					}
				}
			} else { // v == 0
				return max(A, new SparseMatrix(M, N));
			}
		}
		return res;
	}
	
	/**
	 * Compute maximum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param v a real number
	 * 
	 * @param A a real matrix
	 * 
	 * @return max(v, A)
	 * 
	 */
	public static Matrix max(double v, Matrix A) {
		return max(A, v);
	}
	
	/**
	 * Compute maximum between two real matrices A and B.
	 * 
	 * @param A a real matrix
	 * 
	 * @param B a real matrix
	 * 
	 * @return max(A, B);
	 */
	public static Matrix max(Matrix A, Matrix B) {
		Matrix res = null;
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		double v = 0;
		if (A instanceof DenseMatrix) {
			res = A.copy();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (B instanceof DenseMatrix) {
				double[][] BData = ((DenseMatrix) B).getData();
				double[] BRow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					BRow = BData[i];
					resRow = resData[i];
					for (int j = 0; j < N; j++) {
						v = BRow[j];
						if (resRow[j] < v)
							resRow[j] = v;
					}
				}
			} else if (B instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) B).getIc();
				int[] jr = ((SparseMatrix) B).getJr();
				int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
				double[] pr = ((SparseMatrix) B).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						for (int j = 0; j < N; j++) {
							if (resRow[j] < 0)
								resRow[j] = 0;
						}
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							if (resRow[c] < 0)
								resRow[c] = 0;
						}
						resRow[currentColumnIdx] = Math.max(pr[valCSRIndices[k]], resRow[currentColumnIdx]);
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						if (resRow[c] < 0)
							resRow[c] = 0;
					}
				}
			}
		} else if (A instanceof SparseMatrix) {
			if (B instanceof DenseMatrix) {
				return max(B, A);
			} else if (B instanceof SparseMatrix) {
				
				res = new SparseMatrix(M, N);
				int[] ir1 = null;
				int[] jc1 = null;
				double[] pr1 = null;
				ir1 = ((SparseMatrix) A).getIr();
				jc1 = ((SparseMatrix) A).getJc();
				pr1 = ((SparseMatrix) A).getPr();
				int[] ir2 = null;
				int[] jc2 = null;
				double[] pr2 = null;
				ir2 = ((SparseMatrix) B).getIr();
				jc2 = ((SparseMatrix) B).getJc();
				pr2 = ((SparseMatrix) B).getPr();
				
				int k1 = 0;
				int k2 = 0;
				int r1 = -1;
				int r2 = -1;
				int i = -1;
				// double v = 0;
				
				for (int j = 0; j < N; j++) {
					k1 = jc1[j];
					k2 = jc2[j];
					
					// Both A and B's j-th columns are empty.
					if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
						continue;
					
					while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {
						
						if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
							i = ir1[k1];
							v = pr1[k1];
							if (v < 0)
								v = 0;
							k1++;
						} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
							i = ir2[k2];
							v = pr2[k2];
							if (v < 0)
								v = 0;
							k2++;
						} else { // Both A and B's j-th columns have not been fully processed.
							r1 = ir1[k1];
							r2 = ir2[k2];				
							if (r1 < r2) {
								i = r1;
								v = pr1[k1];
								if (v < 0)
									v = 0;
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = Math.max(pr1[k1], pr2[k2]);
								k1++;
								k2++;
							} else {
								i = r2;
								v = pr2[k2];
								if (v < 0)
									v = 0;
								k2++;
							}
						}
						if (v != 0)
							res.setEntry(i, j, v);
					}
				}
			}
		}
			
		return res;
		
	}
	
	/**
	 * Compute minimum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param A a real matrix
	 * 
	 * @param v a real number
	 * 
	 * @return min(A, v)
	 * 
	 */
	public static Matrix min(Matrix A, double v) {
		
		if (A == null)
			return null;
		
		Matrix res = null;
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (A instanceof DenseMatrix) {
			res = A.copy();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (resRow[j] > v) {
						resRow[j] = v;
					}
				}
			}
		} else if (A instanceof SparseMatrix) {
			if (v != 0) {
				res = new DenseMatrix(M, N, v);
				double[][] resData = ((DenseMatrix) res).getData();
				double[] resRow = null;
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						if (v > 0)
							ArrayOperator.assignVector(resRow, 0);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						if (v > 0)
							for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
								resRow[c] = 0;
							}
						resRow[currentColumnIdx] = Math.min(pr[valCSRIndices[k]], v);
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						if (v > 0)
							resRow[c] = 0;
					}
				}
			} else { // v == 0
				return min(A, new SparseMatrix(M, N));
			}
		}
		return res;
	}
	
	/**
	 * Compute minimum between elements of A and a real number and return
	 * as a matrix with the same shape of A.
	 * 
	 * @param v a real number
	 * 
	 * @param A a real matrix
	 * 
	 * @return min(v, A)
	 * 
	 */
	public static Matrix min(double v, Matrix A) {
		return min(A, v);
	}
	
	/**
	 * Compute minimum between two real matrices A and B.
	 * 
	 * @param A a real matrix
	 * 
	 * @param B a real matrix
	 * 
	 * @return max(A, B);
	 */
	public static Matrix min(Matrix A, Matrix B) {
		Matrix res = null;
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		double v = 0;
		if (A instanceof DenseMatrix) {
			res = A.copy();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (B instanceof DenseMatrix) {
				double[][] BData = ((DenseMatrix) B).getData();
				double[] BRow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					BRow = BData[i];
					resRow = resData[i];
					for (int j = 0; j < N; j++) {
						v = BRow[j];
						if (resRow[j] > v)
							resRow[j] = v;
					}
				}
			} else if (B instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) B).getIc();
				int[] jr = ((SparseMatrix) B).getJr();
				int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
				double[] pr = ((SparseMatrix) B).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						for (int j = 0; j < N; j++) {
							if (resRow[j] > 0)
								resRow[j] = 0;
						}
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							if (resRow[c] > 0)
								resRow[c] = 0;
						}
						resRow[currentColumnIdx] = Math.min(pr[valCSRIndices[k]], resRow[currentColumnIdx]);
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						if (resRow[c] > 0)
							resRow[c] = 0;
					}
				}
			}
		} else if (A instanceof SparseMatrix) {
			if (B instanceof DenseMatrix) {
				return min(B, A);
			} else if (B instanceof SparseMatrix) {
				
				res = new SparseMatrix(M, N);
				int[] ir1 = null;
				int[] jc1 = null;
				double[] pr1 = null;
				ir1 = ((SparseMatrix) A).getIr();
				jc1 = ((SparseMatrix) A).getJc();
				pr1 = ((SparseMatrix) A).getPr();
				int[] ir2 = null;
				int[] jc2 = null;
				double[] pr2 = null;
				ir2 = ((SparseMatrix) B).getIr();
				jc2 = ((SparseMatrix) B).getJc();
				pr2 = ((SparseMatrix) B).getPr();
				
				int k1 = 0;
				int k2 = 0;
				int r1 = -1;
				int r2 = -1;
				int i = -1;
				// double v = 0;
				
				for (int j = 0; j < N; j++) {
					k1 = jc1[j];
					k2 = jc2[j];
					
					// Both A and B's j-th columns are empty.
					if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
						continue;
					
					while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {
						
						if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
							i = ir1[k1];
							v = pr1[k1];
							if (v > 0)
								v = 0;
							k1++;
						} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
							i = ir2[k2];
							v = pr2[k2];
							if (v > 0)
								v = 0;
							k2++;
						} else { // Both A and B's j-th columns have not been fully processed.
							r1 = ir1[k1];
							r2 = ir2[k2];				
							if (r1 < r2) {
								i = r1;
								v = pr1[k1];
								if (v > 0)
									v = 0;
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = Math.min(pr1[k1], pr2[k2]);
								k1++;
								k2++;
							} else {
								i = r2;
								v = pr2[k2];
								if (v > 0)
									v = 0;
								k2++;
							}
						}
						if (v != 0)
							res.setEntry(i, j, v);
					}
				}
			}
		}
			
		return res;
		
	}
	
	/**
	 * Compute maximum between elements of V and a real number 
	 * and return as a vector with the same shape of V.
	 * 
	 * @param V a real vector
	 * 
	 * @param v a real number
	 * 
	 * @return max(V, v)
	 * 
	 */
	public static Vector max(Vector V, double v) {
		Vector res = null;
		int dim = V.getDim();
		if (V instanceof DenseVector) {
			res = V.copy();
			double[] pr = ((DenseVector) res).getPr();
			for (int k = 0; k < dim; k++) {
				if (pr[k] < v)
					pr[k] = v;
			}
		} else if (V instanceof SparseVector) {
			if (v != 0) {
				res = new DenseVector(dim, v);
				double[] prRes = ((DenseVector) res).getPr();
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = -1;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
						if (v < 0)
							prRes[idx] = 0;
					}
					if (v < pr[k])
						prRes[currentIdx] = pr[k];
					lastIdx = currentIdx;
				}
				for (int idx = lastIdx + 1; idx < dim; idx++) {
					if (v < 0)
						prRes[idx] = 0;
				}
			} else { // v == 0
				res = V.copy();
				double[] pr = ((SparseVector) res).getPr();
				int nnz = ((SparseVector) res).getNNZ();
				for (int k = 0; k < nnz; k++) {
					if (pr[k] < 0)
						pr[k] = 0;
				}
				((SparseVector) res).clean();
			}
		}
		return res;
	}
	
	/**
	 * Compute maximum between elements of V and a real number 
	 * and return as a vector with the same shape of V.
	 * 
	 * @param v a real number
	 * 
	 * @param V a real vector
	 * 
	 * @return max(v, V)
	 * 
	 */
	public static Vector max(double v, Vector V) {
		return max(V, v);
	}
	
	/**
	 * Compute maximum between two vectors U and V.
	 * 
	 * @param U a real vector
	 * 
	 * @param V a real vector
	 * 
	 * @return max(U, V)
	 * 
	 */
	public static Vector max(Vector U, Vector V) {
		Vector res = null;
		int dim = U.getDim();
		double v = 0;
		if (U instanceof DenseVector) {
			res = U.copy();
			double[] prRes = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				// double[] prU = ((DenseVector) U).getPr();
				double[] prV = ((DenseVector) V).getPr();
				for (int k = 0; k < dim; k++) {
					v = prV[k];
					if (prRes[k] < v)
						prRes[k] = v;
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = -1;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
						if (prRes[idx] < 0)
							prRes[idx] = 0;
					}
					v = pr[k];
					if (prRes[currentIdx] < v)
						prRes[currentIdx] = v;
					lastIdx = currentIdx;
				}
				for (int idx = lastIdx + 1; idx < dim; idx++) {
					if (prRes[idx] < 0)
						prRes[idx] = 0;
				}
			}
		} else if (U instanceof SparseVector) {
			if (V instanceof DenseVector) {
				return max(V, U);
			} else if (V instanceof SparseVector) {
				res = new SparseVector(dim);
				int[] ir1 = ((SparseVector) V).getIr();
				double[] pr1 = ((SparseVector) V).getPr();
				int nnz1 = ((SparseVector) V).getNNZ();
				int[] ir2 = ((SparseVector) U).getIr();
				double[] pr2 = ((SparseVector) U).getPr();
				int nnz2 = ((SparseVector) U).getNNZ();
				if (!(nnz1 == 0 && nnz2 == 0)) {
					int k1 = 0;
					int k2 = 0;
					int r1 = 0;
					int r2 = 0;
					// double v = 0;
					int i = -1;
					while (k1 < nnz1 || k2 < nnz2) {
						if (k2 == nnz2) { // V has been processed.
							i = ir1[k1];
							v = pr1[k1];
							if (v < 0)
								v = 0;
							k1++;
						} else if (k1 == nnz1) { // this has been processed.
							i = ir2[k2];
							v = pr2[k2];
							if (v < 0)
								v = 0;
							k2++;
						} else { // Both this and V have not been fully processed.
							r1 = ir1[k1];
							r2 = ir2[k2];
							if (r1 < r2) {
								i = r1;
								v = pr1[k1];
								if (v < 0)
									v = 0;
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = Math.max(pr1[k1], pr2[k2]);
								k1++;
								k2++;
							} else {
								i = r2;
								v = pr2[k2];
								if (v < 0)
									v = 0;
								k2++;
							}
						}
						if (v != 0) {
							res.set(i, v);
						}
					}
				}
			}
		}
		return res;
	}
	
	/**
	 * Compute minimum between elements of V and a real number 
	 * and return as a vector with the same shape of V.
	 * 
	 * @param V a real vector
	 * 
	 * @param v a real number
	 * 
	 * @return min(V, v)
	 * 
	 */
	public static Vector min(Vector V, double v) {
		Vector res = null;
		int dim = V.getDim();
		if (V instanceof DenseVector) {
			res = V.copy();
			double[] pr = ((DenseVector) res).getPr();
			for (int k = 0; k < dim; k++) {
				if (pr[k] > v)
					pr[k] = v;
			}
		} else if (V instanceof SparseVector) {
			if (v != 0) {
				res = new DenseVector(dim, v);
				double[] prRes = ((DenseVector) res).getPr();
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = -1;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
						if (v > 0)
							prRes[idx] = 0;
					}
					if (v > pr[k])
						prRes[currentIdx] = pr[k];
					lastIdx = currentIdx;
				}
				for (int idx = lastIdx + 1; idx < dim; idx++) {
					if (v > 0)
						prRes[idx] = 0;
				}
			} else { // v == 0
				res = V.copy();
				double[] pr = ((SparseVector) res).getPr();
				int nnz = ((SparseVector) res).getNNZ();
				for (int k = 0; k < nnz; k++) {
					if (pr[k] > 0)
						pr[k] = 0;
				}
				((SparseVector) res).clean();
			}
		}
		return res;
	}
	
	/**
	 * Compute minimum between elements of V and a real number 
	 * and return as a vector with the same shape of V.
	 * 
	 * @param v a real number
	 * 
	 * @param V a real vector
	 * 
	 * @return min(v, V)
	 * 
	 */
	public static Vector min(double v, Vector V) {
		return min(V, v);
	}
	
	/**
	 * Compute minimum between two vectors U and V.
	 * 
	 * @param U a real vector
	 * 
	 * @param V a real vector
	 * 
	 * @return min(U, V)
	 * 
	 */
	public static Vector min(Vector U, Vector V) {
		Vector res = null;
		int dim = U.getDim();
		double v = 0;
		if (U instanceof DenseVector) {
			res = U.copy();
			double[] prRes = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				// double[] prU = ((DenseVector) U).getPr();
				double[] prV = ((DenseVector) V).getPr();
				for (int k = 0; k < dim; k++) {
					v = prV[k];
					if (prRes[k] > v)
						prRes[k] = v;
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = -1;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int idx = lastIdx + 1; idx < currentIdx; idx++) {
						if (prRes[idx] > 0)
							prRes[idx] = 0;
					}
					v = pr[k];
					if (prRes[currentIdx] > v)
						prRes[currentIdx] = v;
					lastIdx = currentIdx;
				}
				for (int idx = lastIdx + 1; idx < dim; idx++) {
					if (prRes[idx] > 0)
						prRes[idx] = 0;
				}
			}
		} else if (U instanceof SparseVector) {
			if (V instanceof DenseVector) {
				return max(V, U);
			} else if (V instanceof SparseVector) {
				res = new SparseVector(dim);
				int[] ir1 = ((SparseVector) V).getIr();
				double[] pr1 = ((SparseVector) V).getPr();
				int nnz1 = ((SparseVector) V).getNNZ();
				int[] ir2 = ((SparseVector) U).getIr();
				double[] pr2 = ((SparseVector) U).getPr();
				int nnz2 = ((SparseVector) U).getNNZ();
				if (!(nnz1 == 0 && nnz2 == 0)) {
					int k1 = 0;
					int k2 = 0;
					int r1 = 0;
					int r2 = 0;
					// double v = 0;
					int i = -1;
					while (k1 < nnz1 || k2 < nnz2) {
						if (k2 == nnz2) { // V has been processed.
							i = ir1[k1];
							v = pr1[k1];
							if (v > 0)
								v = 0;
							k1++;
						} else if (k1 == nnz1) { // this has been processed.
							i = ir2[k2];
							v = pr2[k2];
							if (v > 0)
								v = 0;
							k2++;
						} else { // Both this and V have not been fully processed.
							r1 = ir1[k1];
							r2 = ir2[k2];
							if (r1 < r2) {
								i = r1;
								v = pr1[k1];
								if (v > 0)
									v = 0;
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = Math.min(pr1[k1], pr2[k2]);
								k1++;
								k2++;
							} else {
								i = r2;
								v = pr2[k2];
								if (v > 0)
									v = 0;
								k2++;
							}
						}
						if (v != 0) {
							res.set(i, v);
						}
					}
				}
			}
		}
		return res;
	}
	
	/**
	 * Logical indexing A by B for the syntax A(B). A logical matrix 
	 * provides a different type of array indexing in MATLAB. While
	 * most indices are numeric, indicating a certain row or column
	 * number, logical indices are positional. That is, it is the
	 * position of each 1 in the logical matrix that determines which
	 * array element is being referred to.
	 * 
	 * @param A a real matrix
	 * 
	 * @param B a logical matrix with elements being either 1 or 0
	 * 
	 * @return A(B)
	 * 
	 */
	public static Matrix logicalIndexing(Matrix A, Matrix B) {
		
		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices should have same size!");
			System.exit(1);
		}
		
		ArrayList<Double> vals = new ArrayList<Double>();
		
		double b;
		for (int j = 0; j < nA; j++) {
			for (int i = 0; i < dA; i++) {
				b = B.getEntry(i, j);
				if (b == 1)
					vals.add(A.getEntry(i, j));
				else if (b != 0)
					System.err.println("Elements of the logical matrix should be either 1 or 0!");
			}
		}
		
		Double[] Data = new Double[vals.size()];
		vals.toArray(Data);
		
		double[] data = new double[vals.size()];
		for (int i = 0; i < vals.size(); i++) {
			data[i] = Data[i];
		}		
		
		if (data.length != 0)
			return new DenseMatrix(data, 1);
		else
			return null;
		
	}
	
	/**
	 * Linear indexing V by an index array.
	 * 
	 * @param V an {@code int} array
	 * 
	 * @param indices an {@code int} array of selected indices
	 * 
	 * @return V(indices)
	 * 
	 */
	public static int[] linearIndexing(int[] V, int[] indices) {
		
		if (indices == null || indices.length == 0) {
			return null;
		}
		
		int[] res = new int[indices.length];
		for (int i = 0; i < indices.length; i++) {
			res[i] = V[indices[i]];
		}
		
		return res;
		
	}
	
	/**
	 * Linear indexing V by an index array.
	 * 
	 * @param V an {@code double} array
	 * 
	 * @param indices an {@code int} array of selected indices
	 * 
	 * @return V(indices)
	 * 
	 */
	public static double[] linearIndexing(double[] V, int[] indices) {
		
		if (indices == null || indices.length == 0) {
			return null;
		}
		
		double[] res = new double[indices.length];
		for (int i = 0; i < indices.length; i++) {
			res[i] = V[indices[i]];
		}
		
		return res;
		
	}
	
	/**
	 * Linear indexing A by an index array.
	 * 
	 * @param A a real matrix
	 * 
	 * @param indices an {@code int} array of selected indices
	 * 
	 * @return A(indices)
	 * 
	 */
	public static Matrix linearIndexing(Matrix A, int[] indices) {
		
		if (indices == null || indices.length == 0) {
			return null;
		}
		
		Matrix res = null;
		if (A instanceof DenseMatrix) {
			res = new DenseMatrix(indices.length, 1);
		} else {
			res = new SparseMatrix(indices.length, 1);
		}
		
		int nRow = A.getRowDimension();
		// int nCol = A.getColumnDimension();
		int r = -1;
		int c = -1;
		int index = -1;
		for (int i = 0; i < indices.length; i++) {
			index = indices[i];
			r = index % nRow;
			c = index / nRow;
			res.setEntry(i, 0, A.getEntry(r, c));
		}
		
		return res;
		
	}
	
	
	/**
	 * Matrix assignment by linear indexing for the syntax A(B) = V.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param idx a linear index vector
	 *          
	 * @param V a column matrix to assign A(idx)
	 * 
	 */
	public static void linearIndexingAssignment(Matrix A, int[] idx, Matrix V) {
		
		if (V == null)
			return;
		
		int nV = V.getColumnDimension();
		int dV = V.getRowDimension();
		
		if (nV != 1)
			System.err.println("Assignment matrix should be a column matrix!");
		
		if (idx.length != dV)
			System.err.println("Assignment with different number of elements!");
		
		int nRow = A.getRowDimension();
		// int nCol = A.getColumnDimension();
		int r = -1;
		int c = -1;
		int index = -1;
		for (int i = 0; i < idx.length; i++) {
			index = idx[i];
			r = index % nRow;
			c = index / nRow;
			A.setEntry(r, c, V.getEntry(i, 0));
		}
		
	}
	
	/**
	 * Matrix assignment by linear indexing for the syntax A(B) = v.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param idx a linear index vector
	 *          
	 * @param v a real scalar to assign A(idx)
	 * 
	 */
	public static void linearIndexingAssignment(Matrix A, int[] idx, double v) {
		
		int nRow = A.getRowDimension();
		// int nCol = A.getColumnDimension();
		int r = -1;
		int c = -1;
		int index = -1;
		for (int i = 0; i < idx.length; i++) {
			index = idx[i];
			r = index % nRow;
			c = index / nRow;
			A.setEntry(r, c, v);
		}
		
	}
	
	/**
	 * Matrix assignment by logical indexing for the syntax A(B) = v.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param B a logical matrix where position of each 1 determines
	 *          which array element is being assigned
	 *          
	 * @param v a real scalar to assign A(B)
	 * 
	 */
	public static void logicalIndexingAssignment(Matrix A, Matrix B, double v) {
		
		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices for logical indexing should have same size!");
			System.exit(1);
		}
		
		double b;
		if (B instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) B).getIr();
			int[] jc = ((SparseMatrix) B).getJc();
			double[] pr = ((SparseMatrix) B).getPr();
			for (int j = 0; j < nB; j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					b = pr[k];
					if (b == 1)
						A.setEntry(ir[k], j, v);
					else if (b != 0)
						err("Elements of the logical matrix should be either 1 or 0!");
				}
			}
		} else if (B instanceof DenseMatrix) {
			double[][] BData = ((DenseMatrix) B).getData();
			double[] BRow = null;
			for (int i = 0; i < dA; i++) {
				BRow = BData[i];
				for (int j = 0; j < nA; j++) {
					b = BRow[j];
					if (b == 1) {
						A.setEntry(i, j, v);
					}
					else if (b != 0)
						System.err.println("Elements of the logical matrix should be either 1 or 0!");
				}
			}
		}
		
	}
	
	/**
	 * Matrix assignment by logical indexing for the syntax A(B) = V.
	 * 
	 * @param A a matrix to be assigned
	 * 
	 * @param B a logical matrix where position of each 1 determines
	 *          which array element is being assigned
	 *          
	 * @param V a column matrix to assign A(B)
	 * 
	 */
	public static void logicalIndexingAssignment(Matrix A, Matrix B, Matrix V) {
		int nA = A.getColumnDimension();
		int dA = A.getRowDimension();
		int nB = B.getColumnDimension();
		int dB = B.getRowDimension();
		if (nA != nB || dA != dB) {
			System.err.println("The input matrices for logical indexing should have same size!");
			System.exit(1);
		}
		
		if (V == null)
			return;
		
		int nV = V.getColumnDimension();
		int dV = V.getRowDimension();
		
		if (nV != 1) {
			System.err.println("Assignment matrix should be a column matrix!");
			exit(1);
		}
		
		/*double b;
		int cnt = 0;
		for (int j = 0; j < nA; j++) {
			for (int i = 0; i < dA; i++) {
				b = B.getEntry(i, j);
				if (b == 1) {
					A.setEntry(i, j, V.getEntry(cnt++, 0));
				}
				else if (b != 0)
					System.err.println("Elements of the logical matrix should be either 1 or 0!");
			}
		}*/
		
		double b;
		int cnt = 0;
		if (B instanceof SparseMatrix) {
			int[] ir = ((SparseMatrix) B).getIr();
			int[] jc = ((SparseMatrix) B).getJc();
			double[] pr = ((SparseMatrix) B).getPr();
			for (int j = 0; j < nB; j++) {
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					b = pr[k];
					if (b == 1)
						A.setEntry(ir[k], j, V.getEntry(cnt++, 0));
					else if (b != 0)
						err("Elements of the logical matrix should be either 1 or 0!");
				}
			}
		} else if (B instanceof DenseMatrix) {
			double[][] BData = ((DenseMatrix) B).getData();
			for (int j = 0; j < nA; j++) {
				for (int i = 0; i < dA; i++) {
					b = BData[i][j];
					if (b == 1) {
						A.setEntry(i, j, V.getEntry(cnt++, 0));
					}
					else if (b != 0)
						System.err.println("Elements of the logical matrix should be either 1 or 0!");
				}
			}
		}
		
		if (cnt != dV)
			System.err.println("Assignment with different number of elements!");
		
	}
	
	/**
	 * Get the nonnegative part of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @return a matrix which is the nonnegative part of a matrix A
	 * 
	 */
	public static Matrix subplus(Matrix A) {
		Matrix res = A.copy();
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (resRow[j] < 0)
						resRow[j] = 0;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				if (pr[k] < 0)
					pr[k] = 0;
			}
			((SparseMatrix) res).clean();
		}
		return res;
	}
	
	/**
	 * Round towards zero.
	 * 
	 * @param x a real number
	 * 
	 * @return fix(x)
	 * 
	 */
	public static int fix(double x) {

		if (x > 0) {
			return (int)Math.floor(x);
		} else {
			return (int)Math.ceil(x);
		}

	}
	
	/**
	 * Generates a linearly spaced integer array with distance of
	 * D between two consecutive numbers. colon(J, D, K) is
	 * the same as [J, J+D, ..., J+m*D] where m = fix((K-J)/D).
	 * 
	 * @param begin starting point (inclusive)
	 * 
	 * @param d distance between two consecutive numbers
	 * 
	 * @param end ending point (inclusive if possible)
	 * 
	 * @return indices array for the syntax begin:d:end
	 * 
	 *//*
	public static int[] colon(int begin, int d, int end) {

		int m = fix((end - begin) / d);
		if (m < 0) {
			System.err.println("Difference error!");
			System.exit(1);
		}
		
		int[] res = new int[m + 1];
		
		for (int i = 0; i <= m; i++) {
			res[i] = begin + i * d;
		}

		return res;

	}
	
	*//**
	 * Same as colon(begin, 1, end).
	 * 
	 * @param begin starting point (inclusive)
	 * 
	 * @param end ending point (inclusive)
	 * 
	 * @return indices array for the syntax begin:end
	 * 
	 *//*
	public static int[] colon(int begin, int end) {
		return colon(begin, 1, end);
	}*/
	
	/**
	 * Returns an array that contains 1's where
	 * the elements of X are NaN's and 0's where they are not.
	 * 
	 * @param A a matrix
	 * 
	 * @return a 0-1 matrix: isnan(A)
	 * 
	 */
	public static Matrix isnan(Matrix A) {
		Matrix res = A.copy();
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (Double.isNaN(resRow[j]))
						resRow[j] = 1;
					else
						resRow[j] = 0;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				if (Double.isNaN(pr[k]))
					pr[k] = 1;
				else
					pr[k] = 0;
			}
			((SparseMatrix) res).clean();
		}
		return res;
	}
	
	/**
	 * returns an array that contains 1's where the
	 * elements of X are +Inf or -Inf and 0's where they are not.
	 * 
	 * @param A a real matrix
	 * 
	 * @return a 0-1 matrix: isinf(A)
	 * 
	 */
	public static Matrix isinf(Matrix A) {
		Matrix res = A.copy();
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (Double.isInfinite(resRow[j]))
						resRow[j] = 1;
					else
						resRow[j] = 0;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				if (Double.isInfinite(pr[k]))
					pr[k] = 1;
				else
					pr[k] = 0;
			}
			((SparseMatrix) res).clean();
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between A and B and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * A and B must have the same dimensions.
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return A | B or or(A, B)
	 * 
	 */
	public static Matrix or(Matrix A, Matrix B) {
		Matrix res = A.plus(B);
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (resRow[j] > 1)
						resRow[j] = 1;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				if (pr[k] > 1)
					pr[k] = 1;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between A and B and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * A and B must have the same dimensions.
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return A & B or and(A, B)
	 * 
	 */
	public static Matrix and(Matrix A, Matrix B) {
		Matrix res = A.times(B);
		return res;
	}
	
	/**
	 * Performs a logical NOT of input array A, and returns an array
     * containing elements set to either 1 (TRUE) or 0 (FALSE). An 
     * element of the output array is set to 1 if A contains a zero
     * value element at that same array location. Otherwise, that
     * element is set to 0.
	 * 
	 * @param A a matrix
	 * 
	 * @return ~A or not(A)
	 * 
	 */
	public static Matrix not(Matrix A) {
		Matrix res = minus(1, A);
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X ~= Y or ne(X, Y)
	 * 
	 */
	public static Matrix ne(Matrix X, Matrix Y) {
		Matrix res = X.minus(Y);
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (resRow[j] != 0)
						resRow[j] = 1;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				if (pr[k] != 0)
					pr[k] = 1;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X ~= x or ne(X, x)
	 * 
	 */
	public static Matrix ne(Matrix X, double x) {
		Matrix res = X.minus(x);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] != 0)
					resRow[j] = 1;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return X ~= x or ne(X, x)
	 * 
	 */
	public static Matrix ne(double x, Matrix X) {
		Matrix res = X.minus(x);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] != 0)
					resRow[j] = 1;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X == Y or eq(X, Y)
	 * 
	 */
	public static Matrix eq(Matrix X, Matrix Y) {
		return minus(1, ne(X, Y));
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X == x or eq(X, x)
	 * 
	 */
	public static Matrix eq(Matrix X, double x) {
		Matrix res = X.minus(x);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] != 0)
					resRow[j] = 0;
				else
					resRow[j] = 1;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return X == x or eq(X, x)
	 * 
	 */
	public static Matrix eq(double x, Matrix X) {
		return eq(X, x);
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X >= x or ge(X, x)
	 * 
	 */
	public static Matrix ge(Matrix X, double x) {
		Matrix res = X.minus(x);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] >= 0)
					resRow[j] = 1;
				else
					resRow[j] = 0;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x >= X or ge(x, X)
	 */
	public static Matrix ge(double x, Matrix X) {
		Matrix res = minus(x, X);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] >= 0)
					resRow[j] = 1;
				else
					resRow[j] = 0;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X >= Y or ge(X, Y)
	 * 
	 */
	public static Matrix ge(Matrix X, Matrix Y) {
		Matrix res = full(X.minus(Y));
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] >= 0)
					resRow[j] = 1;
				else
					resRow[j] = 0;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X <= x or le(X, x)
	 * 
	 */
	public static Matrix le(Matrix X, double x) {
		return ge(x, X);
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x <= X or le(x, X)
	 * 
	 */
	public static Matrix le(double x, Matrix X) {
		return ge(X, x);
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X <= Y or le(X, Y)
	 * 
	 */
	public static Matrix le(Matrix X, Matrix Y) {
		return ge(Y, X);
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X > x or gt(X, x)
	 * 
	 */
	public static Matrix gt(Matrix X, double x) {
		Matrix res = X.minus(x);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] > 0)
					resRow[j] = 1;
				else
					resRow[j] = 0;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x > X or gt(x, X)
	 */
	public static Matrix gt(double x, Matrix X) {
		Matrix res = minus(x, X);
		// res must be a dense matrix.
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] > 0)
					resRow[j] = 1;
				else
					resRow[j] = 0;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X > Y or gt(X, Y)
	 * 
	 */
	public static Matrix gt(Matrix X, Matrix Y) {
		Matrix res = full(X.minus(Y));
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[][] resData = ((DenseMatrix) res).getData();
		double[] resRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			for (int j = 0; j < N; j++) {
				if (resRow[j] > 0)
					resRow[j] = 1;
				else
					resRow[j] = 0;
			}
		}
		return res;
	}
	
	/**
	 * Do element by element comparisons between X and x and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param X a matrix
	 * 
	 * @param x a real scalar
	 * 
	 * @return X < x or lt(X, x)
	 * 
	 */
	public static Matrix lt(Matrix X, double x) {
		return gt(x, X);
	}
	
	/**
	 * Do element by element comparisons between x and X and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * 
	 * @param x a real scalar
	 * 
	 * @param X a matrix
	 * 
	 * @return x < X or lt(x, X)
	 * 
	 */
	public static Matrix lt(double x, Matrix X) {
		return gt(X, x);
	}
	
	/**
	 * Do element by element comparisons between X and Y and returns
	 * a matrix of the same size with elements set to 1 where the
	 * relation is true and elements set to 0 where it is not.
	 * X and Y must have the same dimensions.
	 * 
	 * @param X a matrix
	 * 
	 * @param Y a matrix
	 * 
	 * @return X < Y or lt(X, Y)
	 * 
	 */
	public static Matrix lt(Matrix X, Matrix Y) {
		return gt(Y, X);
	}
	
	/**
	 * Compute element-wise absolute value of all elements of matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return abs(A)
	 */
	public static Matrix abs(Matrix A) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		Matrix res = A.copy();
		
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				for (int j = 0; j < nCol; j++) {
					resRow[j] = Math.abs(resRow[j]);
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			for (int k = 0; k < pr.length; k++) {
				pr[k] = Math.abs(pr[k]);
			}
		}

		return res;

	}
	
	/**
	 * Compute element-wise absolute value of all elements of a vector.
	 * 
	 * @param V a vector
	 * 
	 * @return abs(V)
	 */
	public static Vector abs(Vector V) {
		Vector res = V.copy();
		if (res instanceof DenseVector) {
			double[] pr = ((DenseVector) res).getPr();
			for (int k = 0; k < res.getDim(); k++) {
				pr[k] = Math.abs(pr[k]);
			}
		} else if (res instanceof SparseVector) {
			double[] pr = ((SparseVector) res).getPr();
			int nnz = ((SparseVector) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] = Math.abs(pr[k]);
			}
		}
		return res;
	}
	
	/**
	 * Compute element-wise exponentiation of a vector.
	 * 
	 * @param A a real matrix
	 * 
	 * @param p exponent
	 * 
	 * @return A.^p
	 */
	public static Matrix pow(Matrix A, double p) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();

		Matrix res = A.copy();
		
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				for (int j = 0; j < nCol; j++) {
					resRow[j] = Math.pow(resRow[j], p);
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			for (int k = 0; k < pr.length; k++) {
				pr[k] = Math.pow(pr[k], p);
			}
		}

		return res;

	}
	
	/**
	 * Compute element-wise exponentiation of a vector.
	 * 
	 * @param V a real vector
	 * 
	 * @param p exponent
	 * 
	 * @return V.^p
	 */
	public static Vector pow(Vector V, double p) {
		Vector res = V.copy();
		if (res instanceof DenseVector) {
			double[] pr = ((DenseVector) res).getPr();
			for (int k = 0; k < res.getDim(); k++) {
				pr[k] = Math.pow(pr[k], p);
			}
		} else if (res instanceof SparseVector) {
			double[] pr = ((SparseVector) res).getPr();
			int nnz = ((SparseVector) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] = Math.pow(pr[k], p);
			}
		}
		return res;
	}
	
	/**
     * Compute the maximal value and the corresponding row 
     * index of a vector V. The index starts from 0.
	 * 
     * @param V a real vector
     * 
     * @return a {@code double} array [M, IX] where M is the 
     *         maximal value, and IX is the corresponding
     * 		   index
     */
	public static double[] max(Vector V) {
		double[] res = new double[2];
		double maxVal = Double.NEGATIVE_INFINITY;
		double maxIdx = -1;
		double v = 0;
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			for (int k = 0; k < pr.length; k++) {
				v = pr[k];
				if (maxVal < v) {
					maxVal = v;
					maxIdx = k;
				}
			}
			
		} else if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			int dim = V.getDim();
			if (nnz == 0) {
				maxVal = 0;
				maxIdx = 0;
			} else {
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx;) {
						if (maxVal < 0) {
							maxVal = 0;
							maxIdx = i;
						}
						break;
					}
					v = pr[k];
					if (maxVal < v) {
						maxVal = v;
						maxIdx = currentIdx;
					}
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim;) {
					if (maxVal < 0) {
						maxVal = 0;
						maxIdx = i;
					}
					break;
				}
			}
		}
		res[0] = maxVal;
		res[1] = maxIdx;
		return res;
	}
	
	/**
     * Compute the maximal value for each column and the 
     * corresponding row indices of a matrix A. The row index
     * starts from 0.
	 * 
     * @param A a real matrix
     * 
     * @return a {@code Vector} array [M, IX] where M contains
     * 		   the maximal values, and IX contains the corresponding
     * 		   indices
     */
	public static Vector[] max(Matrix A) {
		return max(A, 1);
	}
	
	/**
	 * Compute the maximal value for each row or each column
	 * and the corresponding indices of a matrix A. The row or 
	 * column indices start from 0.
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param dim 1: column-wise; 2: row-wise
	 * 
	 * @return a {@code double[]} array [M, IX] where M contains
     * 		   the maximal values, and IX contains the corresponding
     * 		   indices
	 */
	public static double[][] max(double[][] A, int dim) {
		double[][] res = new double[2][];
		double[][] AData = A;
		int M = A.length;
		int N = A[0].length;
		double maxVal = 0;
		int maxIdx = -1;
		double v = 0;
		double[] maxValues = null;
		maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
		double[] maxIndices = null;
		maxIndices = ArrayOperator.allocate1DArray(N, 0);
		
		double[] ARow = null;
		
		if (dim == 1) {
			maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
			maxIndices = ArrayOperator.allocate1DArray(N, 0);
			for (int i = 0; i < M; i++) {
				ARow = AData[i];
				for (int j = 0; j < N; j++) {
					v = ARow[j];
					if (maxValues[j] < v) {
						maxValues[j] = v;
						maxIndices[j] = i;
					}
				}
			}
		} else if (dim == 2) {
			maxValues = ArrayOperator.allocate1DArray(M, Double.NEGATIVE_INFINITY);
			maxIndices = ArrayOperator.allocate1DArray(M, 0);
			for (int i = 0; i < M; i++) {
				ARow = AData[i];
				maxVal = ARow[0];
				maxIdx = 0;
				for (int j = 1; j < N; j++) {
					v = ARow[j];
					if (maxVal < v) {
						maxVal = v;
						maxIdx = j;
					}
				}
				maxValues[i] = maxVal;
				maxIndices[i] = maxIdx;
			}
		}
		
		res[0] = maxValues;
		res[1] = maxIndices;
		return res;
	}
	
    /**
     * Compute the maximal value for each row or each column
	 * and the corresponding indices of a matrix A. The row or 
	 * column indices start from 0.
	 * 
     * @param A a real matrix
     * 
     * @param dim 1: column-wise; 2: row-wise
     * 
     * @return a {@code Vector} array [M, IX] where M contains
     * 		   the maximal values, and IX contains the corresponding
     * 		   indices
     */
	public static Vector[] max(Matrix A, int dim) {
		Vector[] res = new DenseVector[2];
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		double maxVal = 0;
		int maxIdx = -1;
		double v = 0;
		double[] maxValues = null;
		maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
		double[] maxIndices = null;
		maxIndices = ArrayOperator.allocate1DArray(N, 0);
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			
			if (dim == 1) {
				maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
				maxIndices = ArrayOperator.allocate1DArray(N, 0);
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						v = ARow[j];
						if (maxValues[j] < v) {
							maxValues[j] = v;
							maxIndices[j] = i;
						}
					}
				}
			} else if (dim == 2) {
				maxValues = ArrayOperator.allocate1DArray(M, Double.NEGATIVE_INFINITY);
				maxIndices = ArrayOperator.allocate1DArray(M, 0);
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					maxVal = ARow[0];
					maxIdx = 0;
					for (int j = 1; j < N; j++) {
						v = ARow[j];
						if (maxVal < v) {
							maxVal = v;
							maxIdx = j;
						}
					}
					maxValues[i] = maxVal;
					maxIndices[i] = maxIdx;
				}
			}
		} else if (A instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) A).getPr();
			if (dim == 1) {
				maxValues = ArrayOperator.allocate1DArray(N, Double.NEGATIVE_INFINITY);
				maxIndices = ArrayOperator.allocate1DArray(N, 0);
				int[] ir = ((SparseMatrix) A).getIr();
				int[] jc = ((SparseMatrix) A).getJc();
				for (int j = 0; j < N; j++) {
					if (jc[j] == jc[j + 1]) {
						maxValues[j] = 0;
						maxIndices[j] = 0;
						continue;
					}
					maxVal = Double.NEGATIVE_INFINITY;
					maxIdx = -1;
					int lastRowIdx = -1;
					int currentRowIdx = 0;
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						currentRowIdx = ir[k];
						for (int r = lastRowIdx + 1; r < currentRowIdx;) {
							if (maxVal < 0) {
								maxVal = 0;
								maxIdx = r;
							}
							break;
						}
						v = pr[k];
						if (maxVal < v) {
							maxVal = v;
							maxIdx = ir[k];
						}
						lastRowIdx = currentRowIdx;
					}
					for (int r = lastRowIdx + 1; r < M;) {
						if (maxVal < 0) {
							maxVal = 0;
							maxIdx = r;
						}
						break;
					}
					maxValues[j] = maxVal;
					maxIndices[j] = maxIdx;
				}
			} else if (dim == 2) {
				maxValues = ArrayOperator.allocate1DArray(M, Double.NEGATIVE_INFINITY);
				maxIndices = ArrayOperator.allocate1DArray(M, 0);
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				for (int i = 0; i < M; i++) {
					if (jr[i] ==  jr[i + 1]) {
						maxValues[i] = 0;
						maxIndices[i] = 0;
						continue;
					}
					maxVal = Double.NEGATIVE_INFINITY;
					maxIdx = -1;
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx;) {
							if (maxVal < 0) {
								maxVal = 0;
								maxIdx = c;
							}
							break;
						}
						v = pr[valCSRIndices[k]];
						if (maxVal < v) {
							maxVal = v;
							maxIdx = ic[k];
						}
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N;) {
						if (maxVal < 0) {
							maxVal = 0;
							maxIdx = c;
						}
						break;
					}
					maxValues[i] = maxVal;
					maxIndices[i] = maxIdx;
				}
			}
		}
		res[0] = new DenseVector(maxValues);
		res[1] = new DenseVector(maxIndices);
		return res;
	}
	
	/**
     * Compute the minimal value and the corresponding row 
     * index of a vector V. The index starts from 0.
	 * 
     * @param V a real vector
     * 
     * @return a {@code double} array [M, IX] where M is the 
     *         minimal value, and IX is the corresponding
     * 		   index
     */
	public static double[] min(Vector V) {
		double[] res = new double[2];
		double minVal = Double.POSITIVE_INFINITY;
		double minIdx = -1;
		double v = 0;
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			for (int k = 0; k < pr.length; k++) {
				v = pr[k];
				if (minVal > v) {
					minVal = v;
					minIdx = k;
				}
			}
			
		} else if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			int dim = V.getDim();
			if (nnz == 0) {
				minVal = 0;
				minIdx = 0;
			} else {
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx;) {
						if (minVal > 0) {
							minVal = 0;
							minIdx = i;
						}
						break;
					}
					v = pr[k];
					if (minVal > v) {
						minVal = v;
						minIdx = currentIdx;
					}
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim;) {
					if (minVal > 0) {
						minVal = 0;
						minIdx = i;
					}
					break;
				}
			}
		}
		res[0] = minVal;
		res[1] = minIdx;
		return res;
	}
	
	/**
     * Compute the minimal value for each column and the 
     * corresponding row indices of a matrix A. The row index
     * starts from 0.
	 * 
     * @param A a real matrix
     * 
     * @return a {@code Vector} array [M, IX] where M contains
     * 		   the minimal values, and IX contains the corresponding
     * 		   indices
     */
	public static Vector[] min(Matrix A) {
		return min(A, 1);
	}
	
	/**
     * Compute the minimal value for each row or each column
	 * and the corresponding indices of a matrix A. The row or 
	 * column indices start from 0.
	 * 
     * @param A a real matrix
     * 
     * @param dim 1: column-wise; 2: row-wise
     * 
     * @return a {@code Vector} array [M, IX] where M contains
     * 		   the minimal values, and IX contains the corresponding
     * 		   indices
     */
	public static Vector[] min(Matrix A, int dim) {
		Vector[] res = new DenseVector[2];
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		double minVal = 0;
		int minIdx = -1;
		double v = 0;
		double[] minValues = null;
		minValues = ArrayOperator.allocate1DArray(N, Double.POSITIVE_INFINITY);
		double[] minIndices = null;
		minIndices = ArrayOperator.allocate1DArray(N, 0);
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			
			if (dim == 1) {
				minValues = ArrayOperator.allocate1DArray(N, Double.POSITIVE_INFINITY);
				minIndices = ArrayOperator.allocate1DArray(N, 0);
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						v = ARow[j];
						if (minValues[j] > v) {
							minValues[j] = v;
							minIndices[j] = i;
						}
					}
				}
			} else if (dim == 2) {
				minValues = ArrayOperator.allocate1DArray(M, Double.POSITIVE_INFINITY);
				minIndices = ArrayOperator.allocate1DArray(M, 0);
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					minVal = ARow[0];
					minIdx = 0;
					for (int j = 1; j < N; j++) {
						v = ARow[j];
						if (minVal > v) {
							minVal = v;
							minIdx = j;
						}
					}
					minValues[i] = minVal;
					minIndices[i] = minIdx;
				}
			}
		} else if (A instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) A).getPr();
			if (dim == 1) {
				minValues = ArrayOperator.allocate1DArray(N, Double.POSITIVE_INFINITY);
				minIndices = ArrayOperator.allocate1DArray(N, 0);
				int[] ir = ((SparseMatrix) A).getIr();
				int[] jc = ((SparseMatrix) A).getJc();
				for (int j = 0; j < N; j++) {
					if (jc[j] == jc[j + 1]) {
						minValues[j] = 0;
						minIndices[j] = 0;
						continue;
					}
					minVal = Double.POSITIVE_INFINITY;
					minIdx = -1;
					int lastRowIdx = -1;
					int currentRowIdx = 0;
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						currentRowIdx = ir[k];
						for (int r = lastRowIdx + 1; r < currentRowIdx;) {
							if (minVal > 0) {
								minVal = 0;
								minIdx = r;
							}
							break;
						}
						v = pr[k];
						if (minVal > v) {
							minVal = v;
							minIdx = ir[k];
						}
						lastRowIdx = currentRowIdx;
					}
					for (int r = lastRowIdx + 1; r < M;) {
						if (minVal > 0) {
							minVal = 0;
							minIdx = r;
						}
						break;
					}
					minValues[j] = minVal;
					minIndices[j] = minIdx;
				}
			} else if (dim == 2) {
				minValues = ArrayOperator.allocate1DArray(M, Double.POSITIVE_INFINITY);
				minIndices = ArrayOperator.allocate1DArray(M, 0);
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				for (int i = 0; i < M; i++) {
					if (jr[i] ==  jr[i + 1]) {
						minValues[i] = 0;
						minIndices[i] = 0;
						continue;
					}
					minVal = Double.POSITIVE_INFINITY;
					minIdx = -1;
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx;) {
							if (minVal > 0) {
								minVal = 0;
								minIdx = c;
							}
							break;
						}
						v = pr[valCSRIndices[k]];
						if (minVal > v) {
							minVal = v;
							minIdx = ic[k];
						}
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N;) {
						if (minVal > 0) {
							minVal = 0;
							minIdx = c;
						}
						break;
					}
					minValues[i] = minVal;
					minIndices[i] = minIdx;
				}
			}
		}
		res[0] = new DenseVector(minValues);
		res[1] = new DenseVector(minIndices);
		return res;
	}
	
	/**
	 * Compute the sum of elements for each row of a real matrix.
	 * 
	 * @param A a real matrix
	 * 
	 * @return a dense vector containing the sum of elements of 
	 *         each row of A, i.e. sum(A, 1)
	 */
	public static DenseVector sum(Matrix A) {
		return sum(A, 1);
	}

	/**
	 * Compute the sum of elements for each row or each column of
	 * a real matrix.
	 * 
	 * @param A a real matrix
	 * 
	 * @param dim 1: column-wise; 2: row-wise
	 * 
	 * @return a dense vector containing the sum of elements of
	 * 		   each row or each column of A
	 */
	public static DenseVector sum(Matrix A, int dim) {
		double[] sumValues = null;
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		double s = 0;
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] ARow = null;
			if (dim == 1) {
				sumValues = ArrayOperator.allocate1DArray(N, 0);
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						s = ARow[j];
						if (s != 0)
							sumValues[j] += s;
					}
				}
			} else if (dim == 2) {
				sumValues = ArrayOperator.allocate1DArray(M, 0);
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					s = 0;
					for (int j = 0; j < N; j++) {
						s += ARow[j];
					}
					sumValues[i] = s;
				}
			}
		} else if (A instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) A).getPr();
			if (dim == 1) {
				sumValues = ArrayOperator.allocate1DArray(N, 0);
				// int[] ir = ((SparseMatrix) A).getIr();
				int[] jc = ((SparseMatrix) A).getJc();
				for (int j = 0; j < N; j++) {
					if (jc[j] == jc[j + 1]) {
						sumValues[j] = 0;
						continue;
					}
					s = 0;	
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						s += pr[k];
					}	
					sumValues[j] = s;
				}
			} else if (dim == 2) {
				sumValues = ArrayOperator.allocate1DArray(M, 0);
				// int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				for (int i = 0; i < M; i++) {
					if (jr[i] ==  jr[i + 1]) {
						sumValues[i] = 0;
						continue;
					}
					s = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						s += pr[valCSRIndices[k]];
					}
					sumValues[i] = s;
				}
			}
		}
		return new DenseVector(sumValues);
	}

	/**
	 * Compute the sum of a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return sum(V)
	 *//*
	public static double sum(double[] V) {
		double res = 0;
		for (int i = 0; i < V.length; i++)
			res += V[i];
		return res;
	}*/
	
	/**
	 * Compute the sum of all elements of a vector.
	 * 
	 * @param V a real dense or sparse vector
	 * 
	 * @return sum(V)
	 */
	public static double sum(Vector V) {
		double res = 0;
		if (V instanceof DenseVector) {
			double[] pr = ((DenseVector) V).getPr();
			for (int k = 0; k < pr.length; k++) {
				res += pr[k];
			}
			
		} else if (V instanceof SparseVector) {
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			for (int k = 0; k < nnz; k++) {
				res += pr[k];
			}
		}
		return res;
	}
	
	/**
	 * Compute the Euclidean norm of a matrix.
	 * 
	 * @param A a real matrix
	 * 
	 * @return ||A||_2
	 */
	public static double norm(Matrix A) {
		return norm(A, 2);
	}
	
	/**
	 * Compute the induced vector norm of a matrix (row or column
	 * matrices are allowed).
	 * 
	 * @param A a matrix or a vector
	 * 
	 * @param type 1, 2, or, inf for a matrix or a positive real 
	 *             number for a row or column matrix
	 * 
	 * @return ||A||_{type}
	 * 
	 */
	public static double norm(Matrix A, double type) {

		double res = 0;
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		
		if (nRow == 1) {
			if (Double.isInfinite(type)) {
				return max(max(abs(A), 2)[0])[0];
			} else if (type > 0) {
				return Math.pow(sum(sum(pow(abs(A), type))), 1.0 / type);
			} else {
				System.err.printf("Error norm type: %f\n", type);
				System.exit(1);
			}
		}
		
		if (nCol == 1) {
			if (Double.isInfinite(type)) {
				return max(max(abs(A), 1)[0])[0];
			} else if (type > 0) {
				return Math.pow(sum(sum(pow(abs(A), type))), 1.0 / type);
			} else {
				System.err.printf("Error norm type: %f\n", type);
				System.exit(1);
			}
		}
		
		if (type == 2) {
			double eigenvalue = EigenValueDecomposition.computeEigenvalues(A.transpose().mtimes(A))[0];
			res = eigenvalue <= 0 ? 0 : Math.sqrt(eigenvalue);
			// res = SingularValueDecomposition.computeSingularValues(A)[0];
		} else if (Double.isInfinite(type)) {
			res = max(sum(abs(A), 2))[0];
		} else if (type == 1) {
			res = max(sum(abs(A), 1))[0];
		} else {
			System.err.printf("Sorry, %f-norm of a matrix is not supported currently.\n", type);
		}
		
		return res;
		
	}
	
	/**
	 * Compute the norm of a matrix (row or column matrices
	 * are allowed).
	 * 
	 * @param A a matrix
	 * 
	 * @param type 1, 2
	 * 
	 * @return ||A||_{type}
	 * 
	 */
	public static double norm(Matrix A, int type) {
		return norm(A, (double)type);
	}
	
	/**
	 * Calculate the Frobenius norm of a matrix A.
	 * 
	 * @param A a matrix
	 * 
	 * @param type can only be "fro"
	 * 
	 * @return ||A||_F
	 * 
	 */
	public static double norm(Matrix A, String type) {

		double res = 0;
		if (type.compareToIgnoreCase("fro") == 0) {
			res = Math.sqrt(innerProduct(A, A));
		} else if (type.equals("inf")){
			res = norm(A, inf);
		} else if (type.equals("nuclear")) {
			res = ArrayOperator.sum(SingularValueDecomposition.computeSingularValues(A));
		} else {
			System.err.println(String.format("Norm %s unimplemented!\n" , type));
		}
		return res;
	}
	
	/**
	 * Compute the norm of a vector.
	 * 
	 * @param V a real vector
	 * 
	 * @param p a positive {@code double} value
	 * 
	 * @return ||V||_p
	 */
	public static double norm(Vector V, double p) {
		if (p == 1) {
			return sum(abs(V));
		} else if (p == 2) {
			return Math.sqrt(innerProduct(V, V));
		} else if (Double.isInfinite(p)) {
			return max(abs(V))[0];
		} else if (p > 0) {
			return Math.pow(sum(pow(abs(V), p)), 1.0 / p);
		} else {
			System.err.println("Wrong argument for p");
			System.exit(1);
		}
		return -1;
	}
	
	/**
	 * Compute the Euclidean norm of a vector.
	 * 
	 * @param V a real vector
	 * 
	 * @return ||V||_2
	 */
	public static double norm(Vector V) {
		return norm(V, 2);
	}
	
	/**
	 * Compute the norm of a vector.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param p a positive {@code double} value
	 * 
	 * @return ||V||_p
	 */
	public static double norm(double[] V, double p) {
		return norm(new DenseVector(V), p);
	}
	
	/**
	 * Compute the Euclidean norm of a vector.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return ||V||_2
	 */
	public static double norm(double[] V) {
		return norm(V, 2);
	}
	
	/**
	 * Compute the inner product of two vectors, i.e. res = <V1, V2>.
	 * 
	 * @param V1 the first vector
	 * 
	 * @param V2 the second vector
	 * 
	 * @return <V1, V2>
	 */
	public static double innerProduct(Vector V1, Vector V2) {
		if (V1.getDim() != V2.getDim()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		double res = 0;
		double v = 0;
		if (V1 instanceof DenseVector) {
			double[] pr1 = ((DenseVector) V1).getPr();
			if (V2 instanceof DenseVector) {
				if (V1 == V2) {
					for (int k = 0; k < pr1.length; k++) {
						v = pr1[k];
						res += v * v; 
					}
					return res;
				}
				
				double[] pr2 = ((DenseVector) V2).getPr();
				for (int k = 0; k < pr1.length; k++) {
					res += pr1[k] * pr2[k]; 
				}
			} else if (V2 instanceof SparseVector) {
				int[] ir = ((SparseVector) V2).getIr();
				double[] pr2 = ((SparseVector) V2).getPr();
				int nnz = ((SparseVector) V2).getNNZ();
				for (int k = 0; k < nnz; k++) {
					res += pr1[ir[k]] * pr2[k];
				}
			}
		} else if (V1 instanceof SparseVector) {
			if (V2 instanceof DenseVector) {
				return innerProduct(V2, V1);
			} else if (V2 instanceof SparseVector) {
				int[] ir1 = ((SparseVector) V1).getIr();
				double[] pr1 = ((SparseVector) V1).getPr();
				int nnz1 = ((SparseVector) V1).getNNZ();
				if (V1 == V2) {
					for (int k = 0; k < nnz1; k++) {
						v = pr1[k];
						res += v * v;
					}
					return res;
				} else {
					int[] ir2 = ((SparseVector) V2).getIr();
					double[] pr2 = ((SparseVector) V2).getPr();
					int nnz2 = ((SparseVector) V2).getNNZ();
					int k1 = 0;
					int k2 = 0;
					int i1 = 0;
					int i2 = 0;
					while (true) {
						if (k1 >= nnz1 || k2 >= nnz2) {
							break;
						}
						i1 = ir1[k1];
						i2 = ir2[k2];
						if (i1 < i2) {
							k1++;
						} else if (i1 > i2) {
							k2++;
						} else {
							res += pr1[k1] * pr2[k2];
							k1++;
							k2++;
						}
					}
				}
			}
		}
		return res;
	}
	
	/**
	 * Compute the inner product of two matrices, i.e. res = <A, B>.
	 * 
	 * @param A a matrix
	 * 
	 * @param B a matrix
	 * 
	 * @return <A, B>
	 * 
	 */
	public static double innerProduct(Matrix A, Matrix B) {
		
		double s = 0;
		
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			if (B instanceof DenseMatrix) {
				double[][] BData = ((DenseMatrix) B).getData();
				double[] BRow = null;
				double[] ARow = null;
				// double[] resRow = null;
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					BRow = BData[i];
					for (int j = 0; j < N; j++) {
						s += ARow[j] * BRow[j];
					}
				}
			} else if (B instanceof SparseMatrix) {
				int[] ir = null;
				int[] jc = null;
				double[] pr = null;
				ir = ((SparseMatrix) B).getIr();
				jc = ((SparseMatrix) B).getJc();
				pr = ((SparseMatrix) B).getPr();
				int r = -1;
				for (int j = 0; j < B.getColumnDimension(); j++) {
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						r = ir[k];
						// A[r][j] = pr[k]
						s += AData[r][j] * pr[k];
					}
				}
			}
		} else if (A instanceof SparseMatrix) {
			if (B instanceof DenseMatrix) {
				return innerProduct(B, A);
			} else if (B instanceof SparseMatrix) {
				int[] ir1 = null;
				int[] jc1 = null;
				double[] pr1 = null;
				ir1 = ((SparseMatrix) A).getIr();
				jc1 = ((SparseMatrix) A).getJc();
				pr1 = ((SparseMatrix) A).getPr();
				int[] ir2 = null;
				int[] jc2 = null;
				double[] pr2 = null;
				ir2 = ((SparseMatrix) B).getIr();
				jc2 = ((SparseMatrix) B).getJc();
				pr2 = ((SparseMatrix) B).getPr();
				
				int k1 = 0;
				int k2 = 0;
				int r1 = -1;
				int r2 = -1;
				// int i = -1;
				double v = 0;
				
				for (int j = 0; j < N; j++) {
					k1 = jc1[j];
					k2 = jc2[j];
					
					// If the j-th column of A or this is empty, we don't need to compute.
					if (k1 == jc1[j + 1] || k2 == jc2[j + 1])
						continue;
					
					while (k1 < jc1[j + 1] && k2 < jc2[j + 1]) {
						
						r1 = ir1[k1];
						r2 = ir2[k2];
						if (r1 < r2) {
							k1++;
						} else if (r1 == r2) {
							// i = r1;
							v = pr1[k1] * pr2[k2];
							k1++;
							k2++;
							if (v != 0) {
								s += v;
							}
						} else {
							k2++;
						}
						
					}
					
				}
			}
		}
		return s;
		
	}
	
	/**
	 * Calculate the sigmoid of a matrix A by rows. Specifically, supposing 
	 * that the input activation matrix is [a11, a12; a21, a22], the output 
	 * value is 
	 * <p>
	 * [exp(a11) / exp(a11) + exp(a12), exp(a12) / exp(a11) + exp(a12); 
	 * </br>
	 * exp(a21) / exp(a21) + exp(a22), exp(a22) / exp(a21) + exp(a22)].
	 * 
	 * @param A a real matrix
	 * 
	 * @return sigmoid(A)
	 */
	public static Matrix sigmoid(Matrix A) {
		double[][] data = full(A.copy()).getData();
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		double[] row_i = null;
		double old = 0;
		double current = 0;
		double max = 0;
		double sum = 0;
		double v = 0;
		for (int i = 0; i < M; i++) {
			row_i = data[i];
			old = row_i[0];
			current = 0;
			max = old;
			for (int j = 1; j < N; j++) {
				current = row_i[j];
				if (max < current)
					max = current;
				old = current;
			}
			sum = 0;
			for (int j = 0; j < N; j++) {
				v = Math.exp(row_i[j] - max);
				sum += v;
				row_i[j] = v;
			}
			for (int j = 0; j < N; j++) {
				row_i[j] /= sum; 
			}
		}
		return new DenseMatrix(data);
	}
	
	/**
	 * Compute the maximum of elements in a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return max(V)
	 */
	public static double max(double[] V) {
		// double max = max(V, 0, V.length - 1);
		double max = V[0];
		for (int i = 1; i < V.length; i++) {
			if (max < V[i])
				max = V[i];
		}
		return max;
	}
	
	/**
	 * Compute the maximum of elements in an interval 
	 * of a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param start start index (inclusive)
	 * 
	 * @param end end index (inclusive)
	 * 
	 * @return max(V(start:end))
	 */
	public static double max(double[] V, int start, int end) {
		if (start == end)
			return V[start];
		if (start == end - 1)
			return Math.max(V[start], V[end]);
		
		int middle = (start + end) / 2;
		double leftMax = max(V, start, middle);
		double rightMax = max(V, middle + 1, end);
		return Math.max(leftMax, rightMax);
	}
	
	/**
	 * Calculate the left division of the form A \ B. A \ B is the
	 * matrix division of A into B, which is roughly the same as
	 * INV(A)*B , except it is computed in a different way. For 
	 * implementation, we actually solve the system of linear 
	 * equations A * X = B.
	 * 
	 * @param A divisor
	 * 
	 * @param B dividend
	 * 
	 * @return A \ B
	 * 
	 */
	public static Matrix mldivide(Matrix A, Matrix B) {
		return new QRDecomposition(A).solve(B);
	}
	
	/**
	 * Calculate the right division of B into A, i.e., A / B. For
	 * implementation, we actually solve the system of linear 
	 * equations X * B = A.
	 * <p>
	 * Note: X = A / B <=> X * B = A <=> B' * X' = A' <=> X' = B' \ A'
	 * <=> X = (B' \ A')'
	 * </p>
	 * 
	 * @param A dividend
	 * 
	 * @param B divisor
	 * 
	 * @return A / B
	 * 
	 */
	public static Matrix mrdivide(Matrix A, Matrix B) {
		return mldivide(B.transpose(), A.transpose()).transpose();
	}
	
	/**
	 * Compute the rank of a matrix. The rank function provides 
	 * an estimate of the number of linearly independent rows or 
	 * columns of a matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return rank of the given matrix
	 */
	public static int rank(Matrix A) {
		return SingularValueDecomposition.rank(A);
	}
	
	/**
	 * Construct an m-by-n dense identity matrix.
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @return an m-by-n dense identity matrix
	 * 
	 */
	public static DenseMatrix eye(int m, int n) {
		double[][] res = ArrayOperator.allocate2DArray(m, n, 0);
		int len = m >= n ? n : m;
		for (int i = 0; i < len; i++) {
			res[i][i] = 1;
		}
		return new DenseMatrix(res);
	} 
	
	/**
	 * Construct an n-by-n dense identity matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return an n-by-n dense identity matrix
	 * 
	 */
	public static DenseMatrix eye(int n) {
		return eye(n, n);
	}
	
	/**
	 * Generate a dense identity matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return a dense identity matrix with its shape specified by size 
	 * 
	 */
	public static Matrix eye(int... size) {
		if (size.length != 2) {
			System.err.println("Input size vector should have two elements!");
		}
		return eye(size[0], size[1]);
	}
	
	/**
	 * Construct an m-by-n sparse identity matrix.
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @return an m-by-n sparse identity matrix
	 * 
	 */
	public static SparseMatrix speye(int m, int n) {
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		int len = m >= n ? n : m;
		for (int i = 0; i < len; i++) {
			map.put(Pair.of(i, i), 1.0);
		}
		return SparseMatrix.createSparseMatrix(map, m, n);
	}
	
	/**
	 * Construct an n-by-n sparse identity matrix.
	 * 
	 * @param n number of rows and columns
	 * 
	 * @return an n-by-n sparse identity matrix
	 * 
	 */
	public static SparseMatrix speye(int n) {
		/*TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (int i = 0; i < n; i++) {
			map.put(Pair.of(i, i), 1.0);
		}
		return SparseMatrix.createSparseMatrix(map, n, n);*/
		return speye(n, n);
	}
	
	/**
	 * Generate a sparse identity matrix with its size
	 * specified by a two dimensional integer array.
	 * 
	 * @param size a two dimensional integer array 
	 * 
	 * @return a sparse identity matrix with its shape specified by size 
	 * 
	 */
	public static Matrix speye(int... size) {
		if (size.length != 2) {
			System.err.println("Input size vector should have two elements!");
		}
		return speye(size[0], size[1]);
	}
	
	/**
	 * Create a m-by-n Hilbert matrix. The elements of the 
	 * Hilbert matrices are H(i, j) = 1 / (i + j + 1), where
	 * i and j are zero based indices.
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @return a m-by-n Hilbert matrix
	 */
	public static Matrix hilb(int m, int n) {
		DenseMatrix A = new DenseMatrix(m, n);
		double[][] data = A.getData();
		double[] A_i = null;
		for (int i = 0; i < m; i++) {
			A_i = data[i];
			for (int j = 0; j < n; j++) {
				A_i[j] = 1.0 / (i + j + 1);
			}
		}
		return A;
	}
	
	public static Vector times(Vector V1, Vector V2) {
		return V1.times(V2);
	}
	
	public static Vector plus(Vector V1, Vector V2) {
		return V1.plus(V2);
	}
	
	public static Vector minus(Vector V1, Vector V2) {
		return V1.minus(V2);
	}
	
	public static Matrix times(Matrix X, Matrix Y) {
		int nX = X.getColumnDimension();
		int dX = X.getRowDimension();
		int nY = Y.getColumnDimension();
		int dY = Y.getRowDimension();
		if (dX == 1 && nX == 1) {
			return times(X.getEntry(0, 0), Y);
		} else if (dY == 1 && nY == 1) {
			return times(X, Y.getEntry(0, 0));
		}
		if (nX != nY || dX != dY) {
			System.err.println("The operands for Hadmada product should be of same shapes!");
		}
		return X.times(Y);
	}
	
	public static Matrix times(Matrix A, double v) {
		return A.times(v);
	}
	
	public static Matrix times(double v, Matrix A) {
		return A.times(v);
	}
	
	public static Matrix mtimes(Matrix M1, Matrix M2) {
		return M1.mtimes(M2);
	}
	
	public static Matrix plus(Matrix M1, Matrix M2) {
		return M1.plus(M2);
	}
	
	public static Matrix plus(Matrix A, double v) {
		return A.plus(v);
	}
	
	public static Matrix plus(double v, Matrix A) {
		return A.plus(v);
	}
	
	public static Matrix minus(Matrix M1, Matrix M2) {
		return M1.minus(M2);
	}
	
	public static Matrix minus(Matrix A, double v) {
		return A.minus(v);
	}
	
	/**
	 * res = v - A.
	 * @param v
	 * @param A
	 * @return v - A
	 */
	public static Matrix minus(double v, Matrix A) {
		return uminus(A).plus(v);
	}
	
	/**
	 * Set all the elements of X to be those of Y, this is particularly
	 * useful when we want to change elements of the object referred by X 
	 * rather than the reference X itself.
	 *  
	 * @param X a matrix to be set
	 * 
	 * @param Y a matrix to set X
	 * 
	 */
	public static void setMatrix(Matrix X, Matrix Y) {
		assign(X, Y);
	}
	
	/**
	 * Unary minus.
	 * 
	 * @param A a matrix
	 * 
	 * @return -A
	 */
	public static Matrix uminus(Matrix A) {
		if (A == null) {
			return null;
		} else {
			return A.times(-1);
		}
	}
	
	public static DenseVector full(Vector V) {
		if (V instanceof SparseVector) {
			int dim = V.getDim();
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			double[] values = ArrayOperator.allocateVector(dim, 0);
			for (int k = 0; k < ((SparseVector) V).getNNZ(); k++) {
				values[ir[k]] = pr[k];
			}
			return new DenseVector(values);
		} else {
			return (DenseVector) V;
		}
	}
	
	public static SparseVector sparse(Vector V) {
		if (V instanceof DenseVector) {
			double[] values = ((DenseVector) V).getPr();
			TreeMap<Integer, Double> map = new TreeMap<Integer, Double>();
			for (int k = 0; k < values.length; k++) {
				if (values[k] != 0) {
					map.put(k, values[k]);
				}
			}
			int nnz = map.size();
			int[] ir = new int[nnz];
			double[] pr = new double[nnz];
			int dim = values.length;
			int ind = 0;
			for (Entry<Integer, Double> entry : map.entrySet()) {
				ir[ind] = entry.getKey();
				pr[ind] = entry.getValue();
				ind++;
			}
			return new SparseVector(ir, pr, nnz, dim);
		} else {
			return (SparseVector) V;
		}
	}
	
	public static DenseMatrix full(Matrix S) {
		if (S instanceof SparseMatrix) {
			int M = S.getRowDimension();
			int N = S.getColumnDimension();
			double[][] data = new double[M][];
			int[] ic = ((SparseMatrix) S).getIc();
			int[] jr = ((SparseMatrix) S).getJr();
			int[] valCSRIndices = ((SparseMatrix) S).getValCSRIndices();
			double[] pr = ((SparseMatrix) S).getPr();
			for (int i = 0; i < M; i++) {
				double[] rowData = ArrayOperator.allocateVector(N, 0);
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					rowData[ic[k]] = pr[valCSRIndices[k]];
				}
				data[i] = rowData;
			}
			return new DenseMatrix(data);
		} else {
			return (DenseMatrix) S;
		}
	}
	
	public static SparseMatrix sparse(Matrix A) {
		if (A instanceof DenseMatrix) {
			int rIdx = 0;
			int cIdx = 0;		
			int nzmax = 0;
			double value = 0;
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
			int numRows = A.getRowDimension();
			int numColumns = A.getColumnDimension();
			double[][] data = ((DenseMatrix) A).getData();
			for (int j = 0; j < numColumns; j++) {
				cIdx = j;
				for (int i = 0; i < numRows; i++) {
					rIdx = i;
					value = data[i][j];
					if (value != 0) {
						map.put(Pair.of(cIdx, rIdx), value);
						nzmax++;
					}
				}
			}
			int[] ir = new int[nzmax];
			int[] jc = new int[numColumns + 1];
			double[] pr = new double[nzmax];
			int k = 0;
			jc[0] = 0;
			int currentColumn = 0;
			for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
				rIdx = entry.getKey().second;
				cIdx = entry.getKey().first;
				pr[k] = entry.getValue();
				ir[k] = rIdx;
				if (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
			while (currentColumn < numColumns) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			jc[numColumns] = k;
			return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		} else {
			return (SparseMatrix) A;
		}
	}
	
	public static Vector[] sparseMatrix2SparseRowVectors(Matrix S) {
		if (!(S instanceof SparseMatrix)) {
			System.err.println("SparseMatrix input is expected.");
			System.exit(1);
		}
		int M = S.getRowDimension();
		int N = S.getColumnDimension();
		Vector[] Vs = new Vector[M];
		int[] ic = ((SparseMatrix) S).getIc();
		int[] jr = ((SparseMatrix) S).getJr();
		double[] pr = ((SparseMatrix) S).getPr();
		int[] valCSRIndices = ((SparseMatrix) S).getValCSRIndices();
		int[] indices = null;
		double[] values = null;
		int nnz = 0;
		int dim = N;
		for (int r = 0; r < M; r++) {
			nnz = jr[r + 1] - jr[r];
			indices = new int[nnz];
			values = new double[nnz];
			int idx = 0;
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				indices[idx] = ic[k];
				values[idx] = pr[valCSRIndices[k]];
				idx++;
			}
			Vs[r] = new SparseVector(indices, values, nnz, dim);
		}
		return Vs;
	}
	
	public static Vector[] sparseMatrix2SparseColumnVectors(Matrix S) {
		if (!(S instanceof SparseMatrix)) {
			System.err.println("SparseMatrix input is expected.");
			System.exit(1);
		}
		int M = S.getRowDimension();
		int N = S.getColumnDimension();
		Vector[] Vs = new Vector[N];
		int[] ir = ((SparseMatrix) S).getIr();
		int[] jc = ((SparseMatrix) S).getJc();
		double[] pr = ((SparseMatrix) S).getPr();
		int[] indices = null;
		double[] values = null;
		int nnz = 0;
		int dim = M;
		for (int c = 0; c < N; c++) {
			nnz = jc[c + 1] - jc[c];
			indices = new int[nnz];
			values = new double[nnz];
			int idx = 0;
			for (int k = jc[c]; k < jc[c + 1]; k++) {
				indices[idx] = ir[k];
				values[idx] = pr[k];
				idx++;
			}
			Vs[c] = new SparseVector(indices, values, nnz, dim);
		}
		return Vs;
	}
	
	public static Matrix sparseRowVectors2SparseMatrix(Vector[] Vs) {
		
		// return sparseColumnVectors2SparseMatrix(Vs).transpose();
		
		int nnz = 0;
		int numColumns = Vs.length > 0 ? Vs[0].getDim() : 0;
		for (int i = 0; i < Vs.length; i++) {
			if (!(Vs[i] instanceof SparseVector)) {
				fprintf("Vs[%d] should be a sparse vector.%n", i);
				System.exit(1);
			}
			nnz += ((SparseVector) Vs[i]).getNNZ();
			if (numColumns != Vs[i].getDim()) {
				fprintf("Vs[%d]'s dimension doesn't match.%n", i);
				System.exit(1);
			}
		}
		
		int numRows = Vs.length;
		int nzmax = nnz;
		int[] ic = new int[nzmax];
		int[] jr = new int[numRows + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int cnt = 0;
		jr[0] = 0;
		int currentRow = 0;
		for (int i = 0; i < numRows; i++) {
			int[] indices = ((SparseVector) Vs[i]).getIr();
			double[] values = ((SparseVector) Vs[i]).getPr();
			nnz = ((SparseVector) Vs[i]).getNNZ();
			for (int k = 0; k < nnz; k++) {
				cIdx = indices[k];
				rIdx = i;
				pr[cnt] = values[k];
				ic[cnt] = cIdx;
				while (currentRow < rIdx) {
					jr[currentRow + 1] = cnt;
					currentRow++;
				}
				cnt++;
			}
		}
		while (currentRow < numRows) {
			jr[currentRow + 1] = cnt;
			currentRow++;
		}
		// jr[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSRArrays(ic, jr, pr, numRows, numColumns, nzmax);
		
	}
	
	public static Matrix sparseColumnVectors2SparseMatrix(Vector[] Vs) {
		
		int nnz = 0;
		int numRows = Vs.length > 0 ? Vs[0].getDim() : 0;
		for (int i = 0; i < Vs.length; i++) {
			if (!(Vs[i] instanceof SparseVector)) {
				fprintf("Vs[%d] should be a sparse vector.%n", i);
				System.exit(1);
			}
			nnz += ((SparseVector) Vs[i]).getNNZ();
			if (numRows != Vs[i].getDim()) {
				fprintf("Vs[%d]'s dimension doesn't match.%n", i);
				System.exit(1);
			}
		}
		
		int numColumns = Vs.length;
		int nzmax = nnz;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (int c = 0; c < numColumns; c++) {
			int[] indices = ((SparseVector) Vs[c]).getIr();
			double[] values = ((SparseVector) Vs[c]).getPr();
			nnz = ((SparseVector) Vs[c]).getNNZ();
			for (int r = 0; r < nnz; r++) {
				rIdx = indices[r];
				cIdx = c;
				pr[k] = values[r];
				ir[k] = rIdx;
				while (currentColumn < cIdx) {
					jc[currentColumn + 1] = k;
					currentColumn++;
				}
				k++;
			}
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		// jc[numColumns] = k;

		return SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		
	}
	
}
