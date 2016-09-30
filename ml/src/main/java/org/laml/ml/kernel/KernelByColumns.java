package org.laml.ml.kernel;

import static org.laml.la.utils.ArrayOperator.allocate2DArray;
import static org.laml.la.utils.InPlaceOperator.expAssign;
import static org.laml.la.utils.InPlaceOperator.timesAssign;
import static org.laml.la.utils.Matlab.innerProduct;
import static org.laml.la.utils.Matlab.l2DistanceSquare;
import static org.laml.la.utils.Matlab.pow;
import static org.laml.la.utils.Matlab.sum;
import static org.laml.la.utils.Matlab.times;
import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;
import org.laml.la.vector.Vector;

/***
 * Java implementation of commonly used kernel functions.
 * 
 * @version 1.0 Jan. 27th, 2014
 * @author Mingjie Qian
 */
public class KernelByColumns {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}
	
	/**
	 * Computes Gram matrix of a specified kernel. Given a data matrix
	 * X (d x n), it returns Gram matrix K (n x n).
	 * 
	 * @param kernelType 'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam   --    | degree | sigma |    --
	 * 
	 * @param X a matrix

	 * @return Gram matrix (n x n)
	 * 
	 */
	public static Matrix calcKernel(String kernelType, 
			double kernelParam, Matrix X) {
		return calcKernel(kernelType, kernelParam, X, X);
	}
	
	/**
	 * Computes Gram matrix of a specified kernel. Given two sets of vectors
	 * A (n1 vectors) and B (n2 vectors), it returns Gram matrix K (n1 x n2).
	 * 
	 * @param kernelType 'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam   --    | degree | sigma |    --
	 * 
	 * @param A a 1D {@code Vector} array
	 * 
	 * @param B a 1D {@code Vector} array
	 * 
	 * @return Gram matrix (n1 x n2)
	 */
	public static Matrix calcKernel(String kernelType, 
			double kernelParam, Vector[] A, Vector[] B) {
		
		Matrix K = null;
		int nA = A.length;
		int nB = B.length;
		if (kernelType.equals("linear")) {
			double[][] resData = allocate2DArray(nA, nB, 0);
			double[] resRow = null;
			Vector V = null;
			for (int i = 0; i < nA; i++) {
				resRow = resData[i];
				V = A[i];
				for (int j = 0; j < nB; j++) {
					resRow[j] = innerProduct(V, B[j]);
				}
			}
			K = new DenseMatrix(resData);
			// K = A.transpose().mtimes(B);
		} else if (kernelType.equals("cosine")) {
			double[] AA = new double[nA];
			Vector V = null;
			for (int i = 0; i < nA; i++) {
				V = A[i];
				AA[i] = sum(V.times(V));
			}
			double[] BB = new double[nB];
			for (int i = 0; i < nB; i++) {
				V = B[i];
				BB[i] = sum(V.times(V));
			}
			double[][] resData = allocate2DArray(nA, nB, 0);
			double[] resRow = null;
			for (int i = 0; i < nA; i++) {
				resRow = resData[i];
				V = A[i];
				for (int j = 0; j < nB; j++) {
					resRow[j] = innerProduct(V, B[j]) / Math.sqrt(AA[i] * BB[j]);
				}
			}
			K = new DenseMatrix(resData);
			// K = dotMultiply(scalarDivide(1, sqrt(kron(AA.transpose(), BB))), AB);
		} else if (kernelType.equals("poly")) {
			double[][] resData = allocate2DArray(nA, nB, 0);
			double[] resRow = null;
			Vector V = null;
			for (int i = 0; i < nA; i++) {
				resRow = resData[i];
				V = A[i];
				for (int j = 0; j < nB; j++) {
					resRow[j] = Math.pow(innerProduct(V, B[j]), kernelParam);
				}
			}
			// K = pow(A.transpose().mtimes(B), kernelParam);
		} else if (kernelType.equals("rbf")) {
			K = l2DistanceSquare(A, B);
			timesAssign(K, -1 / (2 * Math.pow(kernelParam, 2)));
			expAssign(K);
			// K = exp(l2DistanceSquare(X1, X2).times(-1 / (2 * Math.pow(kernelParam, 2))));
		}
		return K;
		
	}

	/**
	 * Computes Gram matrix of a specified kernel. Given two data matrices
	 * X1 (d x n1), X2 (d x n2), it returns Gram matrix K (n1 x n2).
	 * 
	 * @param kernelType 'linear' | 'poly' | 'rbf' | 'cosine'
	 * 
	 * @param kernelParam   --    | degree | sigma |    --
	 * 
	 * @param X1 a matrix
	 * 
	 * @param X2 a matrix
	 * 
	 * @return Gram matrix (n1 x n2)
	 * 
	 */
	public static Matrix calcKernel(String kernelType, 
			double kernelParam, Matrix X1, Matrix X2) {
		
		Matrix K = null;
		if (kernelType.equals("linear")) {
			K = X1.transpose().mtimes(X2);
		} else if (kernelType.equals("cosine")) {
			Matrix A = X1;
			Matrix B = X2;
			double[] AA = sum(times(A, A)).getPr();
			double[] BB = sum(times(B, B)).getPr();
			Matrix AB = A.transpose().mtimes(B);
			int M = AB.getRowDimension();
			int N = AB.getColumnDimension();
			if (AB instanceof DenseMatrix) {
				double[][] resData = ((DenseMatrix) AB).getData();
				double[] resRow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] /= Math.sqrt(AA[i] * BB[j]);
					}
				}
			} else if (AB instanceof SparseMatrix) {
				double[] pr = ((SparseMatrix) AB).getPr();
				int[] ir = ((SparseMatrix) AB).getIr();
				int[] jc = ((SparseMatrix) AB).getJc();
				for (int j = 0; j < N; j++) {
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						pr[k] /= Math.sqrt(AA[ir[k]] * BB[j]);
					}
				}
			}
			K = AB;
			// K = dotMultiply(scalarDivide(1, sqrt(kron(AA.transpose(), BB))), AB);
		} else if (kernelType.equals("poly")) {
			K = pow(X1.transpose().mtimes(X2), kernelParam);
		} else if (kernelType.equals("rbf")) {
			K = l2DistanceSquare(X1, X2);
			timesAssign(K, -1 / (2 * Math.pow(kernelParam, 2)));
			expAssign(K);
			// K = exp(l2DistanceSquare(X1, X2).times(-1 / (2 * Math.pow(kernelParam, 2))));
		}
		return K;

	}

}
