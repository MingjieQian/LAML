package la.decomposition;

import static la.utils.Printer.fprintf;
import static la.utils.Printer.printMatrix;
import static la.utils.Printer.printVector;
import static la.utils.Matlab.full;
import static la.utils.Matlab.sparse;
import static la.utils.Matlab.sparseMatrix2SparseRowVectors;
import static la.utils.Matlab.sparseRowVectors2SparseMatrix;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import la.utils.ArrayOperator;

/***
 * A Java implementation of LU decomposition using 
 * Gaussian elimination with row pivoting.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 7th, 2013
 */
public class LUDecomposition {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = new double[][] {
				{1, -2, 3},
				{2, -5, 12},
				{0, 2, -10}
		};
		Matrix A = new DenseMatrix(data);
		fprintf("A:\n");
		printMatrix(A);
		
		Matrix[] LUP = LUDecomposition.decompose(A);
		Matrix L = LUP[0];
		Matrix U = LUP[1];
		Matrix P = LUP[2];
		
		fprintf("L:\n");
		printMatrix(L);
		
		fprintf("U:\n");
		printMatrix(U);
		
		fprintf("P:\n");
		printMatrix(P);
		
		fprintf("PA:\n");
		printMatrix(P.mtimes(A));
		
		fprintf("LU:\n");
		printMatrix(L.mtimes(U));
		
		long start = 0;
		start = System.currentTimeMillis();
		
		LUDecomposition LUDecomp = new LUDecomposition(A);
		Vector b = new DenseVector(new double[] {2, 3, 4});
		Vector x = LUDecomp.solve(b);
		fprintf("Solution for Ax = b:\n");
		printVector(x);
		fprintf("b = \n");
		printVector(b);
		fprintf("Ax = \n");
		printVector(A.operate(x));
		
		fprintf("A^{-1}:\n");
		printMatrix(LUDecomp.inverse());
		
		fprintf("det(A) = %.2f\n", LUDecomp.det());
		System.out.format("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000F);
		fprintf("**********************************\n");
		
		A = sparse(A);
		fprintf("A:\n");
		printMatrix(A);
		
		LUP = LUDecomposition.decompose(A);
		L = LUP[0];
		U = LUP[1];
		P = LUP[2];
		
		fprintf("L:\n");
		printMatrix(L);
		
		fprintf("U:\n");
		printMatrix(U);
		
		fprintf("P:\n");
		printMatrix(P);
		
		fprintf("PA:\n");
		printMatrix(P.mtimes(A));
		
		fprintf("LU:\n");
		printMatrix(L.mtimes(U));
		
		start = System.currentTimeMillis();
		
		LUDecomp = new LUDecomposition(sparse(A));
		b = new DenseVector(new double[] {2, 3, 4});
		x = LUDecomp.solve(b);
		fprintf("Solution for Ax = b:\n");
		printVector(x);
		fprintf("Ax = \n");
		printVector(A.operate(x));
		fprintf("b = \n");
		printVector(b);
		
		Matrix B = new DenseMatrix(new double[][] {
				{2, 4},
				{3, 3}, 
				{4, 2} }
				);
		Matrix X = LUDecomp.solve(B);
		fprintf("Solution for AX = B:\n");
		printMatrix(X);
		fprintf("AX = \n");
		printMatrix(A.mtimes(X));
		fprintf("B = \n");
		printMatrix(B);
		
		fprintf("A^{-1}:\n");
		printMatrix(LUDecomp.inverse());
		
		fprintf("det(A) = %.2f\n", LUDecomp.det());
		System.out.format("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000F);
		
	}
	
	/**
	 * The lower triangular matrix in the LU decomposition such that
	 * PA = LU.
	 */
	private Matrix L;
	
	/**
	 * The upper triangular matrix in the LU decomposition such that
	 * PA = LU.
	 */
	private Matrix U;
	
	/**
	 * The row permutation matrix in the LU decomposition such that
	 * PA = LU.
	 */
	private Matrix P;
	
	/**
	 * Number of row exchanges.
	 */
	private int numRowExchange;
	
	/**
	 * Get the lower triangular matrix in the LU decomposition.
	 * 
	 * @return L
	 */
	public Matrix getL() {
		return L;
	}
	
	/**
	 * Get the upper triangular matrix in the LU decomposition.
	 * 
	 * @return U
	 */
	public Matrix getU() {
		return U;
	}
	
	/**
	 * Get the row permutation matrix in the LU decomposition.
	 * 
	 * @return P
	 */
	public Matrix getP() {
		return P;
	}
	
	/**
	 * Construct an LUDecomposition instance from a square real matrix.
	 * 
	 * @param A a dense or sparse real square matrix
	 */
	public LUDecomposition(Matrix A) {
		Matrix[] LUP = run(A);
		L = LUP[0];
		U = LUP[1];
		P = LUP[2];
	}
	
	/**
	 * Do LU decomposition from a square real matrix.
	 * 
	 * @param A a dense or sparse real square matrix
	 * 
	 * @return a {@code Matrix} array [L, U, P]
	 */
	private Matrix[] run(Matrix A) {
		/*if (L == null || U == null || P == null) {
			Matrix[] LUP = run(A);
			L = LUP[0];
			U = LUP[1];
			P = LUP[2];
		}*/
		int n = A.getRowDimension();
		if (n != A.getColumnDimension()) {
			System.err.println("A should be a square matrix.");
			System.exit(1);
		}
		numRowExchange = 0;
		Matrix[] LUP = new Matrix[3];
		
		/*
		 * LU = PA
		 */
		
		if (A instanceof DenseMatrix) {
			double[][] L = new DenseMatrix(n, n, 0).getData();
			double[][] AData = ((DenseMatrix) A.copy()).getData();
			double[][] P = new DenseMatrix(n, n, 0).getData();
			// Initialize P = eye(n)
			for (int i = 0; i < n; i++) {
				P[i][i] = 1;
			}
			for (int i = 0; i < n; i++) {
				// j = argmax_{k \in {i, i+1, ..., n}} |A_{ki}|
				double maxVal = AData[i][i];
				int j = i;
				for (int k = i + 1; k < n; k++) {
					if (maxVal < AData[k][i]) {
						j = k;
						maxVal = AData[k][i];
					}
				}
				if (maxVal == 0) {
					System.err.println("Matrix A is singular.");
					LUP[0] = null;
					LUP[1] = null;
					LUP[2] = null;
					return LUP;
				}
				if (j != i) {
					// Swap rows i and j of A, L, and P
					// A: Swap columns from i to n - 1
					// swap(AData[i], AData[j], i, n);
					double[] temp = null;
					temp = AData[i];
					AData[i] = AData[j];
					AData[j] = temp;
					// L: Swap columns from 0 to i
					// swap(L[i], L[j], 0, i + 1);
					temp = L[i];
					L[i] = L[j];
					L[j] = temp;
					// P: Swap non-zero columns
					/*double[] P_i = P[i];
					double[] P_j = P[j];
					for (int k = 0; k < n; k++) {
						if (P_i[k] == 1) {
							P_i[k] = 0;
							P_j[k] = 1;
						} else if (P_j[k] == 1) {
							P_i[k] = 1;
							P_j[k] = 0;
						}
					}*/
					// swap(P[i], P[j], 0, n);
					temp = P[i];
					P[i] = P[j];
					P[j] = temp;
					numRowExchange++;
				}
				// Elimination step
				L[i][i] = 1;
				double[] A_i = AData[i];
				double L_ki = 0;
				for (int k = i + 1; k < n; k++) {
					// L[k][i] = AData[k][i] / AData[i][i];
					L_ki = AData[k][i] / maxVal;
					L[k][i] = L_ki;
					double[] A_k = AData[k];
					// AData[k][i] = 0;
					A_k[i] = 0;
					for (int l = i + 1; l < n; l++) {
						// AData[k][l] -= L[k][i] * AData[i][l];
						// AData[k][l] -= L_ki * A_i[l];
						A_k[l] -= L_ki * A_i[l];
					}
				}
			}
			LUP[0] = new DenseMatrix(L);
			LUP[1] = new DenseMatrix(AData);
			LUP[2] = new DenseMatrix(P);
		} else if (A instanceof SparseMatrix) {
			/*double[][] L = new DenseMatrix(n, n, 0).getData();
			double[][] AData = ((DenseMatrix) A.copy()).getData();
			double[][] P = new DenseMatrix(n, n, 0).getData();*/
			
			Vector[] LVs = sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
			Vector[] AVs = sparseMatrix2SparseRowVectors(A);
			Vector[] PVs = sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
			
			// Initialize P = eye(n)
			for (int i = 0; i < n; i++) {
				PVs[i].set(i, 1);
			}
			for (int i = 0; i < n; i++) {
				// j = argmax_{k \in {i, i+1, ..., n}} |A_{ki}|
				double maxVal = AVs[i].get(i);
				int j = i;
				for (int k = i + 1; k < n; k++) {
					double v = AVs[k].get(i);
					if (maxVal < v) {
						j = k;
						maxVal = v;
					}
				}
				if (maxVal == 0) {
					System.err.println("Matrix A is singular.");
					LUP[0] = null;
					LUP[1] = null;
					LUP[2] = null;
					return LUP;
				}
				if (j != i) {
					// Swap rows i and j of A, L, and P
					// A: Swap columns from i to n - 1
					// swap(AData[i], AData[j], i, n);
					Vector temp = null;
					temp = AVs[i];
					AVs[i] = AVs[j];
					AVs[j] = temp;
					// L: Swap columns from 0 to i
					// swap(L[i], L[j], 0, i + 1);
					temp = LVs[i];
					LVs[i] = LVs[j];
					LVs[j] = temp;
					// P: Swap non-zero columns
					/*double[] P_i = P[i];
					double[] P_j = P[j];
					for (int k = 0; k < n; k++) {
						if (P_i[k] == 1) {
							P_i[k] = 0;
							P_j[k] = 1;
						} else if (P_j[k] == 1) {
							P_i[k] = 1;
							P_j[k] = 0;
						}
					}*/
					// swap(P[i], P[j], 0, n);
					temp = PVs[i];
					PVs[i] = PVs[j];
					PVs[j] = temp;
					numRowExchange++;
				}
				// Elimination step
				LVs[i].set(i, 1);
				Vector A_i = AVs[i];
				double L_ki = 0;
				for (int k = i + 1; k < n; k++) {
					// L[k][i] = AData[k][i] / AData[i][i];
					L_ki = AVs[k].get(i) / maxVal;
					// L[k][i] = L_ki;
					LVs[k].set(i, L_ki);
					Vector A_k = AVs[k];
					// AData[k][i] = 0;
					A_k.set(i, 0);
					for (int l = i + 1; l < n; l++) {
						// AData[k][l] -= L[k][i] * AData[i][l];
						// AData[k][l] -= L_ki * A_i[l];
						A_k.set(l, A_k.get(l) - L_ki * A_i.get(l));
					}
				}
			}
			LUP[0] = sparseRowVectors2SparseMatrix(LVs);
			LUP[1] = sparseRowVectors2SparseMatrix(AVs);
			LUP[2] = sparseRowVectors2SparseMatrix(PVs);
		}
		return LUP;
	}
	
	/**
	 * Do LU decomposition from a square real matrix.
	 * 
	 * @param A a dense or sparse real square matrix
	 * 
	 * @return a {@code Matrix} array [L, U, P]
	 */
	public static Matrix[] decompose(Matrix A) {
		int n = A.getRowDimension();
		if (n != A.getColumnDimension()) {
			System.err.println("A should be a square matrix.");
			System.exit(1);
		}
		Matrix[] LUP = new Matrix[3];
		
		/*
		 * LU = PA
		 */
		
		if (A instanceof DenseMatrix) {
			double[][] L = new DenseMatrix(n, n, 0).getData();
			double[][] AData = ((DenseMatrix) A.copy()).getData();
			double[][] P = new DenseMatrix(n, n, 0).getData();
			// Initialize P = eye(n)
			for (int i = 0; i < n; i++) {
				P[i][i] = 1;
			}
			for (int i = 0; i < n; i++) {
				// j = argmax_{k \in {i, i+1, ..., n}} |A_{ki}|
				double maxVal = AData[i][i];
				int j = i;
				for (int k = i + 1; k < n; k++) {
					if (maxVal < AData[k][i]) {
						j = k;
						maxVal = AData[k][i];
					}
				}
				if (maxVal == 0) {
					System.err.println("Matrix A is singular.");
					LUP[0] = null;
					LUP[1] = null;
					LUP[2] = null;
					return LUP;
				}
				if (j != i) {
					// Swap rows i and j of A, L, and P
					// A: Swap columns from i to n - 1
					// swap(AData[i], AData[j], i, n);
					double[] temp = null;
					temp = AData[i];
					AData[i] = AData[j];
					AData[j] = temp;
					// L: Swap columns from 0 to i
					// swap(L[i], L[j], 0, i + 1);
					temp = L[i];
					L[i] = L[j];
					L[j] = temp;
					// P: Swap non-zero columns
					/*double[] P_i = P[i];
					double[] P_j = P[j];
					for (int k = 0; k < n; k++) {
						if (P_i[k] == 1) {
							P_i[k] = 0;
							P_j[k] = 1;
						} else if (P_j[k] == 1) {
							P_i[k] = 1;
							P_j[k] = 0;
						}
					}*/
					// swap(P[i], P[j], 0, n);
					temp = P[i];
					P[i] = P[j];
					P[j] = temp;
				}
				// Elimination step
				L[i][i] = 1;
				double[] A_i = AData[i];
				double L_ki = 0;
				/*for (int k = i + 1; k < n; k++) {
					// L[k][i] = AData[k][i] / AData[i][i];
					L_ki = AData[k][i] / maxVal;
					L[k][i] = L_ki;
					AData[k][i] = 0;
					for (int l = i + 1; l < n; l++) {
						// AData[k][l] -= L[k][i] * AData[i][l];
						AData[k][l] -= L_ki * A_i[l];
					}
				}*/
				for (int k = i + 1; k < n; k++) {
					// L[k][i] = AData[k][i] / AData[i][i];
					L_ki = AData[k][i] / maxVal;
					L[k][i] = L_ki;
					double[] A_k = AData[k];
					// AData[k][i] = 0;
					A_k[i] = 0;
					for (int l = i + 1; l < n; l++) {
						// AData[k][l] -= L[k][i] * AData[i][l];
						// AData[k][l] -= L_ki * A_i[l];
						A_k[l] -= L_ki * A_i[l];
					}
				}
			}
			LUP[0] = new DenseMatrix(L);
			LUP[1] = new DenseMatrix(AData);
			LUP[2] = new DenseMatrix(P);
		} else if (A instanceof SparseMatrix) {
			/*double[][] L = new DenseMatrix(n, n, 0).getData();
			double[][] AData = ((DenseMatrix) A.copy()).getData();
			double[][] P = new DenseMatrix(n, n, 0).getData();*/
			
			Vector[] LVs = sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
			Vector[] AVs = sparseMatrix2SparseRowVectors(A);
			Vector[] PVs = sparseMatrix2SparseRowVectors(new SparseMatrix(n, n));
			
			// Initialize P = eye(n)
			for (int i = 0; i < n; i++) {
				PVs[i].set(i, 1);
			}
			for (int i = 0; i < n; i++) {
				// j = argmax_{k \in {i, i+1, ..., n}} |A_{ki}|
				double maxVal = AVs[i].get(i);
				int j = i;
				for (int k = i + 1; k < n; k++) {
					double v = AVs[k].get(i);
					if (maxVal < v) {
						j = k;
						maxVal = v;
					}
				}
				if (maxVal == 0) {
					System.err.println("Matrix A is singular.");
					LUP[0] = null;
					LUP[1] = null;
					LUP[2] = null;
					return LUP;
				}
				if (j != i) {
					// Swap rows i and j of A, L, and P
					// A: Swap columns from i to n - 1
					// swap(AData[i], AData[j], i, n);
					Vector temp = null;
					temp = AVs[i];
					AVs[i] = AVs[j];
					AVs[j] = temp;
					// L: Swap columns from 0 to i
					// swap(L[i], L[j], 0, i + 1);
					temp = LVs[i];
					LVs[i] = LVs[j];
					LVs[j] = temp;
					// P: Swap non-zero columns
					/*double[] P_i = P[i];
					double[] P_j = P[j];
					for (int k = 0; k < n; k++) {
						if (P_i[k] == 1) {
							P_i[k] = 0;
							P_j[k] = 1;
						} else if (P_j[k] == 1) {
							P_i[k] = 1;
							P_j[k] = 0;
						}
					}*/
					// swap(P[i], P[j], 0, n);
					temp = PVs[i];
					PVs[i] = PVs[j];
					PVs[j] = temp;
				}
				// Elimination step
				LVs[i].set(i, 1);
				Vector A_i = AVs[i];
				double L_ki = 0;
				for (int k = i + 1; k < n; k++) {
					// L[k][i] = AData[k][i] / AData[i][i];
					L_ki = AVs[k].get(i) / maxVal;
					// L[k][i] = L_ki;
					LVs[k].set(i, L_ki);
					Vector A_k = AVs[k];
					// AData[k][i] = 0;
					A_k.set(i, 0);
					for (int l = i + 1; l < n; l++) {
						// AData[k][l] -= L[k][i] * AData[i][l];
						// AData[k][l] -= L_ki * A_i[l];
						A_k.set(l, A_k.get(l) - L_ki * A_i.get(l));
					}
				}
			}
			LUP[0] = sparseRowVectors2SparseMatrix(LVs);
			LUP[1] = sparseRowVectors2SparseMatrix(AVs);
			LUP[2] = sparseRowVectors2SparseMatrix(PVs);
		}
		return LUP;
	}
	
	@SuppressWarnings("unused")
	private static void swap(double[] V1, double[] V2, int start, int end) {
		double temp = 0;
		for (int i = start; i < end; i++) {
			temp = V1[i];
			V1[i] = V2[i];
			V2[i] = temp;
		}
	}
	
	/**
	 * Solve the system of linear equations Ax = b for a real 
	 * square matrix A.
	 * 
	 * @param b a 1D {@code double} array
	 * 
	 * @return x s.t. Ax = b
	 */
	public Vector solve(double[] b) {
		return solve(new DenseVector(b));
	}
	
	/**
	 * Solve the system of linear equations Ax = b for a real 
	 * square matrix A.
	 * 
	 * @param b a dense or sparse real vector
	 * 
	 * @return x s.t. Ax = b
	 */
	public Vector solve(Vector b) {
		/*if (L == null || U == null || P == null) {
			Matrix[] LUP = run(A);
			L = LUP[0];
			U = LUP[1];
			P = LUP[2];
		}*/
		Vector res = null;
		if (L instanceof DenseMatrix) {
			// PAx = Pb = d
			// LUx = d
			double[] d = ((DenseVector) full(P.operate(b))).getPr();
			// Ly = d
			int n = L.getColumnDimension();
			double[][] LData = ((DenseMatrix) full(L)).getData();
			double[] LData_i = null;
			double[] y = new double[n];
			double v = 0;
			for (int i = 0; i < n; i++) {
				v = d[i];
				LData_i = LData[i];
				for (int k = 0; k < i; k++) {
					v -= LData_i[k] * y[k];
				}
				y[i] = v;
			}
			// Ux = y
			double[][] UData = ((DenseMatrix) full(U)).getData();
			double[] UData_i = null;
			double[] x = new double[n];
			v = 0;
			for (int i = n - 1; i > -1; i--) {
				UData_i = UData[i];
				v = y[i];
				for (int k = n - 1; k > i; k--) {
					v -= UData_i[k] * x[k];
				}
				x[i] = v / UData_i[i];
			}
			res = new DenseVector(x);
		} else if (L instanceof SparseMatrix){
			// PAx = Pb = d
			// LUx = d
			double[] d = ((DenseVector) full(P.operate(b))).getPr();
			// Ly = d
			int n = L.getColumnDimension();
			// double[][] LData = ((DenseMatrix) full(L)).getData();
			Vector[] LVs = sparseMatrix2SparseRowVectors(L);
			Vector LRow_i = null;
			double[] y = new double[n];
			double v = 0;
			for (int i = 0; i < n; i++) {
				v = d[i];
				LRow_i = LVs[i];
				int[] ir = ((SparseVector) LRow_i).getIr();
				double[] pr = ((SparseVector) LRow_i).getPr();
				int nnz = ((SparseVector) LRow_i).getNNZ();
				int idx = -1;
				for (int k = 0; k < nnz; k++) {
					idx = ir[k];
					if (idx >= i) {
						break;
					}
					v -= pr[k] * y[idx];
				}
				/*for (int k = 0; k < i; k++) {
					v -= LRow_i[k] * y[k];
				}*/
				y[i] = v;
			}
			// Ux = y
			// double[][] UData = ((DenseMatrix) full(U)).getData();
			Vector[] UVs = sparseMatrix2SparseRowVectors(U);
			Vector URow_i = null;
			double[] x = new double[n];
			v = 0;
			for (int i = n - 1; i > -1; i--) {
				URow_i = UVs[i];
				v = y[i];
				int[] ir = ((SparseVector) URow_i).getIr();
				double[] pr = ((SparseVector) URow_i).getPr();
				int nnz = ((SparseVector) URow_i).getNNZ();
				int idx = -1;
				int k = nnz - 1;
				while (true) {
					idx = ir[k];
					if (idx <= i) {
						break;
					}
					v -= pr[k] * x[idx];
					k--;
				}
				/*for (int k = n - 1; k > i; k--) {
					v -= URow_i[k] * x[k];
				}*/
				x[i] = v / URow_i.get(i);
			}
			res = new DenseVector(x);
		}
		return res;
	}
	
	/**
	 * Solve the system of linear equations AX = B for a real 
	 * square matrix A.
	 * 
	 * @param B a 2D {@code double} array
	 * 
	 * @return X s.t. AX = B
	 */
	public Matrix solve(double[][] B) {
		return solve(new DenseMatrix(B));
	}
	
	/**
	 * Solve the system of linear equations AX = B for a real 
	 * square matrix A.
	 * 
	 * @param B a dense or sparse real matrix
	 * 
	 * @return X s.t. AX = B
	 */
	public Matrix solve(Matrix B) {
		Matrix res = null;
		if (L instanceof DenseMatrix) {
			// PAX = PB = D
			// LUX = D
			double[][] D = full(P.mtimes(B)).getData();
			double[] DRow_i = null;
			// LY = D
			int n = L.getColumnDimension();
			double[][] LData = full(L).getData();
			double[] LRow_i = null;
			double[][] Y = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0);
			double[] YRow_i = null;

			double v = 0;
			for (int i = 0; i < n; i++) {
				LRow_i = LData[i];
				DRow_i = D[i];
				YRow_i = Y[i];
				for (int j = 0; j < B.getColumnDimension(); j++) {
					v = DRow_i[j];
					for (int k = 0; k < i; k++) {
						v -= LRow_i[k] * Y[k][j];
					}
					YRow_i[j] = v;
				}
			}
			// UX = Y
			double[][] UData = full(U).getData();
			double[] URow_i = null;
			double[][] X = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0);
			double[] XRow_i = null;
			for (int i = n - 1; i > -1; i--) {
				URow_i = UData[i];
				YRow_i = Y[i];
				XRow_i = X[i];
				for (int j = 0; j < B.getColumnDimension(); j++) {
					v = YRow_i[j];
					for (int k = n - 1; k > i; k--) {
						v -= URow_i[k] * X[k][j];
					}
					XRow_i[j] = v / URow_i[i];
				}
			}
			res = new DenseMatrix(X);
		} else if (L instanceof SparseMatrix){
			// PAX = PB = D
			// LUX = D
			double[][] D = full(P.mtimes(B)).getData();
			double[] DRow_i = null;
			// LY = D
			int n = L.getColumnDimension();
			Vector[] LVs = sparseMatrix2SparseRowVectors(L);
			Vector LRow_i = null;
			double[][] Y = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0);
			double[] YRow_i = null;
			
			double v = 0;
			for (int i = 0; i < n; i++) {
				LRow_i = LVs[i];
				int[] ir = ((SparseVector) LRow_i).getIr();
				double[] pr = ((SparseVector) LRow_i).getPr();
				int nnz = ((SparseVector) LRow_i).getNNZ();
				int idx = -1;
				DRow_i = D[i];
				YRow_i = Y[i];
				for (int j = 0; j < B.getColumnDimension(); j++) {
					v = DRow_i[j];
					for (int k = 0; k < nnz; k++) {
						idx = ir[k];
						if (idx >= i) {
							break;
						}
						v -= pr[k] * Y[idx][j];
					}
					/*for (int k = 0; k < i; k++) {
						v -= LRow_i[k] * y[k];
					}*/
					YRow_i[j] = v;
				}
			}
			// UX = Y
			Vector[] UVs = sparseMatrix2SparseRowVectors(U);
			Vector URow_i = null;
			double[][] X = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0);
			double[] XRow_i = null;
			
			for (int i = n - 1; i > -1; i--) {
				URow_i = UVs[i];
				int[] ir = ((SparseVector) URow_i).getIr();
				double[] pr = ((SparseVector) URow_i).getPr();
				int nnz = ((SparseVector) URow_i).getNNZ();
				int idx = -1;
				YRow_i = Y[i];
				XRow_i = X[i];
				for (int j = 0; j < B.getColumnDimension(); j++) {
					v = YRow_i[j];
					int k = nnz - 1;
					while (true) {
						idx = ir[k];
						if (idx <= i) {
							break;
						}
						v -= pr[k] * X[idx][j];
						k--;
					}
					/*for (int k = n - 1; k > i; k--) {
						v -= URow_i[k] * x[k];
					}*/
					XRow_i[j] = v / URow_i.get(i);
				}
			}
			res = new DenseMatrix(X);
		}
		return res;
	}
	
	/**
	 * Compute the inverse of this real square matrix.
	 * 
	 * @return this<sup>-1</sup>
	 */
	public Matrix inverse() {
		if (U == null) {
			return null;
		}
		/*
		 * When computing A^{-1}, A * A^{-1} = I.
		 * Thus we have A^{-1} is the solutions for A * X = [e_1, e_2, ..., e_n].
		 */
		int n = L.getColumnDimension();
		double[][] AInverseTransposeData = new double[n][];
		double[][] eye = new double[n][];
		for (int i = 0; i < n; i++) {
			eye[i] = ArrayOperator.allocateVector(n, 0);
			eye[i][i] = 1;
		}
		for (int i = 0; i < n; i++) {
			AInverseTransposeData[i] = full(solve(eye[i])).getPr();
		}
		return new DenseMatrix(AInverseTransposeData).transpose();
	}
	
	/**
	 * Compute the determinant of this real square matrix.
	 * 
	 * @return det(this)
	 */
	public double det() {
		if (U == null) {
			return 0;
		}
		double s = 1;
		for (int k = 0; k < U.getColumnDimension(); k++) {
			s *= U.getEntry(k, k);
			if (s == 0) {
				break;
			}
		}
		return numRowExchange % 2 == 0 ? s : -s;
	}

}
