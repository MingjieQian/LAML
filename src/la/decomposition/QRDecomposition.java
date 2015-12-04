package la.decomposition;

import static ml.utils.Matlab.full;
import static ml.utils.Matlab.hilb;
import static ml.utils.Matlab.sparse;
import static ml.utils.Matlab.sparseColumnVectors2SparseMatrix;
import static ml.utils.Matlab.sparseMatrix2SparseColumnVectors;
import static ml.utils.Matlab.sparseMatrix2SparseRowVectors;
import static ml.utils.Matlab.sparseRowVectors2SparseMatrix;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Printer.printVector;

import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import ml.utils.ArrayOperator;
import ml.utils.Pair;

/***
 * A Java implementation of QR decomposition using 
 * Householder transformations with column pivoting.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 12th, 2013
 */
public class QRDecomposition {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 4;
		int n = 3;
		Matrix A = hilb(m, n);
		
		fprintf("When A is full:\n");
		
		fprintf("A:\n");
		printMatrix(A);
		
		long start = 0;
		start = System.currentTimeMillis();
		
		Matrix[] QRP = QRDecomposition.decompose(A);
		Matrix Q = QRP[0];
		Matrix R = QRP[1];
		Matrix P = QRP[2];
		
		fprintf("Q:\n");
		printMatrix(Q);
		
		fprintf("R:\n");
		printMatrix(R);
		
		fprintf("P:\n");
		printMatrix(P);
		
		fprintf("AP:\n");
		printMatrix(A.mtimes(P));
		
		fprintf("QR:\n");
		printMatrix(Q.mtimes(R));
		
		fprintf("Q'Q:\n");
		printMatrix(Q.transpose().mtimes(Q));
		
		fprintf("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000F);
		fprintf("**********************************\n");
		
		// fprintf("|AP - QR| = ");
		
		A = sparse(hilb(m, n));
		
		fprintf("When A is sparse:\n");
		
		fprintf("A:\n");
		printMatrix(A);
		
		start = System.currentTimeMillis();
		
		QRP = QRDecomposition.decompose(A);
		Q = QRP[0];
		R = QRP[1];
		P = QRP[2];
		
		fprintf("Q:\n");
		printMatrix(Q);
		
		fprintf("R:\n");
		printMatrix(R);
		
		fprintf("P:\n");
		printMatrix(P);
		
		fprintf("AP:\n");
		printMatrix(A.mtimes(P));
		
		fprintf("QR:\n");
		printMatrix(Q.mtimes(R));
		
		fprintf("Q'Q:\n");
		printMatrix(Q.transpose().mtimes(Q));
		
		fprintf("Elapsed time: %.2f seconds.\n", (System.currentTimeMillis() - start) / 1000F);
		
		QRDecomposition QRDecomp = new QRDecomposition((A));
		Vector b = new DenseVector(new double[] {2, 3, 4, 9});
		Vector x = QRDecomp.solve(b);
		fprintf("Solution for Ax = b:\n");
		printVector(x);
		fprintf("b = \n");
		printVector(b);
		fprintf("Ax = \n");
		printVector(A.operate(x));
		
	}

	/**
	 * The orthogonal matrix in the QR decomposition
	 * such that AP = QR.
	 */
	private Matrix Q;
	
	/**
	 * The upper triangular matrix in the QR decomposition
	 * such that AP = QR.
	 */
	private Matrix R;
	
	/**
	 * The column permutation matrix in the QR decomposition
	 * such that AP = QR.
	 */
	private Matrix P;
	
	/**
	 * Get the orthogonal matrix in the QR decomposition.
	 * 
	 * @return Q
	 */
	public Matrix getQ() {
		return Q;
	}
	
	/**
	 * Get the upper triangular matrix in the QR decomposition.
	 * 
	 * @return R
	 */
	public Matrix getR() {
		return R;
	}
	
	/**
	 * Get the column permutation matrix in the QR decomposition.
	 * 
	 * @return P
	 */
	public Matrix getP() {
		return P;
	}
	
	/**
	 * Construct a QRDecomposition instance from an arbitrary 
	 * real matrix A.
	 *  
	 * @param A a general dense or sparse real matrix
	 */
	public QRDecomposition(Matrix A) {
		Matrix[] QRP = run(A);
		Q = QRP[0];
		R = QRP[1];
		P = QRP[2];
	}
	
	/**
	 * Construct a QRDecomposition instance from an arbitrary 
	 * real matrix represented by a 2D {@code double} array.
	 *  
	 * @param A a 2D {@code double} array
	 */
	public QRDecomposition(double[][] A) {
		Matrix[] QRP = run(new DenseMatrix(A));
		Q = QRP[0];
		R = QRP[1];
		P = QRP[2];
	}
	
	/**
	 * Do QR decomposition for an arbitrary real matrix.
	 * 
	 * @param A a general dense or sparse real matrix
	 * 
	 * @return a {@code Matrix} array [Q, R, P]
	 */
	private Matrix[] run(Matrix A) {
		return decompose(A);
	}

	/**
	 * Solve the system of linear equations Ax = b in the least 
	 * square sense, i.e. X minimizes norm(Ax - b). The rank k 
	 * of A is determined from the QR decomposition with column 
	 * pivoting. The computed solution X has at most k nonzero 
	 * elements per column. If k < n, this is usually not the 
	 * same solution as x = pinv(A) * b, which returns a least 
	 * squares solution.
	 * 
	 * @param b a 1D {@code double} array
	 * 
	 * @return this \ b
	 */
	public Vector solve(double[] b) {
		return solve(new DenseVector(b));
	}

	/**
	 * Solve the system of linear equations Ax = b in the least 
	 * square sense, i.e. X minimizes norm(Ax - b). The rank k 
	 * of A is determined from the QR decomposition with column 
	 * pivoting. The computed solution X has at most k nonzero 
	 * elements per column. If k < n, this is usually not the 
	 * same solution as x = pinv(A) * b, which returns a least 
	 * squares solution.
	 * 
	 * @param b a dense or sparse real vector
	 * 
	 * @return this \ b
	 */
	public Vector solve(Vector b) {

		// AP = QR
		// Ax = b
		// APP'x = b
		// QRP'x = b
		// Ry = Q'b = d
		double[] d = full(Q.transpose().operate(b)).getPr();
		
		// y = R \ d
		int rank = 0;
		int m = R.getRowDimension();
		int n = R.getColumnDimension();
		int len = Math.min(m, n);
		for (int i = 0; i < len; i++) {
			if (R.getEntry(i, i) == 0) {
				rank = i;
				break;
			} else {
				rank++;
			}
		}
		double[] y = ArrayOperator.allocate1DArray(n, 0);
		
		if (R instanceof DenseMatrix) {
			double[][] RData = ((DenseMatrix) R).getData();
			double[] RData_i = null;
			double v = 0;
			for (int i = rank - 1; i > -1; i--) {
				RData_i = RData[i];
				v = d[i];
				for (int k = n - 1; k > i; k--) {
					v -= RData_i[k] * y[k];
				}
				y[i] = v / RData_i[i];
			}

		} else if (R instanceof SparseMatrix) {
			Vector[] RVs = sparseMatrix2SparseRowVectors(R);
			Vector RRow_i = null;
			double v = 0;
			for (int i = rank - 1; i > -1; i--) {
				RRow_i = RVs[i];
				v = d[i];
				int[] ir = ((SparseVector) RRow_i).getIr();
				double[] pr = ((SparseVector) RRow_i).getPr();
				int nnz = ((SparseVector) RRow_i).getNNZ();
				int idx = -1;
				int k = nnz - 1;
				while (true) {
					idx = ir[k];
					if (idx <= i) {
						break;
					}
					v -= pr[k] * y[idx];
					k--;
				}
				/*for (int k = n - 1; k > i; k--) {
					v -= URow_i[k] * x[k];
				}*/
				y[i] = v / RRow_i.get(i);
			}
		}

		// x = Py
		Vector x = P.operate(new DenseVector(y));
		return x;
	}
	
	/**
	 * Solve the system of linear equations AX = B in the least 
	 * square sense, i.e. X minimizes norm(AX - B). The rank k 
	 * of A is determined from the QR decomposition with column 
	 * pivoting. The computed solution X has at most k nonzero 
	 * elements per column. If k < n, this is usually not the 
	 * same solution as X = pinv(A) * B, which returns a least 
	 * squares solution.
	 * 
	 * @param B a 2D {@code double} array
	 * 
	 * @return this \ B
	 */
	public Matrix solve(double[][] B) {
		return solve(new DenseMatrix(B));
	}
	
	/**
	 * Solve the system of linear equations AX = B in the least 
	 * square sense, i.e. X minimizes norm(AX - B). The rank k 
	 * of A is determined from the QR decomposition with column 
	 * pivoting. The computed solution X has at most k nonzero 
	 * elements per column. If k < n, this is usually not the 
	 * same solution as X = pinv(A) * B, which returns a least 
	 * squares solution.
	 * 
	 * @param B a dense or sparse real matrix
	 * 
	 * @return this \ B
	 */
	public Matrix solve(Matrix B) {
		// AP = QR
		// Ax = B
		// APP'X = B
		// QRP'X = B
		// RY = Q'B = D
		double[][] D = full(Q.transpose().mtimes(B)).getData();
		double[] DRow_i = null;
		
		// Y = R \ D
		int rank = 0;
		int m = R.getRowDimension();
		int n = R.getColumnDimension();
		int len = Math.min(m, n);
		for (int i = 0; i < len; i++) {
			if (R.getEntry(i, i) == 0) {
				rank = i;
				break;
			} else {
				rank++;
			}
		}
		double[][] Y = ArrayOperator.allocate2DArray(n, B.getColumnDimension(), 0);
		double[] YRow_i = null;
		
		if (R instanceof DenseMatrix) {
			double[][] RData = ((DenseMatrix) R).getData();
			double[] RData_i = null;
			double v = 0;
			
			for (int i = rank - 1; i > -1; i--) {
				RData_i = RData[i];
				DRow_i = D[i];
				YRow_i = Y[i];
				for (int j = 0; j < B.getColumnDimension(); j++) {
					v = DRow_i[j];
					for (int k = n - 1; k > i; k--) {
						v -= RData_i[k] * Y[k][j];
					}
					YRow_i[j] = v / RData_i[i];
				}
			}
		} else if (R instanceof SparseMatrix) {
			Vector[] RVs = sparseMatrix2SparseRowVectors(R);
			Vector RRow_i = null;
			double v = 0;
			for (int i = rank - 1; i > -1; i--) {
				RRow_i = RVs[i];
				DRow_i = D[i];
				YRow_i = Y[i];
				for (int j = 0; j < B.getColumnDimension(); j++) {
					v = DRow_i[j];
					int[] ir = ((SparseVector) RRow_i).getIr();
					double[] pr = ((SparseVector) RRow_i).getPr();
					int nnz = ((SparseVector) RRow_i).getNNZ();
					int idx = -1;
					int k = nnz - 1;
					while (true) {
						idx = ir[k];
						if (idx <= i) {
							break;
						}
						v -= pr[k] * Y[idx][j];
						k--;
					}
					YRow_i[j] = v / RRow_i.get(i);
				}
			}
		}

		// X = PY
		Matrix X = P.mtimes(new DenseMatrix(Y));
		return X;
	}

	/**
	 * QR decomposition with column pivoting. That is, AP = QR.
	 * 
	 * @param A a dense or sparse real matrix
	 * 
	 * @return a {@code Matrix} array [Q, R, P]
	 */
	public static Matrix[] decompose(Matrix A) {
		A = A.copy();
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		Matrix[] QRP = new Matrix[3];
		double[] d = ArrayOperator.allocateVector(n, 0);
		Vector[] PVs = sparseMatrix2SparseColumnVectors(new SparseMatrix(n, n));
		// Initialize P = eye(n)
		for (int i = 0; i < n; i++) {
			PVs[i].set(i, 1);
		}
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double[] c = ArrayOperator.allocateVector(n, 0);
			/*for (int j = 0; j < n; j++) {
				double s = 0;
				for (int i = 0; i < m; i++) {
					s += Math.pow(AData[i][j], 2);
				}
				c[j] = s;
			}*/
			for (int j = 0; j < n; j++) {
				if (j >= m) {
					break;
				}
				for (int jj = j; jj < n; jj++) {
					double s = 0;
					for (int i = j; i < m; i++) {
						s += Math.pow(AData[i][jj], 2);
					}
					c[jj] = s;
				}
				int i = j;
				double maxVal = c[j];
				for (int k = j + 1; k < n; k++) {
					if (maxVal < c[k]) {
						i = k;
						maxVal = c[k];
					}
				}
				if (maxVal == 0) {
					System.out.println("Rank(A) < n.");
					QRP[0] = computeQ(A);
					QRP[1] = computeR(A, d);
					QRP[2] = sparseRowVectors2SparseMatrix(PVs);
					return QRP;
				}
				if (i != j) {
					// Swap A(:, i) <=> A(:, j)
					double temp = 0;
					for (int k = 0; k < m; k++) {
						temp = AData[k][i];
						AData[k][i] = AData[k][j];
						AData[k][j] = temp;
					}
					// Swap c[i] <=> c[j]
					temp = c[i];
					c[i] = c[j];
					c[j] = temp;
					// Swap P(:, i) <=> P(:, j)
					Vector V = PVs[i];
					PVs[i] = PVs[j];
					PVs[j] = V;
				}
				// Compute the norm of A(j:m, j), which is always sqrt(c[j])
				// since the array c will be updated
				double s = Math.sqrt(c[j]);
				d[j] = AData[j][j] > 0 ? -s : s;
				double r = Math.sqrt(s * (s + Math.abs(AData[j][j])));
				AData[j][j] -= d[j];
				for (int k = j; k < m; k++) {
					AData[k][j] /= r;
				}
				for (int k = j + 1; k < n; k++) {
					s = 0;
					for (int t = j; t < m; t++) {
						s += AData[t][j] * AData[t][k];
					}
					for (int t = j; t < m; t++) {
						AData[t][k] -= s * AData[t][j];
					}
					// c[k] -= Math.pow(AData[j][k], 2);
				}
				/*fprintf("Processed A, j = %d:\n", j);
				printMatrix(A);*/
			}
		} else if (A instanceof SparseMatrix) {
			Vector[] AVs = sparseMatrix2SparseColumnVectors(A);
			double[] c = ArrayOperator.allocateVector(n, 0);
			/*for (int j = 0; j < n; j++) {
				SparseVector A_j = (SparseVector) AVs[j];
				double[] pr = A_j.getPr();
				int nnz = A_j.getNNZ();
				double s = 0;
				for (int k = 0; k < nnz; k++) {
					s += Math.pow(pr[k], 2);
				}
				c[j] = s;
			}*/
			for (int j = 0; j < n; j++) {
				if (j >= m) {
					break;
				}
				for (int jj = j; jj < n; jj++) {
					SparseVector A_j = (SparseVector) AVs[jj];
					double[] pr = A_j.getPr();
					int[] ir = A_j.getIr();
					int nnz = A_j.getNNZ();
					double s = 0;
					int idx = -1;
					for (int k = 0; k < nnz; k++) {
						idx = ir[k];
						if (idx >= j)
							s += Math.pow(pr[k], 2);
					}
					c[jj] = s;
				}
				int i = j;
				double maxVal = c[j];
				for (int k = j + 1; k < n; k++) {
					if (maxVal < c[k]) {
						i = k;
						maxVal = c[k];
					}
				}
				if (maxVal == 0) {
					System.out.println("Rank(A) < n.");
					QRP[0] = computeQ(A);
					QRP[1] = computeR(A, d);
					QRP[2] = sparseRowVectors2SparseMatrix(PVs);
					return QRP;
				}
				if (i != j) {
					// Swap A(:, i) <=> A(:, j)
					double temp = 0;
					Vector V = null;
					V = AVs[i];
					AVs[i] = AVs[j];
					AVs[j] = V;
					// Swap c[i] <=> c[j]
					temp = c[i];
					c[i] = c[j];
					c[j] = temp;
					// Swap P(:, i) <=> P(:, j)
					V = PVs[i];
					PVs[i] = PVs[j];
					PVs[j] = V;
				}
				// Compute the norm of A(j:m, j), which is always sqrt(c[j])
				// since the array c will be updated
				double s = Math.sqrt(c[j]);
				SparseVector A_j = (SparseVector) AVs[j];
				double Ajj = A_j.get(j);
				d[j] = Ajj > 0 ? -s : s;
				double r = Math.sqrt(s * (s + Math.abs(Ajj)));
				A_j.set(j, Ajj - d[j]);
				// AData[j][j] -= d[j];
				int[] ir = A_j.getIr();
				double[] pr = A_j.getPr();
				int nnz = A_j.getNNZ();
				int idx = 0;
				for (int k = 0; k < nnz; k++) {
					idx = ir[k];
					if (idx < j)
						continue;
					else {
						pr[k] /= r;
					}
				}
				/*for (int k = j; k < m; k++) {
					AData[k][j] /= r;
				}*/
				for (int k = j + 1; k < n; k++) {
					SparseVector A_k = (SparseVector) AVs[k];
					s = 0;
					int[] ir2 = A_k.getIr();
					double[] pr2 = A_k.getPr();
					int nnz2 = A_k.getNNZ();
					if (nnz != 0 && nnz2 != 0) {
						int k1 = 0;
						int k2 = 0;
						int r1 = 0;
						int r2 = 0;
						double v = 0;
						while (k1 < nnz && k2 < nnz2) {
							r1 = ir[k1];
							r2 = ir2[k2];
							if (r1 < r2) {
								k1++;
							} else if (r1 == r2) {
								v = pr[k1] * pr2[k2];
								k1++;
								k2++;
								if (r1 >= j)
									s += v;
							} else {
								k2++;
							}
						}
					}
					/*s = 0;
					for (int t = j; t < m; t++) {
						s += AData[t][j] * AData[t][k];
					}*/
					for (int t = j; t < m; t++) {
						A_k.set(t, A_k.get(t) - s * A_j.get(t));
					}
					/*for (int t = j; t < m; t++) {
						AData[t][k] -= s * AData[t][j];
					}*/
					// c[k] -= Math.pow(A_k.get(j), 2);
				}
			}
			A = sparseColumnVectors2SparseMatrix(AVs);
		}
		
		/*disp("Processed A:");
		printMatrix(A);*/
		
		QRP[0] = computeQ(A);
		QRP[1] = computeR(A, d);
		QRP[2] = sparseColumnVectors2SparseMatrix(PVs);
		
		return QRP;
	}
	
	/**
	 * Compute Q from A, the QR algorithm with column pivoting result.
	 * 
	 * @param A QR algorithm with column pivoting result
	 * 
	 * @return Q
	 */
	private static Matrix computeQ(Matrix A) {
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		double[][] Q = new DenseMatrix(m, m, 0).getData();
		double s = 0;
		double[] y = null;
		for (int i = 0; i < m; i++) {
			// Compute Q^T * e_i
			y = Q[i];
			y[i] = 1;
			for (int j = 0; j < n; j++) {
				s = 0;
				for (int k = j; k < m; k++) {
					s += A.getEntry(k, j) * y[k];
				}
				for (int k = j; k < m; k++) {
					y[k] -= A.getEntry(k, j) * s;
				}
			}
		}
		return new DenseMatrix(Q);
	}
	
	/**
	 * Compute the upper triangular matrix from A, the QR algorithm 
	 * with column pivoting result, the diagonal elements are in the
	 * 1D {@code double} array d.
	 * 
	 * @param A QR algorithm with column pivoting result
	 * 
	 * @param d a 1D {@code double} array containing the diagonal 
	 * 		    elements of the upper triangular matrix
	 * 
	 * @return R
	 */
	private static Matrix computeR(Matrix A, double[] d) {
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		Matrix R = null;
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			for (int i = 0; i < m; i++) {
				double[] A_i = AData[i];
				if (i < n)
					A_i[i] = d[i];
				int len = Math.min(i, n);
				for (int j = 0; j < len; j++) {
					A_i[j] = 0;
				}
			}
			R = A;
		} else if (A instanceof SparseMatrix) {
			TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
			for (int i = 0; i < m; i++) {
				if (i < n)
					map.put(Pair.of(i, i), d[i]);
				for (int j = i + 1; j < n; j++) {
					map.put(Pair.of(i, j), A.getEntry(i, j));
				}
			}
			R = SparseMatrix.createSparseMatrix(map, m, n);
		}
		return R;
	}
	
}
