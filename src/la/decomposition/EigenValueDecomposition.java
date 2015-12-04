package la.decomposition;

import static java.lang.Math.abs;
import static ml.utils.Matlab.eye;
import static ml.utils.Matlab.full;
import static ml.utils.Matlab.hilb;
import static ml.utils.Printer.disp;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printMatrix;

import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import ml.utils.ArrayOperator;
import ml.utils.Pair;

/***
 * A Java implementation for the eigenvalue decomposition
 * of a symmetric real matrix.
 * <p/>
 * The input matrix is first reduced to tridiagonal
 * matrix and then is diagonalized by implicit symmetric 
 * shifted QR algorithm.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 19th, 2013
 */
public class EigenValueDecomposition {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 4;
		int n = 4;
		Matrix A = hilb(m, n);
		/*A = new DenseMatrix(new double[][] {
				{1, 3, 4},
				{3, 2, 8},
				{4, 8, 3} });
		
		A = IO.loadMatrix("D.txt");
		A = A.transpose().mtimes(A);*/
		fprintf("A:\n");
		disp(A);
		long start = 0;
		start = System.currentTimeMillis();
		Matrix[] VD = EigenValueDecomposition.decompose(A);
		System.out.format("Elapsed time: %.4f seconds.\n", (System.currentTimeMillis() - start) / 1000.0);
		fprintf("*****************************************\n");
		
		Matrix V = VD[0];
		Matrix D = VD[1];
		
		fprintf("V:\n");
		printMatrix(V);
		
		fprintf("D:\n");
		printMatrix(D);
		
		fprintf("VDV':\n");
		disp(V.mtimes(D).mtimes(V.transpose()));

		fprintf("A:\n");
		printMatrix(A);

		fprintf("V'V:\n");
		printMatrix(V.transpose().mtimes((V)));
		
	}
	
	/**
	 * Tolerance.
	 */
	public static double tol = 1e-16;
	
	/**
	 * Maximum number of iterations.
	 */
	public static int maxIter;
	
	/**
	 * Eigenvectors.
	 */
	private Matrix V;
	
	/**
	 * A sparse diagonal matrix D with its diagonal being all
	 * eigenvalues in decreasing order (absolute value).
	 */
	private Matrix D;
	
	/**
	 * Get eigenvectors.
	 * 
	 * @return V
	 */
	public Matrix getV() {
		return V;
	}
	
	/**
	 * Get a diagonal matrix containing the eigenvalues 
	 * in decreasing order.
	 * 
	 * @return D
	 */
	public Matrix getD() {
		return D;
	}
	
	/**
	 * Construct this eigenvalue decomposition instance 
	 * from a real symmetric matrix.
	 * 
	 * @param A a real symmetric matrix
	 */
	public EigenValueDecomposition(Matrix A) {
		Matrix[] VD = decompose(A, true);
		V = VD[0];
		D = VD[1];
	}
	
	/**
	 * Construct this eigenvalue decomposition instance 
	 * from a real symmetric matrix.
	 * 
	 * @param A a real symmetric matrix
	 * 
	 * @param tol tolerance
	 */
	public EigenValueDecomposition(Matrix A, double tol) {
		EigenValueDecomposition.tol = tol;
		Matrix[] VD = decompose(A, true);
		V = VD[0];
		D = VD[1];
	}
	
	/**
	 * Construct this eigenvalue decomposition instance 
	 * from a real symmetric matrix.
	 * 
	 * @param A a real symmetric matrix
	 * 
	 * @param computeV if V is to be computed
	 */
	public EigenValueDecomposition(Matrix A, boolean computeV) {
		Matrix[] VD = decompose(A, computeV);
		V = VD[0];
		D = VD[1];
	}
	
	/**
	 * Do eigenvalue decomposition for a real symmetric matrix,
	 * i.e. AV = VD.
	 * 
	 * @param A a real symmetric matrix
	 * 
	 * @return a {@code Matrix} array [V, D]
	 */
	public static Matrix[] decompose(Matrix A) {
		return decompose(A, true);
	}

	/**
	 * Do eigenvalue decomposition for a real symmetric matrix,
	 * i.e. AV = VD.
	 * 
	 * @param A a real symmetric matrix
	 * 
	 * @param computeV if V is to be computed
	 * 
	 * @return a {@code Matrix} array [V, D]
	 */
	public static Matrix[] decompose(Matrix A, boolean computeV) {
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		if (m != n) {
			System.err.println("Input should be a square matrix.");
			System.exit(1);
		}
		maxIter = 30 * n * n;
		// A = QTQ' <=> Q'AQ = T
		Matrix[] QT = tridiagonalize(A, computeV);
		Matrix Q = QT[0];
		Matrix T = QT[1];
		
		// IO.saveMatrix(full(T), "T.txt");
		
		/*disp(A);
		printMatrix(Q);
		printMatrix(T);*/
		
		// T = VDV' <=> V'TV = D
		Matrix[] VD = diagonalizeTD(T, computeV);
		Matrix V = VD[0];
		Matrix D = VD[1];
		/*fprintf("VDV':\n");
		disp(V.mtimes(D).mtimes(V.transpose()));*/
		
		// A = QTQ' = QVDV'Q' = (QV)D(QV)'
		Matrix[] res = new Matrix[2];
		res[0] = computeV ? Q.mtimes(V) : null;
		res[1] = D;
		return res;
	}
	
	/**
	 * Only eigenvalues of a symmetric real matrix are computed.
	 * 
	 * @param A a symmetric real matrix
	 * 
	 * @return a 1D {@code double} array containing the eigenvalues
	 * 				in decreasing order (absolute value)
	 */
	public static double[] computeEigenvalues(Matrix A) {
		SparseMatrix S = (SparseMatrix) decompose(A, false)[1];
		int m = S.getRowDimension();
		int n = S.getColumnDimension();
		int len = m >= n ? n : m;
		double[] s = ArrayOperator.allocateVector(len, 0);
		for (int i = 0; i < len; i++) {
			s[i] = S.getEntry(i, i);
		}
		return s;
	}

	/**
	 * Tridiagonalize a real symmetric matrix A, i.e. A = Q * T * Q' 
	 * such that Q is an orthogonal matrix and T is a tridiagonal matrix.
	 * <p>
	 * A = QTQ' <=> Q'AQ = T
	 * 
	 * @param A a real symmetric matrix
	 * 
	 * @return a {@code Matrix} array [Q, T]
	 */
	@SuppressWarnings("unused")
	private static Matrix[] tridiagonalize(Matrix A) {
		return tridiagonalize(A, true);
	}
	
	/**
	 * Tridiagonalize a real symmetric matrix A, i.e. A = Q * T * Q' 
	 * such that Q is an orthogonal matrix and T is a tridiagonal matrix.
	 * <p>
	 * A = QTQ' <=> Q'AQ = T
	 * 
	 * @param A a real symmetric matrix
	 * 
	 * @param computeV if V is to be computed
	 * 
	 * @return a {@code Matrix} array [Q, T]
	 */
	private static Matrix[] tridiagonalize(Matrix A, boolean computeV) {
		A = full(A).copy();
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		Matrix[] QT = new Matrix[2];
		double[] a = ArrayOperator.allocateVector(n, 0);
		double[] b = ArrayOperator.allocateVector(n, 0);
		double[][] AData = ((DenseMatrix) A).getData();
		double c = 0;
		double s = 0;
		double r = 0;
		for (int j = 0; j < n - 2; j++) {
			a[j] = AData[j][j];
			// Householder transformation on columns of A(j+1:m, j+1:n)
			// Compute the norm of A(j+1:m, j)
			c = 0;
			for (int i = j + 1; i < m; i++) {
				c += Math.pow(AData[i][j], 2);
			}
			if (c == 0)
				continue;
			s = Math.sqrt(c);
			b[j] = AData[j + 1][j] > 0 ? -s : s;
			r = Math.sqrt(s * (s + abs(AData[j + 1][j])));
			
			/*double[] u1 = ArrayOperation.allocate1DArray(n, 0);
			for (int k = j + 1; k < m; k++) {
				u1[k] = AData[k][j];
			}
			u1[j + 1] -= b[j];
			for (int k = j + 1; k < m; k++) {
				u1[k] /= r;
			}
			Matrix H = eye(n).minus(new DenseMatrix(u1, 1).mtimes(new DenseMatrix(u1, 2)));
			disp(new DenseMatrix(u1, 1));
			disp(eye(n));
			disp(H);
			disp(A);
			disp(H.mtimes(A).mtimes(H));
			disp(A);*/
			
			AData[j + 1][j] -= b[j];
			for (int k = j + 1; k < m; k++) {
				AData[k][j] /= r;
			}
			
			double[] w = new double[n - j - 1];
			double[] u = new double[n - j - 1];
			double[] v = new double[n - j - 1];
			
			for (int i = j + 1, t = 0; i < m; i++, t++) {
				u[t] = AData[i][j];
			}
			
			// v = B33 * u
			for (int i = j + 1, t = 0; i < m; i++, t++) {
				double[] ARow_i = AData[i];
				s = 0;
				for (int k = j + 1, l = 0; k < n; k++, l++) {
					s += ARow_i[k] * u[l];
				}
				v[t] = s;
			}
			
			c = ArrayOperator.innerProduct(u, v) / 2;
			for (int i = j + 1, t = 0; i < m; i++, t++) {
				w[t] = v[t] - c * u[t];
			}
			
			/*disp(w);
			Matrix B33 = new DenseMatrix(n - j - 1, n - j - 1, 0);
			for (int i = j + 1, t = 0; i < m; i++, t++) {
				double[] ARow_i = AData[i];
				for (int k = j + 1, l = 0; k < n; k++, l++) {
					B33.setEntry(t, l, ARow_i[k]);
				}
			}
			disp(B33);
			Matrix U = new DenseMatrix(u, 1);
			disp(U.transpose().mtimes(U));
			disp(U);
			Matrix W = B33.mtimes(U).minus(U);
			disp(W);
			disp(B33.minus(plus(U.mtimes(W.transpose()), W.mtimes(U.transpose()))));
			Matrix Hk = eye(n - j - 1).minus(U.mtimes(U.transpose()));
			disp(Hk);
			disp(Hk.mtimes(B33).mtimes(Hk));*/
			for (int i = j + 1, t = 0; i < m; i++, t++) {
				double[] ARow_i = AData[i];
				for (int k = j + 1, l = 0; k < n; k++, l++) {
					ARow_i[k] = ARow_i[k] - (u[t] * w[l] + w[t] * u[l]);
				}
			}
			// disp(A);
			
			/*fprintf("Householder transformation on n - 1 columns:\n");
			disp(A);*/
			// disp(A);
			// Householder transformation on rows of A(j:m, j+1:n)
			
		}
		a[n - 2] = AData[n - 2][n - 2];
		a[n - 1] = AData[n - 1][n - 1];
		b[n - 2] = AData[n - 1][n - 2];
		QT = unpack(A, a, b, computeV);
		return QT;
	}

	/**
	 * Unpack Q and T from the result of tridiagonalization.
	 * 
	 * @param A tridiagonalization result
	 * 
	 * @param a diagonal
	 * 
	 * @param b superdiagonal
	 * 
	 * @param computeV if V is to be computed
	 * 
	 * @return a {@code Matrix} array [Q, T]
	 */
	private static Matrix[] unpack(Matrix A, double[] a, double[] b, boolean computeV) {
		Matrix[] QT = new Matrix[3];
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		DenseMatrix Q = null;
		if (computeV) {
			Q = new DenseMatrix(m, m, 0);
			double[][] QData = Q.getData();
			double s = 0;
			double[] y = null;
			for (int i = 0; i < m; i++) {
				// Compute U^T * e_i
				y = QData[i];
				y[i] = 1;
				for (int j = 0; j < n - 2; j++) {
					s = 0;
					for (int k = j + 1; k < m; k++) {
						s += A.getEntry(k, j) * y[k];
					}
					for (int k = j + 1; k < m; k++) {
						y[k] -= A.getEntry(k, j) * s;
					}
				}
			}
			/*fprintf("Q:\n");
			disp(Q);*/
		}
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (int i = 0; i < m; i++) {
			if (i < n)
				map.put(Pair.of(i, i), a[i]);
			if (i < n - 1) {
				map.put(Pair.of(i, i + 1), b[i]);
				map.put(Pair.of(i + 1, i), b[i]);
			}
		}
		Matrix T = SparseMatrix.createSparseMatrix(map, m, n);
		/*fprintf("T:\n");
		printMatrix(T);
		T = new SparseMatrix(m, n);
		for (int i = 0; i < m; i++) {
			if (i < n)
				T.setEntry(i, i, a[i]);
			if (i < n - 1) {
				T.setEntry(i, i + 1, b[i]);
				T.setEntry(i + 1, i, b[i]);
			}
		}*/
		if (computeV) {
			/*fprintf("T:\n");
			printMatrix(T);*/

			/*fprintf("A:\n");
			printMatrix(A);

			fprintf("Q'Q:\n");
			disp(Q.transpose().mtimes(Q));*/

			/*fprintf("QTQ':\n");
			disp(Q.mtimes(T).mtimes(Q.transpose()));*/
		}

		QT[0] = Q;
		QT[1] = T;
		return QT;
	}

	/**
	 * Do eigenvalue decomposition for a real symmetric tridiagonal 
	 * matrix, i.e. T = VDV'.
	 * 
	 * @param T a real symmetric tridiagonal matrix
	 * 
	 * @return a {@code Matrix} array [V, D]
	 */
	@SuppressWarnings("unused")
	private static Matrix[] diagonalizeTD(Matrix T) {
		return diagonalizeTD(T, true);
	}
	
	/**
	 * Do eigenvalue decomposition for a real symmetric tridiagonal 
	 * matrix, i.e. T = VDV'.
	 * 
	 * @param T a real symmetric tridiagonal matrix
	 * 
	 * @param computeV  if V is to be computed
	 * 
	 * @return a {@code Matrix} array [V, D]
	 */
	private static Matrix[] diagonalizeTD(Matrix T, boolean computeV) {
		
		int m = T.getRowDimension();
		int n = T.getColumnDimension();
		int len = m >= n ? n : m;
		int idx = 0;
		
		/*
		 * The tridiagonal matrix T is
		 * s[0] e[0]
		 * e[0] s[1] e[1]
		 *      e[1] ...
		 *               s[len - 2] e[len - 2]
		 *               e[len - 2] s[len - 1]
		 */
		double[] s = ArrayOperator.allocateVector(len, 0);
		double[] e = ArrayOperator.allocateVector(len, 0);
		
		for (int i = 0; i < len - 1; i++) {
			s[i] = T.getEntry(i, i);
			e[i] = T.getEntry(i, i + 1);
		}
		s[len - 1] = T.getEntry(len - 1, len - 1);
		
		// V': each row of V' is a right singular vector
		double[][] Vt = null;
		if (computeV)
			Vt = eye(n, n).getData();
		
		double[] mu = ArrayOperator.allocate1DArray(len, 0);
		
		/*
		 * T0 = ITI', V = I
		 * T = IT0I' = VT0G1G1'Vt = VT0G1(G1'Vt)
		 *   = VG1G1'T0G1(G1'Vt) = (VG1)(G1'T0G1)(G1'Vt) 
		 *   = (VG1)T1(G1'Vt)
		 *   ...
		 *   = (Gn-1...G2G1)D(G1G2...Gn-1)'
		 *   
		 * where G = |cs  sn|
		 *           |-sn cs|
		 * G' = |cs -sn|
		 *      |sn  cs|
		 */
		
		/*
		 * Find B_hat, i.e. the bottommost unreduced submatrix of B.
		 * Index starts from 0.
		 */
		// *********************************************************
		
		int i_start = 0;
		int i_end = len - 1;
		// int cnt_shifted = 0;
		int ind = 1;
		while (true) {
		
			idx = len - 2;
			while (idx >= 0) {
				if (e[idx] == 0) {
					idx--;
				} else {
					break;
				}
			}
			i_end = idx + 1;
			// Now idx = -1 or e[idx] != 0
			// If idx = -1, then e[0] = 0, i_start = i_end = 0, e = 0
			// Else if e[idx] != 0, then e[i] = 0 for i_end = idx + 1 <= i <= len - 1  
			while (idx >= 0) {
				if (e[idx] != 0) {
					idx--;
				} else {
					break;
				}
			}
			i_start = idx + 1;
			// Now idx = -1 or e[idx] = 0
			// If idx = -1 i_start = 0
			// Else if e[idx] = 0, then e[idx + 1] != 0, e[i_end - 1] != 0
			// i.e. e[i] != 0 for i_start <= i <= i_end - 1

			if (i_start == i_end) {
				break;
			}
			
			// Apply the stopping criterion to B_hat
			// If any e[i] is set to zero, return to loop
			
			boolean set2Zero = false;
			mu[i_start] = abs(s[i_start]);
			for (int j = i_start; j < i_end; j++) {
				mu[j + 1] = abs(s[j + 1]) * mu[j] / (mu[j] + abs(e[j]));
				if (abs(e[j]) <= mu[j] * tol) {
					e[j] = 0;
					set2Zero = true;
				}
			}
			if (set2Zero) {
				continue;
			}
			
			implicitSymmetricShiftedQR(s, e, Vt, i_start, i_end, computeV);
			// cnt_shifted++;
			
			if (ind == maxIter) {
				break;
			}
			
			ind++;
			
		}
		
		// fprintf("cnt_shifted: %d\n", cnt_shifted);
		// *********************************************************
		
		// Quick sort eigenvalues and eigenvectors
		quickSort(s, Vt, 0, len - 1, "descend", computeV);
		
		Matrix[] VD = new Matrix[2];
		VD[0] = computeV ? new DenseMatrix(Vt).transpose() : null;
		VD[1] = buildD(s, m, n);
		
		/*disp("T:");
		printMatrix(T);
		disp("VDV':");
		Matrix V = VD[0];
		Matrix D = VD[1];
		disp(V.mtimes(D).mtimes(V.transpose()));*/

		return VD;
		
	}
	
	/**
	 * Sort the eigenvalues in a specified order. If computeV is true, 
	 * eigenvectors will also be sorted.
	 * 
	 * @param s a 1D {@code double} array containing the eigenvalues
	 * 
	 * @param Vt eigenvectors
	 * 
	 * @param start start index (inclusive)
	 * 
	 * @param end end index (inclusive)
	 * 
	 * @param order a {@code String} either "descend" or "ascend"
	 * 
	 * @param computeV if V is to be computed
	 */
	private static void quickSort(double[] s, double[][] Vt, int start, int end, String order, boolean computeV) {

		int	i,j;
		double temp;
		i = start;
		j = end;
		temp = s[i];
		// double[] tempU = computeUV ? Ut[i] : null;
		double[] tempV = computeV ? Vt[i] : null;
		do{
			if (order.equals("ascend")) {
				while((abs(s[j]) > abs(temp)) && (j > i))
					j--;
			} else if (order.equals("descend")) {
				while((abs(s[j]) < abs(temp)) && (j > i))
					j--;
			} 
			if(j > i){
				s[i] = s[j];
				if (computeV) {
					// Ut[i] = Ut[j];
					Vt[i] = Vt[j];
				}
				i++;
			}
			if (order.equals("ascend")) {
				while((abs(s[i]) < abs(temp)) && (j > i))
					i++;
			} else if (order.equals("descend")) {
				while((abs(s[i]) > abs(temp)) && (j > i))
					i++;
			}
			if(j > i){
				s[j] = s[i];
				if (computeV) {
					// Ut[j] = Ut[i];
					Vt[j] = Vt[i];
				}
				j--;
			}
		} while(i != j);
		s[i] = temp;
		if (computeV) {
			// Ut[i] = tempU;
			Vt[i] = tempV;
		}
		i++;
		j--;
		if(start < j)
			quickSort(s, Vt, start, j, order, computeV);
		if(i < end)
			quickSort(s, Vt, i, end, order, computeV);
		
	}
	
	/**
	 * Build the diagonal matrix containing all eigenvalues.
	 * 
	 * @param s eigenvalues
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @return a diagonal matrix containing all eigenvalues
	 */
	private static Matrix buildD(double[] s, int m, int n) {
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (int i = 0; i < m; i++) {
			if (i < n)
				map.put(Pair.of(i, i), s[i]);
		}
		return SparseMatrix.createSparseMatrix(map, m, n);
	}

	/**
	 * Implicit symmetric shifted QR algorithm on B_hat which is
	 * the bottommost unreduced submatrix of B begin from
	 * i_start (inclusive) to i_end (inclusive).
	 * 
	 * @param s diagonal elements
	 * 
	 * @param e superdiagonal elements
	 * 
	 * @param Vt transposition of eigenvector matrix
	 * 
	 * @param i_start start index of B_hat (inclusive)
	 * 
	 * @param i_end end index of B_hat (inclusive)
	 * 
	 * @param computeV if V is to be computed
	 */
	private static void implicitSymmetricShiftedQR(double[] s, double[] e,
			double[][] Vt, int i_start, int i_end, boolean computeV) {
		
		/*
		 * B(i_start:i_end, i_start:i_end) is unreduced bidiagonal matrix
		 */
		double d = 0;
		
		d = (s[i_end - 1] - s[i_end]) / 2;
		double c = e[i_end - 1] * e[i_end - 1];
		double shift = Math.sqrt(d * d + c);
		shift = d > 0 ? shift : -shift;
		shift =  c / (d + shift);
		double f = s[i_start] - s[i_end] + shift;
		
		double g = e[i_start];
		double cs = 0, sn = 0, r = 0;
		double t, tt;
		double h = 0;
		
		for (int i = i_start; i < i_end; i++) {
			// ROT(f, g, cs, sn, r)
			if (f == 0) {
				cs = 0;
				sn = 1;
				r = g;
				/*sn = g > 0 ? 1 : -1;
				r = g > 0 ? g : -g;*/
			} else if (abs(f) > abs(g)) {
				t = g / f;
				tt = Math.sqrt(1 + t * t);
				/*if (f < 0) {
					tt = -tt;
				}*/
				cs = 1 / tt;
				sn = t * cs;
				r = f * tt;
			} else {
				t = f / g;
				tt = Math.sqrt(1 + t * t);
				/*if (g < 0) {
					tt = -tt;
				}*/
				sn = 1 / tt;
				cs = t * sn;
				r = g * tt;
			}
			// UPDATE(cs, sn, vi, vi+1)
			if (computeV)
				update(cs, sn, Vt[i], Vt[i + 1]);
			
			if (i != i_start) { // Note that i != i_start rather than i != 0!!!
				e[i - 1] = r;
			}
			
			/*Bk = buildB(s, e, m, n);
			fprintf("Bk:\n");
			printMatrix(Bk);*/
			
			// Givens rotation on column_i and column_i+1
			/*
			 * |f,   h   | = |s[i], e[i]  | * |cs -sn|
			 * |g, s[i+1]|   |e[i], s[i+1]|   |sn cs |
			 */
			f = cs * s[i] + sn * e[i];
			h = -sn * s[i] + cs * e[i];
			g = cs * e[i] + sn * s[i + 1];
			s[i + 1] = -sn * e[i] + cs * s[i + 1];
			
			// Givens rotation on row_i and row_i+1
			/*
			 * |cs sn | * |f,   h      0   | = |s[i], e[i]     h'  |
			 * |-sn cs|   |g, s[i+1] e[i+1]|   |e[i], s[i+1] e[i+1]|
			 */
			r = cs * f + sn * g;
			s[i] = r;
			e[i] = -sn * f + cs * g;
			
			s[i + 1] = -sn * h + cs * s[i + 1];
			h = sn * e[i + 1];
			e[i + 1] *= cs;
			
			if (i < i_end - 1) {
				f = e[i]; // f = T(i+1, i)
				g = h;    // g = T(i+2, i)
			}
			
			/*Bk = buildB(s, e, m, n);
			fprintf("Bk:\n");
			printMatrix(Bk);*/
			
		}
		// e[i_end - 1] = f;
		
	}
	
	/**
	 * Update two 1D {@code double} arrays V1 and V2 by Givens rotation
	 * parameterized by cs and sn, i.e.
	 * [V1 V2] * |cs -sn| or |cs  sn| * |V1'|
	 *           |sn  cs|    |-sn cs|   |V2'|
	 * @param cs cos(theta)
	 * 
	 * @param sn sin(theta)
	 * 
	 * @param V1 a 1D {@code double} arrays
	 * 
	 * @param V2 a 1D {@code double} arrays
	 */
	private static void update(double cs, double sn, double[] V1, double[] V2) {
		double temp;
		for (int i = 0; i < V1.length; i++) {
			temp = V1[i];
			V1[i] = cs * temp + sn * V2[i];
			V2[i] = -sn * temp + cs * V2[i];
		}
	}
	
}
