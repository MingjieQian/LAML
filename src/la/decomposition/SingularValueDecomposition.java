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
 * A Java implementation for the singular value decomposition
 * of a general m-by-n matrix.
 * <p/>
 * The input matrix is first reduced to bidiagonal
 * matrix and then is diagonalized by hybrid of standard
 * implicit shifted QR algorithm and implicit zero-shift 
 * QR algorithm. All the singular values are computed to
 * high relative accuracy independent of their magnitudes.
 * 
 * @author Mingjie Qian
 * @version 1.0 Dec. 12th, 2013
 */
public class SingularValueDecomposition {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int m = 6;
		int n = 4;
		Matrix A = hilb(m, n);
		
		// A = new DenseMatrix(new double[][] { {1d, 2d}, {2d, 0d}, {1d, 7d}});
		
		/*A = new DenseMatrix(new double[][] {
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
				{10, 11, 12}
		});*/
		// printMatrix(SingularValueDecomposition.bidiagonalize(A)[1]);
		
		// A = IO.loadMatrix("SVDInput");
		
		/*fprintf("When A is full:\n\n");
		
		fprintf("A:\n");
		printMatrix(A);*/
		
		disp("A:");
		printMatrix(A);
		
		long start = 0;
		start = System.currentTimeMillis();
		
		boolean computeUV = !false;
		Matrix[] USV = SingularValueDecomposition.decompose(A, computeUV);
		
		System.out.format("Elapsed time: %.4f seconds.\n", (System.currentTimeMillis() - start) / 1000.0);
		fprintf("*****************************************\n");
		
		Matrix U = USV[0];
		Matrix S = USV[1];
		Matrix V = USV[2];
		
		if (computeUV) {
			fprintf("USV':\n");
			disp(U.mtimes(S).mtimes(V.transpose()));

			/*disp("US:");
			Matrix US = U.mtimes(S);
			disp(US);

			Matrix VT = V.transpose();
			disp("VT:");
			disp(VT);

			disp("USV':");
			disp(US.mtimes(VT));*/
			
			fprintf("A:\n");
			printMatrix(A);

			fprintf("U'U:\n");
			printMatrix(U.transpose().mtimes((U)));

			fprintf("V'V:\n");
			printMatrix(V.transpose().mtimes((V)));
			
			fprintf("U:\n");
			printMatrix(U);
			
			fprintf("V:\n");
			printMatrix(V);
		
		}
		
		fprintf("S:\n");
		printMatrix(S);
		
		fprintf("rank(A): %d\n", rank(A));

	}
	
	/**
	 * Tolerance.
	 */
	public static double tol = 1e-12;
	
	/**
	 * Maximum number of iterations.
	 */
	public static int maxIter;
	
	/**
	 * Left singular vectors.
	 */
	private Matrix U;
	
	/**
	 * A sparse matrix S with its diagonal being all
	 * singular values in decreasing order.
	 */
	private Matrix S;
	
	/**
	 * Right singular vectors.
	 */
	private Matrix V;
	
	/**
	 * Get the left singular vectors.
	 * 
	 * @return U
	 */
	public Matrix getU() {
		return U;
	}
	
	/**
	 * Get the matrix with its diagonal being all
	 * singular values in decreasing order.
	 * 
	 * @return S
	 */
	public Matrix getS() {
		return S;
	}
	
	/**
	 * Get the right singular vectors.
	 * 
	 * @return V
	 */
	public Matrix getV() {
		return V;
	}
	
	/**
	 * Construct this singular value decomposition instance 
	 * from a matrix.
	 * 
	 * @param A a real matrix
	 */
	public SingularValueDecomposition(Matrix A) {
		Matrix[] USV = decompose(A, true);
		U = USV[0];
		S = USV[1];
		V = USV[2];
	}
	
	/**
	 * Construct this singular value decomposition instance 
	 * from a matrix.
	 * 
	 * @param A a real matrix
	 * 
	 * @param computeUV if U and V are to be computed
	 */
	public SingularValueDecomposition(Matrix A, boolean computeUV) {
		Matrix[] USV = decompose(A, computeUV);
		U = USV[0];
		S = USV[1];
		V = USV[2];
	}

	/**
	 * Do singular value decompose for a general real matrix A, i.e.
	 * A = U * S * V'.
	 *  
	 * @param A an m x n real matrix
	 * 
	 * @return a {@code Matrix} array [U, S, V]
	 */
	public static Matrix[] decompose(Matrix A) {
		return decompose(A, true);
	}

	/**
	 * Do singular value decompose for a general real matrix A, i.e.
	 * A = U * S * V'.
	 *  
	 * @param A an m x n real matrix
	 * 
	 * @param computeUV if U and V are to be computed
	 * 
	 * @return a {@code Matrix} array [U, S, V]
	 */
	public static Matrix[] decompose(Matrix A, boolean computeUV) {
		
		// int m = A.getRowDimension();
		int n = A.getColumnDimension();
		maxIter = 3 * n * n;
				
		// A = U1BV1'
		Matrix[] UBV = bidiagonalize(A, computeUV);
		Matrix B = UBV[1];
		
		// B = U2SV2'
		Matrix[] USV = diagonalizeBD(B, computeUV);
		
		// A = U1BV1' = U1U2SV2'V1' = (U1U2)S(V1V2)'
		Matrix[] res = new Matrix[3];
		
		if (computeUV)
			res[0] = UBV[0].mtimes(USV[0]);
		else
			res[0] = null;
		res[1] = USV[1];
		if (computeUV)
			res[2] = UBV[2].mtimes(USV[2]);
		else
			res[2] = null;
		
		return res;
	}
	
	/**
	 * Only singular values of a matrix are computed.
	 * 
	 * @param A a matrix
	 * 
	 * @return a 1D {@code double} array containing the singular values
	 * 				in decreasing order
	 */
	public static double[] computeSingularValues(Matrix A) {
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
	 * Compute the rank of a matrix. The rank function provides 
	 * an estimate of the number of linearly independent rows or 
	 * columns of a matrix.
	 * 
	 * @param A a matrix
	 * 
	 * @return rank of the given matrix
	 */
	public static int rank(Matrix A) {
		int r = 0;
		double[] s = computeSingularValues(A);
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		double t = m >= n ? m * Math.pow(2, -52) : n * Math.pow(2, -52);
		for (int i = 0; i < s.length; i++) {
			/*double v = s[i];
			boolean b = v >= t;*/
			if (s[i] > t) {
				r++;
			}
		}
		return r;
	}
	
	/**
	 * Do singular value decompose for an m x n bidiagonal real matrix B, i.e.
	 * B = U * S * V'.
	 *  
	 * @param B an m x n bidiagonal real matrix
	 * 
	 * @return a {@code Matrix} array [U, S, V]
	 */
	@SuppressWarnings("unused")
	private static Matrix[] diagonalizeBD(Matrix B) {
		return diagonalizeBD(B, true);
	}
	
	/**
	 * Do singular value decompose for an m x n bidiagonal real matrix B, i.e.
	 * B = U * S * V'.
	 *  
	 * @param B an m x n bidiagonal real matrix
	 * 
	 * @param computeUV if U and V are to be computed
	 * 
	 * @return a {@code Matrix} array [U, S, V]
	 */
	private static Matrix[] diagonalizeBD(Matrix B, boolean computeUV) {
		
		int m = B.getRowDimension();
		int n = B.getColumnDimension();
		int len = m >= n ? n : m;
		int idx = 0;
		
		/*
		 * The bidiagonal matrix B is
		 * s[0] e[0]
		 *      s[1] e[1]
		 *           ...
		 *               s[len - 2] e[len - 2]
		 *                          s[len - 1]
		 */
		double[] s = ArrayOperator.allocateVector(len, 0);
		double[] e = ArrayOperator.allocateVector(len, 0);
		
		/*double[] pr = ((SparseMatrix) B).getPr();
		int nnz = ((SparseMatrix) B).getNNZ();
		int k = 0;
		while (true) {
			s[idx] = pr[k++];
			if (k == nnz)
				break;
			e[idx] = pr[k++];
			idx++;
		}*/
		for (int i = 0; i < len - 1; i++) {
			s[i] = B.getEntry(i, i);
			e[i] = B.getEntry(i, i + 1);
		}
		s[len - 1] = B.getEntry(len - 1, len - 1);
		
		/*
		 * B = USV' where U is the left singular vectors,
		 * and V is the right singular vectors.
		 */
		
		// U': each row of U' is a left singular vector
		double[][] Ut = null;
		if (computeUV)
			Ut = eye(m, m).getData();
		
		// V': each row of V' is a right singular vector
		double[][] Vt = null;
		if (computeUV)
			Vt = eye(n, n).getData();
		
		double[] mu = ArrayOperator.allocate1DArray(len, 0);
		
		double sigma_min = 0;
		double sigma_max = 0;
		
		/*
		 * B = IBI'
		 * Therefore, when pre-multiplying B by Givens rotation transform
		 * on i-th and j-th rows, we need to change the i-th and j-th rows
		 * of U'.
		 * B0 = IB0I' = UG'GBVt = (GUt)'BkVt
		 * (Ut)'BVt = (Ut)'BG'GVt = (Lt)'Bk(GVt)
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
		/*int cnt_zero_shift = 0;
		int cnt_shifted = 0;*/
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
			
			// Estimate the smallest singular value sigma_min and 
			// the largest singular value sigma_max of B_hat
			
			sigma_min = abs(mu[i_start]);
			for (int j = i_start; j <= i_end; j++) {
				if (sigma_min > abs(mu[j])) {
					sigma_min = abs(mu[j]);
				}
			}
			
			sigma_max = abs(s[i_start]);
			for (int j = i_start; j <= i_end; j++) {
				if (sigma_max < abs(s[j])) {
					sigma_max = abs(s[j]);
				}
			}
			for (int j = i_start; j < i_end; j++) {
				if (sigma_max < abs(e[j])) {
					sigma_max = abs(e[j]);
				}
			}
			
			// fprintf("Iter %d:\n", ind);
			
			if (n * sigma_max < sigma_min * Math.max(Double.MIN_VALUE / tol, 0.01)) {
				implicitZeroShiftQR(s, e, Ut, Vt, i_start, i_end, computeUV);
				// cnt_zero_shift++;
			} else {
				/*
				 * Standard shifted QR might converge very slowly and return
				 * wrong results, i.e., for the matrix saved in the file named
				 * "SVDInput"!!!
				 */
				// standardShiftedQR(s, e, Ut, Vt, i_start, i_end, computeUV);
				implicitZeroShiftQR(s, e, Ut, Vt, i_start, i_end, computeUV);
				// cnt_shifted++;
			}
			
			if (ind == maxIter) {
				break;
			}
			
			ind++;
			
		}
		/*fprintf("cnt_zero_shift: %d\n", cnt_zero_shift);
		fprintf("cnt_shifted: %d\n", cnt_shifted);*/
		// *********************************************************
		
		// Make sure that all elements of s are nonnegative
		for (int i = 0; i < len; i++) {
			if (s[i] < 0) {
				if (computeUV)
					ArrayOperator.timesAssign(Ut[i], -1);
				s[i] *= -1;
			}
		}
		
		// Quick sort singular values and singular vectors
		quickSort(s, Ut, Vt, 0, len - 1, "descend", computeUV);
		
		Matrix[] USV = new Matrix[3];
		if (computeUV)
			USV[0] = new DenseMatrix(Ut).transpose();
		else
			USV[0] = null;
		USV[1] = buildS(s, m, n);
		if (computeUV)
			USV[2] = new DenseMatrix(Vt).transpose();
		else
			USV[2] = null;
		return USV;

	}
	
	/**
	 * Sort the singular values in a specified order. If computeUV is true, 
	 * left and right singular vectors will also be sorted.
	 * 
	 * @param s a 1D {@code double} array containing the singular values
	 * 
	 * @param Ut left singular vectors
	 * 
	 * @param Vt right singular vectors
	 * 
	 * @param start start index (inclusive)
	 * 
	 * @param end end index (inclusive)
	 * 
	 * @param order a {@code String} either "descend" or "ascend"
	 * 
	 * @param computeUV if U and V are to be computed
	 */
	private static void quickSort(double[] s, double[][] Ut, double[][] Vt, int start, int end, String order, boolean computeUV) {

		int	i,j;
		double temp;
		i = start;
		j = end;
		temp = s[i];
		double[] tempU = computeUV ? Ut[i] : null;
		double[] tempV = computeUV ? Vt[i] : null;
		do{
			if (order.equals("ascend")) {
				while((s[j] > temp) && (j > i))
					j--;
			} else if (order.equals("descend")) {
				while((s[j] < temp) && (j > i))
					j--;
			} 
			if(j > i){
				s[i] = s[j];
				if (computeUV) {
					Ut[i] = Ut[j];
					Vt[i] = Vt[j];
				}
				i++;
			}
			if (order.equals("ascend")) {
				while((s[i] < temp) && (j > i))
					i++;
			} else if (order.equals("descend")) {
				while((s[i] > temp) && (j > i))
					i++;
			}
			if(j > i){
				s[j] = s[i];
				if (computeUV) {
					Ut[j] = Ut[i];
					Vt[j] = Vt[i];
				}
				j--;
			}
		} while(i != j);
		s[i] = temp;
		if (computeUV) {
			Ut[i] = tempU;
			Vt[i] = tempV;
		}
		i++;
		j--;
		if(start < j)
			quickSort(s, Ut, Vt, start, j, order, computeUV);
		if(i < end)
			quickSort(s, Ut, Vt, i, end, order, computeUV);
		
	}

	/**
	 * Standard implicit shifted QR algorithm on B_hat which is
	 * the bottommost unreduced submatrix of B begin from
	 * i_start (inclusive) to i_end (inclusive).
	 * 
	 * @param s diagonal elements
	 * 
	 * @param e superdiagonal elements
	 * 
	 * @param Ut left singular matrix
	 * 
	 * @param Vt right singular matrix
	 * 
	 * @param i_start start index of B_hat (inclusive)
	 * 
	 * @param i_end end index of B_hat (inclusive)
	 * 
	 * @param computeUV if U and V are to be computed
	 */
	@SuppressWarnings("unused")
	private static void standardShiftedQR(double[] s, double[] e,
			double[][] Ut, double[][] Vt, int i_start, int i_end, boolean computeUV) {
		// Temporarily we use implicit zero-shift QR
		/*int m = Ut.length;
		int n = Vt.length;
		int len = m >= n ? n : m;
		
		Matrix Bk = null;*/
		
		/*
		 * B(i_start:i_end, i_start:i_end) is unreduced bidiagonal matrix
		 */
		double d = 0;
		if (i_end >= 2)
			d = ((s[i_end - 1] + s[i_end]) * (s[i_end - 1] - s[i_end]) + 
				(e[i_end - 2] + e[i_end - 1]) * (e[i_end - 2] - e[i_end - 1])) / 2;
		else {
			d = ((s[i_end - 1] + s[i_end]) * (s[i_end - 1] - s[i_end]) -
				e[i_end - 1] * e[i_end - 1]) / 2;	
		}
		double c = s[i_end - 1] * e[i_end - 1];
		c = c * c;
		double shift = Math.sqrt(d * d + c);
		shift = d > 0 ? shift : -shift;
		shift =  c / (d + shift);
		double f = (s[i_start] + s[i_end]) * (s[i_start] - s[i_end]) - 
				e[i_end - 1] * e[i_end - 1] + shift;
		
		double g = s[i_start] * e[i_start];
		double cs = 0, sn = 0, r = 0;
		double t, tt;
		
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
			if (computeUV) {
				update(cs, sn, Vt[i], Vt[i + 1]);
			}
			
			if (i != i_start) { // Note that i != i_start rather than i != 0!!!
				e[i - 1] = r;
			}
			
			/*Bk = buildB(s, e, m, n);
			fprintf("Bk:\n");
			printMatrix(Bk);*/
			
			f = cs * s[i] + sn * e[i];
			e[i] = cs * e[i] - sn * s[i];
			g = sn * s[i + 1];
			s[i + 1] *= cs;
			
			// ROT(f, g, cs, sn, r)
			if (f == 0) {
				cs = 0;
				sn = 1;
				r = g;
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
			// UPDATE(cs, sn, Ui, Ui+1)
			if (computeUV) {
				update(cs, sn, Ut[i], Ut[i + 1]);
			}
			
			s[i] = r;
			f = cs * e[i] + sn * s[i + 1];
			s[i + 1] = -sn * e[i] + cs * s[i + 1];
			g = sn * e[i + 1];
			e[i + 1] *= cs;
			
			/*Bk = buildB(s, e, m, n);
			fprintf("Bk:\n");
			printMatrix(Bk);*/
			
		}
		e[i_end - 1] = f;
		
		/*Bk = buildB(s, e, m, n);
		fprintf("Bk:\n");
		printMatrix(Bk);*/
		
	}

	/**
	 * Implicit zero-shift QR algorithm on B_hat which is
	 * the bottommost unreduced submatrix of B begin from
	 * i_start (inclusive) to i_end (inclusive).
	 * 
	 * @param s diagonal elements
	 * 
	 * @param e superdiagonal elements
	 * 
	 * @param Ut left singular matrix
	 * 
	 * @param Vt right singular matrix
	 * 
	 * @param i_start start index of B_hat (inclusive)
	 * 
	 * @param i_end end index of B_hat (inclusive)
	 * 
	 * @param computeUV if U and V are to be computed
	 */
	private static void implicitZeroShiftQR(double[] s, double[] e, double[][] Ut, double[][] Vt, int i_start, int i_end, boolean computeUV) {
		
		/*int m = Ut.length;
		int n =Vt.length;
		int len = m >= n ? n : m;*/
		
		// Matrix Bk = null;
		
		double oldcs = 1;
		double oldsn = 0;
		double f = s[i_start];
		double g = e[i_start];
		double h = 0;
		double cs = 0, sn = 0, r = 0;
		double t, tt;
		
		for (int i = i_start; i < i_end; i++) {
			// ROT(f, g, cs, sn, r)
			if (f == 0) {
				cs = 0;
				sn = 1;
				r = g;
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
			if (computeUV) {
				update(cs, sn, Vt[i], Vt[i + 1]);
			}
			
			if (i != i_start) { // Note that i != i_start rather than i != 0!!!
				e[i - 1] = oldsn * r;
			}
			
			/*Bk = buildB(s, e, m, n);
			fprintf("Bk:\n");
			printMatrix(Bk);*/
			
			f = oldcs * r;
			g = s[i + 1] * sn;
			h = s[i + 1] * cs;
			
			// ROT(f, g, cs, sn, r)
			if (f == 0) {
				cs = 0;
				sn = 1;
				r = g;
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
			// UPDATE(cs, sn, Ui, Ui+1)
			if (computeUV) {
				update(cs, sn, Ut[i], Ut[i + 1]);
			}
			
			s[i] = r;
			f = h;
			g = e[i + 1];
			oldcs = cs;
			oldsn = sn;
			
			/*Bk = buildB(s, e, m, n);
			fprintf("Bk:\n");
			printMatrix(Bk);*/
			
		}
		e[i_end - 1] = h * sn;
		s[i_end] = h * cs;
		
		/*Bk = buildB(s, e, m, n);
		fprintf("Bk:\n");
		printMatrix(Bk);*/
		
	}
	
	private static Matrix buildS(double[] s, int m, int n) {
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (int i = 0; i < m; i++) {
			if (i < n)
				map.put(Pair.of(i, i), s[i]);
		}
		return SparseMatrix.createSparseMatrix(map, m, n);
	}
	
	@SuppressWarnings("unused")
	private static Matrix buildB(double[] s, double[] e, int m, int n) {
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (int i = 0; i < m; i++) {
			if (i < n)
				map.put(Pair.of(i, i), s[i]);
			if (i < n - 1) {
				map.put(Pair.of(i, i + 1), e[i]);
			}
			
		}
		return SparseMatrix.createSparseMatrix(map, m, n);
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

	/***
	 * Bidiagonalize a matrix A, i.e. A = U * B * V' such that
	 * U and V are orthogonal matrices and B is a bidiagonal matrix.
	 * 
	 * @param A a dense or sparse matrix
	 * 
	 * @return a {@code Matrix} array [U, B, V]
	 */
	@SuppressWarnings("unused")
	private static Matrix[] bidiagonalize(Matrix A) {
		return bidiagonalize(A, true);
	}
	
	/***
	 * Bidiagonalize a matrix A, i.e. A = U * B * V' such that
	 * U and V are orthogonal matrices and B is a bidiagonal matrix.
	 * 
	 * @param A a dense or sparse matrix
	 * 
	 * @param computeUV if U and V are to be computed
	 * 
	 * @return a {@code Matrix} array [U, B, V]
	 */
	private static Matrix[] bidiagonalize(Matrix A, boolean computeUV) {
		A = full(A).copy();
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		Matrix[] UBV = new Matrix[3];
		double[] d = ArrayOperator.allocateVector(n, 0);
		double[] e = ArrayOperator.allocateVector(n, 0);
		
		if (A instanceof DenseMatrix) {
			double[][] AData = ((DenseMatrix) A).getData();
			double c = 0;
			double s = 0;
			double r = 0;
			for (int j = 0; j < n; j++) {
				if (j >= m) {
					break;
				}
				// Householder transformation on columns of A(j:m, j:n)
				// Compute the norm of A(j:m, j)
				c = 0;
				for (int i = j; i < m; i++) {
					c += Math.pow(AData[i][j], 2);
				}
				if (c != 0) {
					s = Math.sqrt(c);
					d[j] = AData[j][j] > 0 ? -s : s;
					r = Math.sqrt(s * (s + abs(AData[j][j])));
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
					}
				}
				/*fprintf("Householder transformation on n - 1 columns:\n");
				disp(A);*/
				// disp(A);
				// Householder transformation on rows of A(j:m, j+1:n)
				if (j >= n - 1) // We do row-wise HouseHolder transformation n - 1 times
					continue;
				c = 0;
				double[] ARow_j = AData[j];
				for (int k = j + 1; k < n; k++) {
					c += Math.pow(ARow_j[k], 2);
				}
				if (c != 0) {
					s = Math.sqrt(c);
					e[j + 1] = ARow_j[j + 1] > 0 ? -s : s;
					r = Math.sqrt(s * (s + abs(ARow_j[j + 1])));
					ARow_j[j + 1] -= e[j + 1];
					for (int k = j + 1; k < n; k++) {
						ARow_j[k] /= r;
					}
					double[] ARow_k = null;
					for (int k = j + 1; k < m; k++) {
						ARow_k = AData[k];
						s = 0;
						for (int t = j + 1; t < n; t++) {
							s += ARow_j[t] * ARow_k[t];
						}
						for (int t = j + 1; t < n; t++) {
							ARow_k[t] -= s * ARow_j[t];
						}
					}
				}
				/*fprintf("Householder transformation on rows:\n");
				disp(A);*/
			}
		} else if (A instanceof SparseMatrix) {
			
		}
		
		/*disp("Processed A:");
		printMatrix(A);*/
		
		UBV = unpack(A, d, e, computeUV);
		return UBV;
	}

	/**
	 * Unpack U, B, and V from the result of bidiagonalization.
	 * 
	 * @param A bidiagonalization result
	 * 
	 * @param d diagonal
	 * 
	 * @param e superdiagonal
	 * 
	 * @param computeUV if U and V are to be computed
	 * 
	 * @return a {@code Matrix} array [U, B, V]
	 */
	private static Matrix[] unpack(Matrix A, double[] d, double[] e, boolean computeUV) {
		Matrix[] UBV = new Matrix[3];
		int m = A.getRowDimension();
		int n = A.getColumnDimension();
		DenseMatrix U = null;
		if (computeUV) {
			U = new DenseMatrix(m, m, 0);
			double[][] UData = U.getData();
			double s = 0;
			double[] y = null;
			for (int i = 0; i < m; i++) {
				// Compute U^T * e_i
				y = UData[i];
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
			/*fprintf("U:\n");
			disp(U);*/
		}

		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		for (int i = 0; i < m; i++) {
			if (i < n)
				map.put(Pair.of(i, i), d[i]);
			if (i < n - 1) {
				map.put(Pair.of(i, i + 1), e[i + 1]);
			}
			
		}
		Matrix B = SparseMatrix.createSparseMatrix(map, m, n);
		/*fprintf("B:\n");
		printMatrix(B);*/
		
		DenseMatrix V = null;
		if (computeUV) {
			V = new DenseMatrix(n, n, 0);
			double[][] VData = V.getData();
			double s = 0;
			double[] y = null;
			for (int i = 0; i < n; i++) {
				// Compute V^T * e_i
				y = VData[i];
				y[i] = 1;
				for (int j = 0; j < n - 1; j++) { // why not n - 2? Because we do 
					if (j == n - 2) {
						int a = 0;
						a += a;
					}
					s = 0;
					for (int k = j + 1; k < n; k++) {
						s += A.getEntry(j, k) * y[k];
					}
					for (int k = j + 1; k < n; k++) {
						y[k] -= A.getEntry(j, k) * s;
					}
				}
			}
			/*fprintf("V:\n");
			disp(V);*/
		}
		
		UBV[0] = U;
		UBV[1] = B;
		UBV[2] = V;
		
		if (computeUV) {
			/*fprintf("U'U:\n");
			disp(U.transpose().mtimes(U));

			fprintf("V'V:\n");
			disp(V.transpose().mtimes(V));*/

			/*fprintf("UBV':\n");
			disp(U.mtimes(B).mtimes(V.transpose()));*/
			
			/*disp("B:");
			printMatrix(B);*/
			// IO.saveMatrix(full(B), "SVD-B");
		}
		
		return UBV;
	}

}
