package org.laml.la.utils;

import static org.laml.la.utils.Utility.exit;
import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;
import org.laml.la.vector.DenseVector;
import org.laml.la.vector.SparseVector;
import org.laml.la.vector.Vector;

public class Printer {
	
	/**
	 * Print a sparse matrix with specified precision.
	 * 
	 * @param A a sparse matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void printSparseMatrix(Matrix A, int p) {
		
		if (!(A instanceof SparseMatrix)) {
			System.err.println("SparseMatrix input is expected.");
			return;
		}
		if (((SparseMatrix) A).getNNZ() == 0) {
			System.out.println("Empty sparse matrix.");
			System.out.println();
			return;
		}
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		String leftFormat = String.format("  %%%ds, ", String.valueOf(nRow).length() + 1);
		String rightFormat = String.format("%%-%ds", String.valueOf(nCol).length() + 2);
		String format = leftFormat + rightFormat + sprintf("%%%ds", 8 + p - 4);
		SparseMatrix S = (SparseMatrix) A;
		int[] ir = S.getIr();
		int[] jc = S.getJc();
		double[] pr = S.getPr();
		int N = S.getColumnDimension();
		String valueString = "";
		int i = -1;
		for (int j = 0; j < N; j++) {
			for (int k = jc[j]; k < jc[j + 1]; k++) {
				System.out.print("  ");
				i = ir[k];
				double v = pr[k];
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				/*System.out.print(sprintf(sprintf("(%d, %d)%%%ds", i + 1, j + 1, 8 + p - 4), valueString));
				System.out.println();*/
				String leftString = String.format("(%d", i + 1);
				String rightString = String.format("%d)", j + 1);
				System.out.println(String.format(format, leftString, rightString, valueString));
			}
		}
		System.out.println();

	}
	
	/**
	 * Print a sparse matrix.
	 * 
	 * @param A a sparse matrix
	 */
	public static void printSparseMatrix(Matrix A) {
		printSparseMatrix(A, 4);
	}
	
	/**
	 * Print a dense matrix with specified precision.
	 * 
	 * @param A a dense matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void printDenseMatrix(Matrix A, int p) {
		if (!(A instanceof DenseMatrix)) {
			System.err.println("DenseMatrix input is expected.");
			return;
		}
		if (((DenseMatrix) A).getData() == null) {
			System.out.println("Empty matrix.");
			return;
		}
		for (int i = 0; i < A.getRowDimension(); i++) {
			System.out.print("  ");
			for (int j = 0; j < A.getColumnDimension(); j++) {
				String valueString = "";
				double v = A.getEntry(i, j);
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 * Print a dense matrix.
	 * 
	 * @param A a dense matrix
	 */
	public static void printDenseMatrix(Matrix A) {
		printDenseMatrix(A, 4);
	}
	
	/**
	 * Print a matrix with specified precision. A sparse matrix
	 * will be printed like a dense matrix except that zero entries 
	 * will be left blank.
	 * 
	 * @param A a dense or sparse matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printMatrix(Matrix A, int p) {
		if (A == null) {
			System.out.println("Empty matrix.");
			return;
		}
		if (A instanceof SparseMatrix) {
			if (((SparseMatrix) A).getNNZ() == 0) {
				System.out.println("Empty sparse matrix.");
				return;
			} 
			SparseMatrix S = (SparseMatrix) A;
			int[] ic = S.getIc();
			int[] jr = S.getJr();
			double[] pr = S.getPr();
			int[] valCSRIndices = S.getValCSRIndices();
			int M = S.getRowDimension();
			String valueString = "";
			for (int r = 0; r < M; r++) {
				System.out.print("  ");
				int currentColumn = 0;
				int lastColumn = -1;
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					currentColumn = ic[k];
					while (lastColumn < currentColumn - 1) {
						System.out.printf(String.format("%%%ds", 8 + p - 4), " ");
						System.out.print("  ");
						lastColumn++;
					}
					lastColumn = currentColumn;
					double v = pr[valCSRIndices[k]];
					int rv = (int) Math.round(v);
					if (v != rv)
						valueString = sprintf(sprintf("%%.%df", p), v);
					else
						valueString = sprintf("%d", rv);
					System.out.printf(sprintf("%%%ds", 8 + p - 4), valueString);
					System.out.print("  ");
				}
				System.out.println();
			}
			System.out.println();
			return;
		}
		if (A instanceof DenseMatrix) {
			if (((DenseMatrix) A).getData() == null) {
				System.out.println("Empty matrix.");
				return;
			}
			for (int i = 0; i < A.getRowDimension(); i++) {
				System.out.print("  ");
				for (int j = 0; j < A.getColumnDimension(); j++) {
					String valueString = "";
					double v = A.getEntry(i, j);
					int rv = (int) Math.round(v);
					if (v != rv)
						valueString = sprintf(sprintf("%%.%df", p), v);
					else
						valueString = sprintf("%d", rv);
					System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
					System.out.print("  ");
				}
				System.out.println();
			}
			System.out.println();
		}

	}
	
	/**
	 * Print a matrix. A sparse matrix will be printed like a dense 
	 * matrix except that zero entries will be left blank.
	 * 
	 * @param A a dense or sparse matrix
	 * 
	 */
	public static void printMatrix(Matrix A) {
		printMatrix(A, 4);
	}
	
	/**
	 * Print a column matrix with specified precision.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printMatrix(double[] V, int p) {

		for (int i = 0; i < V.length; i++) {
			System.out.print("  ");
			String valueString = "";
			double v = V[i];
			int rv = (int) Math.round(v);
			if (v != rv)
				valueString = sprintf(sprintf("%%.%df", p), v);
			else
				valueString = sprintf("%d", rv);
			System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
			System.out.print("  ");
			System.out.println();
		}
		System.out.println();

	}
	
	/**
	 * Print a column matrix.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 */
	public static void printMatrix(double[] V) {
		printMatrix(V, 4);
	}
	
	/**
	 * Print a row vector with specified precision.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printVector(double[] V, int p) {
		for (int i = 0; i < V.length; i++) {
			System.out.print("  ");
			String valueString = "";
			double v = V[i];
			int rv = (int) Math.round(v);
			if (v != rv)
				valueString = sprintf(sprintf("%%.%df", p), v);
			else
				valueString = sprintf("%d", rv);
			System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
		}
		System.out.println();
		System.out.println();
	}
	
	/**
	 * Print a row vector.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 */
	public static void printVector(double[] V) {
		printVector(V, 4);
	}
	
	/**
	 * Print a row vector with specified precision.
	 * 
	 * @param V a dense or sparse vector
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printVector(Vector V, int p) {
		if (V instanceof DenseVector) {
			printDenseVector(V, p);
		} else {
			printSparseVector(V, p);
		}
	}
	
	/**
	 * Print a row vector.
	 * 
	 * @param V a dense or sparse vector
	 */
	public static void printVector(Vector V) {
		printVector(V, 4);
	}
	
	/**
	 * Print a dense vector with specified precision.
	 * 
	 * @param V a dense vector
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printDenseVector(Vector V, int p) {
		if (V instanceof DenseVector) {
			int dim = V.getDim();
			double[] pr = ((DenseVector) V).getPr();
			for (int k = 0; k < dim; k++) {
				System.out.print("  ");
				double v = pr[k];
				int rv = (int) Math.round(v);
				String valueString;
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				System.out.print(sprintf(sprintf("%%%ds", 8 + p - 4), valueString));
				System.out.println();
			}
			System.out.println();
		} else {
			System.err.println("The input vector should be a DenseVector instance");
			System.exit(1);
		}
	}
	
	/**
	 * Print a dense vector.
	 * 
	 * @param V a dense vector.
	 */
	public static void printDenseVector(Vector V) {
		printDenseVector(V, 4);
	}
	
	/**
	 * Print a sparse vector with specified precision.
	 * 
	 * @param V a sparse vector
	 * 
	 * @param p number of digits after decimal point with rounding
	 * 
	 */
	public static void printSparseVector(Vector V, int p) {
		if (V instanceof SparseVector) {
			int[] ir = ((SparseVector) V).getIr();
			double[] pr = ((SparseVector) V).getPr();
			int nnz = ((SparseVector) V).getNNZ();
			for (int k = 0; k < nnz; k++) {
				System.out.print("  ");
				int idx = ir[k];
				double v = pr[k];
				int rv = (int) Math.round(v);
				String valueString;
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				System.out.print(sprintf(sprintf("(%d, 1)%%%ds", idx + 1, 8 + p - 4), valueString));
				System.out.println();
			}
			System.out.println();
		} else {
			System.err.println("The input vector should be a SparseVector instance");
			System.exit(1);
		}
	}
	
	/**
	 * Print a sparse vector.
	 * 
	 * @param V a sparse vector
	 */
	public static void printSparseVector(Vector V) {
		printSparseVector(V, 4);
	}
	
	/**
	 * Display a vector with specified precision.
	 * 
	 * @param V a dense or sparse vector
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void display(Vector V, int p) {
		printVector(V, p);
	}
	
	/**
	 * Display a vector.
	 * 
	 * @param V a dense or sparse vector
	 */
	public static void display(Vector V) {
		display(V, 4);
	}
	
	/**
	 * Display a 1D {@code double} array with specified precision.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void display(double[] V, int p) {
		printVector(new DenseVector(V), p);
	}
	
	/**
	 * Display a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 */
	public static void display(double[] V) {
		display(V, 4);
	}
	
	/**
	 * Display a matrix with specified precision.
	 * 
	 * @param A a dense or sparse matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void display(Matrix A, int p) {
		if (A instanceof DenseMatrix) {
			printDenseMatrix(A, p);
		} else if (A instanceof SparseMatrix) {
			printSparseMatrix(A, p);
		}
	}
	
	/**
	 * Display a matrix.
	 * 
	 * @param A a dense or sparse matrix
	 */
	public static void display(Matrix A) {
		display(A, 4);
	}
	
	/**
	 * Display a 2D {@code double} array with specified precision.
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void display(double[][] A, int p) {
		printMatrix(new DenseMatrix(A), p);
	}
	
	/**
	 * Display a 2D {@code double} array.
	 * 
	 * @param A a 2D {@code double} array
	 */
	public static void display(double[][] A) {
		display(A, 4);
	}
	
	/**
	 * Display a vector with specified precision.
	 * 
	 * @param V a dense or sparse vector
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void disp(Vector V, int p) {
		display(V, p);
	}
	
	/**
	 * Display a vector.
	 * 
	 * @param V a dense or sparse vector
	 */
	public static void disp(Vector V) {
		display(V, 4);
	}
	
	/**
	 * Display a 1D {@code double} array with specified precision.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void disp(double[] V, int p) {
		display(new DenseVector(V), p);
	}
	
	/**
	 * Display a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 */
	public static void disp(double[] V) {
		display(new DenseVector(V), 4);
	}
	
	/**
	 * Display a matrix with specified precision.
	 * 
	 * @param A a dense or sparse matrix
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void disp(Matrix A, int p) {
		display(A, p);
	}
	
	/**
	 * Display a matrix.
	 * 
	 * @param A a dense or sparse matrix
	 */
	public static void disp(Matrix A) {
		display(A, 4);
	}
	
	/**
	 * Display a 2D {@code double} array with specified precision.
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void disp(double[][] A, int p) {
		display(new DenseMatrix(A), p);
	}
	
	/**
	 * Display a 2D {@code double} array.
	 * 
	 * @param A a 2D {@code double} array
	 */
	public static void disp(double[][] A) {
		display(new DenseMatrix(A), 4);
	}
	
	/**
	 * Display a real scalar.
	 * 
	 * @param v a {@code double} variable
	 */
	public static void disp(double v) {
		System.out.print("  ");
		System.out.println(v);
	}
	
	/**
	 * Display a 1D integer array.
	 * 
	 * @param V a 1D integer array
	 * 
	 */
	public static void disp(int[] V) {
		display(V);
	}
	
	/**
	 * Display an integer 2D array.
	 * 
	 * @param M an integer 2D array
	 */
	public static void disp(int[][] M) {
		display(M);
	}
	
	/**
	 * Display a 1D integer array.
	 * 
	 * @param V a 1D integer array
	 * 
	 */
	public static void display(int[] V) {
		
		if (V == null) {
			System.out.println("Empty vector!");
			return;
		}
		
		for (int i = 0; i < V.length; i++) {
			System.out.print("  ");
			String valueString = "";
			double v = V[i];
			int rv = (int) Math.round(v);
			if (v != rv)
				valueString = String.format("%.4f", v);
			else
				valueString = String.format("%d", rv);
			System.out.print(String.format("%7s", valueString));
			System.out.print("  ");
			// System.out.println();
		}
		System.out.println();
		
	}
	
	/**
	 * Display an integer 2D array.
	 * 
	 * @param M an integer 2D array
	 */
	public static void display(int[][] M) {
		if (M == null) {
			System.out.println("Empty matrix!");
			return;
		}
		for (int i = 0; i < M.length; i++) {
			System.out.print("  ");
			for (int j = 0; j < M[0].length; j++) {
				String valueString = "";
				double v = M[i][j];
				int rv = (int) Math.round(v);
				if (v != rv)
					valueString = String.format("%.4f", v);
				else
					valueString = String.format("%d", rv);
				System.out.print(String.format("%7s", valueString));
				System.out.print("  ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	/**
	 * Display a string.
	 * 
	 * @param str a string to display
	 */
	public static void display(String str) {
		fprintf("%s%n", str);
	}
	
	/**
	 * Display a string.
	 * 
	 * @param str a string to display
	 */
	public static void disp(String str) {
		fprintf("%s%n", str);
	}

	/**
	 * Format variables into a string.
	 * 
	 * @param format a string describing the format of the output fields
	 *        
	 * @param os argument list applying the format to
	 * 
	 * @return a formatted string
	 * 
	 */
	public static String sprintf(String format, Object... os) {
		return String.format(format, os);
	}
	
	/**
	 * Write a formatted string to the standard output (the screen).
	 * 
	 * @param format a string describing the format of the output fields
	 *        
	 * @param os argument list applying the format to
	 * 
	 */
	public static void fprintf(String format, Object... os) {
		System.out.format(format, os);
	}
	
	/**
	 * Write a formatted string to the standard output (the screen).
	 * 
	 * @param format a string describing the format of the output fields
	 *        
	 * @param os argument list applying the format to
	 * 
	 */
	public static void printf(String format, Object... os) {
		System.out.format(format, os);
	}
	
	/**
	 * Print a string to the standard output (the screen).
	 * 
	 * @param content a string
	 */
	public static void print(String content) {
		System.out.print(content);
	}
	
	/**
	 * Print a character to the standard output (the screen).
	 * 
	 * @param c a character
	 */
	public static void print(char c) {
		System.out.print(c);
	}
	
	/**
	 * Print an array of characters to the standard output (the screen).
	 * 
	 * @param s the array of chars to be printed
	 */
	public static void print(char[] s) {
		System.out.print(s);
	}
	
	/**
	 * Print an array of integers to the standard output (the screen).
	 * 
	 * @param A an integer array
	 */
	public static void print(int[] A) {
		int n = A.length;
		for (int i = 0; i < n - 1; i++) {
			System.out.print(A[i]);
			System.out.print(' ');
		}
		System.out.print(A[n - 1]);
	}
	
	/**
	 * Print an object to the standard output (the screen).
	 * 
	 * @param obj an object
	 */
	public static void print(Object obj) {
		System.out.print(obj);
	}
	
	/**
	 * Print a string to the standard output (the screen) 
	 * and then terminate the line.
	 * 
	 * @param content a string
	 */
	public static void println(String content) {
		System.out.println(content);
	}

	/**
	 * Print an array of characters to the standard output (the screen)
	 * and then terminate the line.
	 * 
	 * @param s the array of chars to be printed
	 */
	public static void println(char[] s) {
		System.out.println(s);
	}
	
	/**
	 * Print an array of integers to the standard output (the screen)
	 * and then terminate the line.
	 * 
	 * @param A an integer array
	 */
	public static void println(int[] A) {
		int n = A.length;
		for (int i = 0; i < n - 1; i++) {
			System.out.print(A[i]);
			System.out.print(' ');
		}
		System.out.println(A[n - 1]);
	}
	
	/**
	 * Print an object and then terminate the line.
	 * 
	 * @param obj an object
	 */
	public static void println(Object obj) {
		System.out.println(obj);
	}
	
	public static void println() {
		System.out.println();
	}

	/**
	 * Print the error information in standard output.
	 * 
	 * @param input a {@code String} representing the error 
	 */
	public static void err(String input) {
		System.err.println(input);
	}
	
	/**
	 * Print a formatted string of error information to 
	 * the standard output (the screen).
	 * 
	 * @param format a string describing the format of the output fields
	 *        
	 * @param os argument list applying the format to
	 * 
	 */
	public static void errf(String format, Object... os) {
		System.err.format(format, os);
	}
	
	/**
	 * Display specification for each vector element.
	 * 
	 * @param V a dense vector
	 * 
	 * @param spec specification strings for all elements
	 */
	public static void showSpec(DenseVector V, String[] spec) {
		showSpec(V, spec, 4);
	}
	
	/**
	 * Display specification for each vector element.
	 * 
	 * @param V a dense vector
	 * 
	 * @param spec specification strings for all elements
	 * 
	 * @param p number of digits after decimal point with rounding
	 */
	public static void showSpec(DenseVector V, String[] spec, int p) {
		
		if (V instanceof DenseVector) {
			int dim = V.getDim();
			double[] pr = ((DenseVector) V).getPr();
			for (int k = 0; k < dim; k++) {
				print("  ");
				double v = pr[k];
				int rv = (int) Math.round(v);
				String valueString;
				if (v != rv)
					valueString = sprintf(sprintf("%%.%df", p), v);
				else
					valueString = sprintf("%d", rv);
				println(sprintf(sprintf("%%%ds  %%s", 8 + p - 4), valueString, spec[k]));
			}
			println();
		} else {
			System.err.println("The input vector should be a DenseVector instance");
			exit(1);
		}

	}

}
