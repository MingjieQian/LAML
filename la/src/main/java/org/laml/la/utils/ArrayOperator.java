package org.laml.la.utils;

import static org.laml.la.utils.Matlab.full;
import static org.laml.la.utils.Matlab.mldivide;
import static org.laml.la.utils.Matlab.speye;
import org.laml.la.matrix.DenseMatrix;

/***
 * The {@code ArrayOperation} includes frequently used operation 
 * functions on {@code double} arrays. The argument vector is 
 * required to have been allocated memory before being used in 
 * the array operations.
 * 
 * @author Mingjie Qian
 * @version 1.0, Feb. 21st, 2013
 */
public class ArrayOperator {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// double[] A = {3.0, 2.0, 2.5, 1.3, 2.0, 3.0, 0.2, 5.1, 1.3};
		double[] A = {1.0, 1.0, 1.0, 1.0, 1.0};
		Printer.printVector(A);
		Printer.println(sort(A, "descend"));
		Printer.printVector(A);
	}
	
	/**
	 * Compute A \ I.
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @return inv(A)
	 */
	public static double[][] inv(double[][] A) {
		int M = A.length;
		int N = A[0].length;
		if (M != N) {
			System.err.println("The input 2D array should be square.");
			System.exit(1);
		}
		return full(mldivide(new DenseMatrix(A), speye(M))).getData();
	}
	
	/**
	 * Sort a {@code double} array in a specified order.
	 * 
	 * @param V a {@code double} array
	 * 
	 * @param order "ascend" or "descend"
	 * 
	 * @return original indices for the result
	 */
	public static int[] sort(double[] V, String order) {
		int len = V.length;
		int[] indices = colon(0, 1, len - 1);
		int start = 0;
		int end = len - 1;
		quickSort(V, indices, start, end, order);
		return indices;
	}
	
	/**
	 * Sort a {@code double} array in an ascending order.
	 * 
	 * @param V a {@code double} array
	 * 
	 * @return original indices for the result
	 */
	public static int[] sort(double[] V) {
		return sort(V, "ascend");
	}
	
	/**
	 * Sort a 1D {@code double} array in a specified order.
	 * 
	 * @param values a 1D {@code double} array containing the values to be sort
	 * 
	 * @param indices index array
	 * 
	 * @param start start index (inclusive)
	 * 
	 * @param end end index (inclusive)
	 * 
	 * @param order a {@code String} variable either "descend" or "ascend"
	 * 
	 */
	public static void quickSort(double[] values, int[] indices, int start, int end, String order) {

		int	i,j;
		double temp;
		i = start;
		j = end;
		temp = values[i]; // temp is the pivot
		int tempV = indices[i];
		do{
			if (order.equals("ascend")) {
				while(((values[j]) >= (temp)) && (j > i)) // find the first j such that values[j] < temp
					j--;
			} else if (order.equals("descend")) {
				while(((values[j]) <= (temp)) && (j > i))
					j--;
			} 
			if(j > i){
				values[i] = values[j]; // record this values[j]
				indices[i] = indices[j];
				i++;
			}
			if (order.equals("ascend")) {
				while(((values[i]) <= (temp)) && (i < j)) // find the first i such that values[i] > temp
					i++;
			} else if (order.equals("descend")) {
				while(((values[i]) >= (temp)) && (i < j))
					i++;
			}
			if(j > i){
				values[j] = values[i]; // record this value[i]
				indices[j] = indices[i];
				j--;
			}
		} while(i != j);
		values[i] = temp;
		indices[i] = tempV;
		i++;
		j--;
		if(start < j)
			quickSort(values, indices, start, j, order);
		if(i < end)
			quickSort(values, indices, i, end, order);
		
	}
	
	/**
	 * Sort a 1D {@code double} array in a specified order.
	 * 
	 * @param values a 1D {@code double} array containing the values to be sort
	 * 
	 * @param indices index array
	 * 
	 * @param start start index (inclusive)
	 * 
	 * @param end end index (inclusive)
	 * 
	 * @param order a {@code String} variable either "descend" or "ascend"
	 * 
	 */
	public static void quickSort(double[] values, double[] indices, int start, int end, String order) {

		int	i,j;
		double temp;
		i = start;
		j = end;
		temp = values[i];
		double tempV = indices[i];
		do{
			if (order.equals("ascend")) {
				while(((values[j]) >= (temp)) && (j > i))
					j--;
			} else if (order.equals("descend")) {
				while(((values[j]) <= (temp)) && (j > i))
					j--;
			} 
			if(j > i){
				values[i] = values[j];
				indices[i] = indices[j];
				i++;
			}
			if (order.equals("ascend")) {
				while(((values[i]) <= (temp)) && (i < j))
					i++;
			} else if (order.equals("descend")) {
				while(((values[i]) >= (temp)) && (i < j))
					i++;
			}
			if(j > i){
				values[j] = values[i];
				indices[j] = indices[i];
				j--;
			}
		} while(i != j);
		values[i] = temp;
		indices[i] = tempV;
		i++;
		j--;
		if(start < j)
			quickSort(values, indices, start, j, order);
		if(i < end)
			quickSort(values, indices, i, end, order);
		
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
	 */
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
	
	/**
	 * Same as colon(begin, 1, end).
	 * 
	 * @param begin starting point (inclusive)
	 * 
	 * @param end ending point (inclusive)
	 * 
	 * @return indices array for the syntax begin:end
	 * 
	 */
	public static int[] colon(int begin, int end) {
		return colon(begin, 1, end);
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
	 */
	public static double[] colon(double begin, double d, double end) {

		int m = fix((end - begin) / d);
		if (m < 0) {
			System.err.println("Difference error!");
			System.exit(1);
		}
		
		double[] res = new double[m + 1];
		
		for (int i = 0; i <= m; i++) {
			res[i] = begin + i * d;
		}

		return res;

	}
	
	/**
	 * Same as colon(begin, 1, end).
	 * 
	 * @param begin starting point (inclusive)
	 * 
	 * @param end ending point (inclusive)
	 * 
	 * @return indices array for the syntax begin:end
	 * 
	 */
	public static double[] colon(double begin, double end) {
		return colon(begin, 1, end);
	}

	/**
	 * Compute the maximum argument.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return maximum argument
	 * 
	 */
	public static int argmax(double[] V) {

		int maxIdx = 0;
		double maxVal = V[0];
		for (int i = 1; i < V.length; i++) {
			if (maxVal < V[i]) {
				maxVal = V[i];
				maxIdx = i;
			}
		}
		return maxIdx;

	}
	
	/**
	 * Compute the maximum argument.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param begin beginning index (inclusive)
	 * 
	 * @param end end index (exclusive)
	 * 
	 * @return maximum argument
	 */
	public static int argmax(double[] V, int begin, int end) {
		int maxIdx = begin;
		double maxVal = V[begin];
		for (int i = begin + 1; i < end; i++) {
			if (maxVal < V[i]) {
				maxVal = V[i];
				maxIdx = i;
			}
		}
		return maxIdx;
	}
	
	/**
	 * Compute the minimum argument.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return minimum argument
	 * 
	 */
	public static int argmin(double[] V) {

		int maxIdx = 0;
		double maxVal = V[0];
		for (int i = 1; i < V.length; i++) {
			if (maxVal > V[i]) {
				maxVal = V[i];
				maxIdx = i;
			}
		}
		return maxIdx;

	}
	
	/**
	 * Compute the minimum argument.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param begin beginning index (inclusive)
	 * 
	 * @param end end index (exclusive)
	 * 
	 * @return minimum argument
	 */
	public static int argmin(double[] V, int begin, int end) {
		int maxIdx = begin;
		double maxVal = V[begin];
		for (int i = begin + 1; i < end; i++) {
			if (maxVal > V[i]) {
				maxVal = V[i];
				maxIdx = i;
			}
		}
		return maxIdx;
	}
	
	/**
	 * Compute the minimal element.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return minimal element
	 */
	public static double min(double[] V) {
		double res = V[0];
		for (int i = 1; i < V.length; i++) {
			if (res > V[i]) {
				res = V[i];
			}
		}
		return res;
	}
	
	/**
	 * Compute the maximal element.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return maximal element
	 */
	public static double max(double[] V) {
		double res = V[0];
		for (int i = 1; i < V.length; i++) {
			if (res < V[i]) {
				res = V[i];
			}
		}
		return res;
	}
	
	/**
	 * Compute the minimal element.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param begin beginning index (inclusive)
	 * 
	 * @param end end index (exclusive)
	 * 
	 * @return minimal element
	 */
	public static double min(double[] V, int begin, int end) {
		double res = V[0];
		for (int i = begin + 1; i < end; i++) {
			if (res > V[i]) {
				res = V[i];
			}
		}
		return res;
	}
	
	/**
	 * Compute the maximal element.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param begin beginning index (inclusive)
	 * 
	 * @param end end index (exclusive)
	 * 
	 * @return maximal element
	 */
	public static double max(double[] V, int begin, int end) {
		double res = V[begin];
		for (int i = begin + 1; i < end; i++) {
			if (res < V[i]) {
				res = V[i];
			}
		}
		return res;
	}
	
	/**
	 * Assign a 1D {@code double} array by a real scalar.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assignVector(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] = v;
	}
	
	/**
	 * Assign a 1D {@code double} array by a real scalar.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assign(double[] V, double v) {
		assignVector(V, v);
	}
	
	/**
	 * Assign a 1D {@code int} array by a real scalar.
	 * 
	 * @param V a 1D {@code int} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assignIntegerVector(int[] V, int v) {
		for (int i = 0; i < V.length; i++)
			V[i] = v;
	}
	
	/**
	 * Assign a 1D {@code int} array by a real scalar.
	 * 
	 * @param V a 1D {@code int} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assign(int[] V, int v) {
		assign(V, v);
	}

	/**
	 * Clear all elements of a 1D {@code double} array to zero.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 */
	public static void clearVector(double[] V) {
		assignVector(V, 0);
	}
	
	/**
	 * Clear all elements of a 1D {@code double} array to zero.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 */
	public static void clear(double[] V) {
		clearVector(V);
	}

	/**
	 * Clear all elements of a 2D {@code double} array to zero.
	 * 
	 * @param M a 2D {@code double} array
	 * 
	 */
	public static void clearMatrix(double[][] M) {
		for (int i = 0; i < M.length; i++) {
			assignVector(M[i], 0);
		}
	}
	
	/**
	 * Clear all elements of a 2D {@code double} array to zero.
	 * 
	 * @param M a 2D {@code double} array
	 * 
	 */
	public static void clear(double[][] M) {
		clearMatrix(M);
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 */
	public static double[] allocate1DArray(int n) {
		return allocateVector(n, 0);
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array and assign all elements with a given value.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @param v a real scalar to assign the 1D {@code double} array
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 */
	public static double[] allocate1DArray(int n, double v) {
		return allocateVector(n, v);
	}

	/**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 */
	public static double[] allocateVector(int n) {
		return allocateVector(n, 0);
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code double}
	 * array and assign all elements with a given value.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @param v a real scalar to assign the 1D {@code double} array
	 * 
	 * @return a 1D {@code double} array of length n
	 * 
	 */
	public static double[] allocateVector(int n, double v) {
		double[] res = new double[n];
		assignVector(res, v);
		return res;
	}
	
	/**
	 * Allocate a 2D {@code double} array and assign all 
	 * elements with a given value.
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @param v a real scalar
	 * 
	 * @return a 2D {@code double} array
	 * 
	 */
	public static double[][] allocate2DArray(int m, int n, double v) {
		double[][] res = new double[m][];
		for (int i = 0; i < m; i++) {
			res[i] = new double[n];
			for (int j = 0; j < n; j++) {
				res[i][j] = v;
			}
		}
		return res;
	}
	
	/**
	 * Allocate a 2D {@code double} array and assign all 
	 * elements with zero by default.
	 * 
	 * @param m number of rows
	 * 
	 * @param n number of columns
	 * 
	 * @return a 2D {@code double} array
	 * 
	 */
	public static double[][] allocate2DArray(int m, int n) {
		return allocate2DArray(m, n, 0);
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code int}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @return a 1D {@code int} array of length n
	 * 
	 */
	public static int[] allocateIntegerVector(int n) {
		return allocateIntegerVector(n, 0);
	}
	
	/**
	 * Allocate continuous memory block for a 1D {@code int}
	 * array.
	 * 
	 * @param n number of elements to be allocated
	 * 
	 * @param v an integer to initialize the vector
	 * 
	 * @return a 1D {@code int} array of length n
	 * 
	 */
	public static int[] allocateIntegerVector(int n, int v) {
		int[] res = new int[n];
		assignIntegerVector(res, v);
		return res;
	}

	/**
	 * Allocate memory for a 2D {@code double} array.
	 * 
	 * @param nRows number of rows
	 * 
	 * @param nCols number of columns
	 * 
	 * @return a nRows by nCols 2D {@code double} array
	 * 
	 */
	public static double[][] allocateMatrix(int nRows, int nCols) {
		double[][] res = new double[nRows][];
		for (int i = 0; i < nRows; i++) {
			res[i] = allocateVector(nCols);
		}
		return res;
	}

	/**
	 * Element-wise division and assignment operation. It divides
	 * the first argument with the second argument and assign
	 * the result to the first argument, i.e., V = V / v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void divideAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] /= v;
	}
	
	/**
	 * Element-wise division and assignment operation. It divides
	 * the first argument with the second argument and assign
	 * the result to the first argument, i.e., V1 = V1 ./ V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void divideAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] /= V2[i];
	}
	
	/**
	 * res = res / v.
	 * 
	 * @param res a 2D {@code double} array
	 * 
	 * @param v a real scalar
	 */
	public static void divideAssign(double[][] res, double v) {
		for (int i = 0; i < res.length; i++) {
			double[] row = res[i];
			for (int j = 0; j < row.length; j++) {
				row[j] /= v;
			}
		}
	}

	/**
	 * Element-wise multiplication and assignment operation.
	 * It multiplies the first argument with the second argument
	 * and assign the result to the first argument, i.e., V = V * v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void timesAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] *= v;
	}

	/**
	 * Element-wise multiplication and assignment operation.
	 * It multiplies the first argument with the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 .* V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void timesAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] *= V2[i];
	}
	
	/**
	 * res = res * v.
	 * 
	 * @param res a 2D {@code double} array
	 * 
	 * @param v a real scalar
	 */
	public static void timesAssign(double[][] res, double v) {
		for (int i = 0; i < res.length; i++) {
			double[] row = res[i];
			for (int j = 0; j < row.length; j++) {
				row[j] *= v;
			}
		}
	}

	/**
	 * Compute the sum of a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return sum(V)
	 * 
	 */
	public static double sum(double[] V) {
		double res = 0;
		for (int i = 0; i < V.length; i++)
			res += V[i];
		return res;
	}
	
	/**
	 * Compute the mean of a 1D {@code double} array.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return sum(V) / |V|
	 */
	public static double mean(double[] V) {
		return sum(V) / V.length;
	}
	
	/**
	 * Compute standard deviation.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * flag 0: n - 1 in the divisor, 1: n in the divisor
	 * 
	 * @return standard deviation of V
	 */
	public static double std(double[] V, int flag) {
		int n = V.length;
		if (n == 1) {
			return 0;
		}
		double mean = mean(V);
		double res = 0;
		for (double v : V) {
			double diff = v - mean;
			res += diff * diff;
		}
		if (flag == 0) {
			res /= n - 1;
		} else if (flag == 1) {
			res /= n;
		}
		res = Math.sqrt(res);
		return res;
	}

	/**
	 * Sum a 1D {@code double} array to one, i.e., V[i] = V[i] / sum(V).
	 * 
	 * @param V a 1D {@code double} array
	 */
	public static void sum2one(double[] V) {
		divideAssign(V, sum(V));
	}
	
	/**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument by the second argument
	 * and assign the result to the first argument, i.e., V = V + v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void plusAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] += v;
	}
	
	/**
	 * res += a * V. V can be longer than res, but the 
	 * extra elements will not be used. 
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param a a real scalar
	 * 
	 * @param V a 1D {@code double} array
	 */
	public static void plusAssign(double[] res, double a, double[] V) {
		for (int i = 0; i < res.length; i++) {
			res[i] += a * V[i];
		}
	}

	/**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument by the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 + V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void plusAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] += V2[i];
	}
	
	/**
	 * res = res + v.
	 * 
	 * @param res a 2D {@code double} array
	 * 
	 * @param v a real scalar
	 */
	public static void plusAssign(double[][] res, double v) {
		for (int i = 0; i < res.length; i++) {
			double[] row = res[i];
			for (int j = 0; j < row.length; j++) {
				row[j] += v;
			}
		}
	}
	
	/**
	 * Element-wise subtraction and assignment operation.
	 * It subtracts the first argument by the second argument
	 * and assign the result to the first argument, i.e., V = V - v.
	 * 
	 * @param V a 1D {@code int} array
	 * 
	 * @param v an integer
	 * 
	 */
	public static void minusAssign(int[] V, int v) {
		for (int i = 0; i < V.length; i++)
			V[i] -= v;
	}
	
	/**
	 * Element-wise subtraction and assignment operation.
	 * It subtracts the first argument by the second argument
	 * and assign the result to the first argument, i.e., V = V - v.
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void minusAssign(double[] V, double v) {
		for (int i = 0; i < V.length; i++)
			V[i] -= v;
	}
	
	/**
	 * res -= a * V. V can be longer than res, but the 
	 * extra elements will not be used. 
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param a a real scalar
	 * 
	 * @param V a 1D {@code double} array
	 */
	public static void minusAssign(double[] res, double a, double[] V) {
		for (int i = 0; i < res.length; i++) {
			res[i] -= a * V[i];
		}
	}

	/**
	 * Element-wise subtraction and assignment operation.
	 * It subtracts the first argument by the second argument
	 * and assign the result to the first argument, i.e., V1 = V1 - V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void minusAssign(double[] V1, double[] V2) {
		for (int i = 0; i < V1.length; i++)
			V1[i] -= V2[i];
	}
	
	/**
	 * res = res - v.
	 * 
	 * @param res a 2D {@code double} array
	 * 
	 * @param v a real scalar
	 */
	public static void minusAssign(double[][] res, double v) {
		for (int i = 0; i < res.length; i++) {
			double[] row = res[i];
			for (int j = 0; j < row.length; j++) {
				row[j] -= v;
			}
		}
	}

	/**
	 * Element-wise assignment operation. It assigns the first argument
	 * with the second argument, i.e., V1 = V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void assignVector(double[] V1, double[] V2) {
		/*for (int i = 0; i < V1.length; i++)
			V1[i] = V2[i];*/
		System.arraycopy(V2, 0, V1, 0, V1.length);
	}
	
	/**
	 * Populate a 2D {@code double} array res by another
	 * 2D {@code double} array A, i.e., res[i][j] = A[i][j].
	 * 
	 * @param res a 2D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 */
	public static void assign(double[][] res, double[][] A) {
		for (int i = 0; i < res.length; i++)
			assignVector(res[i], A[i]);
	}
	
	/**
	 * Set all elements of a 2D {@code double} array res by
	 * a real scalar v.
	 * 
	 * @param res a 2D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assign(double[][] res, double v) {
		for (int i = 0; i < res.length; i++)
			assignVector(res[i], v);
	}
	
	/**
	 * res = A * V;
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @return A * V
	 * 
	 */
	public static double[] operate(double[][] A, double[] V) {
		double[] res = new double[A.length];
		double s = 0;
		for (int i = 0; i < res.length; i++) {
			s = 0;
			double[] A_i = A[i];
			for (int j = 0; j < V.length; j++) {
				s += A_i[j] * V[j];
			}
			res[i] = s;
		}
		return res;
	}
	
	/**
	 * V1 = A * V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 */
	public static void operate(double[] V1, double[][] A, double[] V2) {
		
		double s = 0;
		for (int i = 0; i < V1.length; i++) {
			double[] ARow = A[i];
			s = 0;
			for (int j = 0; j < V2.length; j++) {
				s += ARow[j] * V2[j];
			}
			V1[i] = s;
		}
		
	}
	
	/**
	 * V1 = A * V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param A a real matrix
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 *//*
	public static void operate(double[] V1, Matrix A, double[] V2) {
		
		double s = 0;
		for (int i = 0; i < V1.length; i++) {
			s = 0;
			for (int j = 0; j < V2.length; j++) {
				s += A.get(i, j) * V2[j];
			}
			V1[i] = s;
		}
		
	}*/
	
	/**
	 * res' = V' * A
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 * @return V' * A
	 * 
	 */
	public static double[] operate(double[] V, double[][] A) {
		double[] res = new double[A[0].length];
		double s = 0;
		for (int j = 0; j < res.length; j++) {
			s = 0;
			for (int i = 0; i < V.length; i++) {
				s += V[i] * A[i][j];
			}
			res[j] = s;
		}
		return res;
	}
	
	/**
	 * V1' = V2' * A.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 */
	public static void operate(double[] V1, double[] V2, double[][] A) {
		
		double s = 0;
		for (int j = 0; j < V1.length; j++) {
			s = 0;
			for (int i = 0; i < V2.length; i++) {
				s += V2[i] * A[i][j];
			}
			V1[j] = s;
		}
		
	}
	
	/**
	 * V1' = V2' * A.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @param A a 2D {@code double} array
	 * 
	 *//*
	public static void operate(double[] V1, double[] V2, Matrix A) {
		
		double s = 0;
		for (int j = 0; j < V1.length; j++) {
			s = 0;
			for (int i = 0; i < V2.length; i++) {
				s += V2[i] * A.get(i, j);
			}
			V1[j] = s;
		}
		
	}*/
	
	/**
	 * Inner product of two vectors, i.e., <V1, V2>.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @return <V1, V2>
	 * 
	 */
	public static double innerProduct(double[] V1, double[] V2) {
		if (V1 == null || V2 == null) {
			return 0;
		}
		double res = 0;
		for (int i = 0; i < V1.length; i++) {
			res += V1[i] * V2[i];
		}
		return res;
	}
	
	/**
	 * Inner product of two vectors, i.e., <V1, V2>.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @param from beginning index (inclusive)
	 * 
	 * @param to ending index (exclusive)
	 * 
	 * @return \sum_{k = from}^{to} V1[k] * V2[k]
	 * 
	 */
	public static double innerProduct(double[] V1, double[] V2, int from, int to) {
		if (V1 == null || V2 == null) {
			return 0;
		}
		double res = 0;
		for (int i = from; i < to; i++) {
			res += V1[i] * V2[i];
		}
		return res;
	}
	
	/**
	 * Element-wise multiplication.
	 * It multiplies the first argument with the second argument,
	 * i.e., res = V1 .* V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @return V1 .* V2
	 * 
	 */
	public static double[] times(double[] V1, double[] V2) {
		double[] res = new double[V1.length];
		for (int i = 0; i < V1.length; i++)
			res[i] = V1[i] * V2[i];
		return res;
	}
	
	/**
	 * Element-wise addition.
	 * It adds the first argument with the second argument,
	 * i.e., res = V1 + V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @return V1 + V2
	 * 
	 */
	public static double[] plus(double[] V1, double[] V2) {
		double[] res = new double[V1.length];
		for (int i = 0; i < V1.length; i++)
			res[i] = V1[i] + V2[i];
		return res;
	}
	
	/**
	 * Element-wise subtraction.
	 * It subtracts the first argument by the second argument,
	 * i.e., res = V1 - V2.
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 * 
	 * @return V1 - V2
	 * 
	 */
	public static double[] minus(double[] V1, double[] V2) {
		double[] res = new double[V1.length];
		for (int i = 0; i < V1.length; i++)
			res[i] = V1[i] - V2[i];
		return res;
	}
	
}
