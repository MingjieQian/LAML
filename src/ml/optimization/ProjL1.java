package ml.optimization;

import static ml.utils.Matlab.sort;
import static ml.utils.Matlab.sum;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;

import static ml.utils.Matlab.*;
import static ml.utils.Printer.*;
import static ml.utils.InPlaceOperator.*;
import static ml.utils.ArrayOperator.*;

/**
 * Compute proj_tC(X) where C = {X: || X ||_1 <= 1}.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProjL1 implements Projection {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = new double[][]{
				{-3.5}, {2.4}, {1.2}, {-0.9}
		};
		Matrix X = new DenseMatrix(data);
		double t = 1.5;
		display(new ProjL1().compute(t, X));
		
	}
	
	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_1 <= 1}.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return proj_{tC}(X) where C = {X: || X ||_1 <= 1}
	 * 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		if (t < 0) {
			System.err.println("The first input should be a nonnegative real scalar.");
			System.exit(-1);
		}
		
		if (X.getColumnDimension() > 1) {
			System.err.println("The second input should be a vector.");
			System.exit(-1);
		}
		
		Matrix res = X.copy();
		
		Matrix U = X.copy();
		Matrix V = abs(X);
		if (sum(sum(V)) <= t) {
			res = zeros(size(V));
		}
		int d = size(X)[0];
		V = sort(V)[0];
		double[] Delta = new double[d - 1];
		for (int k = 0; k < d - 1; k++) {
			Delta[k] = V.getEntry(k + 1, 0) - V.getEntry(k, 0);
		}
		double[] S = times(Delta, colon(d - 1, -1, 1.0));
		double a = V.getEntry(d - 1, 0);
		double n = 1;
		double sum = S[d - 2];
		for (int j = d - 1; j >= 1; j--) {
			if (sum < t) {
				if (j > 1) {
					sum += S[j - 2];
				}
				a += V.getEntry(j - 1, 0);
				n++;
			} else {
				break;
			}
		}
		double alpha = (a - t) / n;
		V = U;
		minus(res, abs(V), alpha);
		subplusAssign(res);
		timesAssign(res, sign(V));
		// res = times(sign(V), subplus(minus(abs(V), alpha)));

		return res;
	}

	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_1 <= 1}.
	 * 
	 * @param res result matrix
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 */
	@Override
	public void compute(Matrix res, double t, Matrix X) {
		if (t < 0) {
			System.err.println("The first input should be a nonnegative real scalar.");
			System.exit(-1);
		}
		
		if (X.getColumnDimension() > 1) {
			System.err.println("The second input should be a vector.");
			System.exit(-1);
		}
		
		Matrix U = X.copy();
		Matrix V = abs(X);
		if (sum(sum(V)) <= t) {
			res = zeros(size(V));
		}
		int d = size(X)[0];
		V = sort(V)[0];
		double[] Delta = new double[d - 1];
		for (int k = 0; k < d - 1; k++) {
			Delta[k] = V.getEntry(k + 1, 0) - V.getEntry(k, 0);
		}
		double[] S = times(Delta, colon(d - 1, -1, 1.0));
		double a = V.getEntry(d - 1, 0);
		double n = 1;
		double sum = S[d - 2];
		for (int j = d - 1; j >= 1; j--) {
			if (sum < t) {
				if (j > 1) {
					sum += S[j - 2];
				}
				a += V.getEntry(j - 1, 0);
				n++;
			} else {
				break;
			}
		}
		double alpha = (a - t) / n;
		V = U;
		minus(res, abs(V), alpha);
		subplusAssign(res);
		timesAssign(res, sign(V));
		// res = times(sign(V), subplus(minus(abs(V), alpha)));
	}

}
