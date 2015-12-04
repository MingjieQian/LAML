package ml.optimization;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;

import static ml.utils.Matlab.*;
import static ml.utils.Printer.*;
import static ml.utils.InPlaceOperator.*;
import static ml.utils.ArrayOperator.*;

/**
 * Compute prox_th(X) where h = lambda * || X ||_{\infty}.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProxLInfinity implements ProximalMapping {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = new double[][]{
				{-3.5}, {2.4}, {1.2}, {-0.9}
		};
		Matrix X = new DenseMatrix(data);
		double t = 1.5;
		display(new ProxLInfinity(1).compute(t, X));
		
	}
	
	private double lambda;
	
	public ProxLInfinity(double lambda) {
		this.lambda = lambda;
	}
	
	/**
	 * Compute prox_th(X) where h = lambda * || X ||_{\infty}.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = lambda * || X ||_{\infty}
	 * 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		
		if (t < 0) {
			System.err.println("The first input should be a nonnegative real scalar.");
			System.exit(-1);
		}
		
		/*if (X.getColumnDimension() > 1) {
			System.err.println("The second input should be a vector.");
			System.exit(-1);
		}*/
		
		t *= lambda;
		int[] size = size(X);
		X = vec(X);
		
		Matrix res = X.copy();
		
		Matrix U = X.copy();
		Matrix V = abs(X);
		if (sum(sum(V)) <= t) {
			// res = zeros(size(V));
			res = zeros(size);
			return res;
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
		times(res, sign(V), min(alpha, abs(V)));
		// res = times(sign(V), min(alpha, abs(V)));

		res = reshape(res, size);
		return res;
	}

	/**
	 * Compute prox_th(X) where h = lambda * || X ||_{\infty}.
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
		
		/*if (X.getColumnDimension() > 1) {
			System.err.println("The second input should be a vector.");
			System.exit(-1);
		}*/
		
		t *= lambda;
		int[] size = size(X);
		X = vec(X);
		
		Matrix U = X.copy();
		Matrix V = abs(X);
		if (sum(sum(V)) <= t) {
			// res = zeros(size(V));
			res.clear();
			return;
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
		// times(res, sign(V), min(alpha, abs(V)));
		// res = times(sign(V), min(alpha, abs(V)));
		/*Matrix signV = sign(V);
		Matrix minAlphaAbsV = min(alpha, abs(V));*/
		
		if (size[1]== 1) {
			times(res, sign(V), min(alpha, abs(V)));
		} else {
			assign(res, reshape(sign(V).times(min(alpha, abs(V))), size));
		}
		
	}

}
