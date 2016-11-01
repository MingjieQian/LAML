package org.laml.ml.optimization;

import static org.laml.la.utils.InPlaceOperator.times;
import static org.laml.la.utils.Matlab.abs;
import static org.laml.la.utils.Matlab.min;
import static org.laml.la.utils.Matlab.sign;
import static org.laml.la.utils.Matlab.times;
import org.laml.la.matrix.Matrix;

/**
 * Compute proj_tC(X) where C = {X: || X ||_{\infty} <= 1}.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProjLInfinity implements Projection {

	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_{\infty} <= 1}.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real column matrix
	 * 
	 * @return proj_{tC}(X) where C = {X: || X ||_{\infty} <= 1}
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
		
		return times(sign(X), min(abs(X), t));
	}

	/**
	 * Compute proj_{tC}(X) where C = {X: || X ||_{\infty} <= 1}.
	 * 
	 * @param res result matrix
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real column matrix
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
		
		times(res, sign(X), min(abs(X), t));
	}

}
