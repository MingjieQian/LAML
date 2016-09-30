package org.laml.ml.optimization;

import static org.laml.la.utils.InPlaceOperator.times;
import org.laml.la.matrix.Matrix;

/**
 * Compute prox_th(X) where h = lambda * || X ||_F^2.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProxL2Square implements ProximalMapping {
	
	private double lambda;
	
	public ProxL2Square(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Compute prox_th(X) where h = lambda * || X ||_F^2. For a 
	 * vector, || X ||_F is the l_2 norm of X, for a matrix
	 * || X ||_F is the Frobenius norm of X.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real column matrix
	 * 
	 * @return prox_th(X) where h = lambda * || X ||_F^2
	 * 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		t *= lambda;
		// Prox_th(X) = 1 / (1 + 2t) u
		Matrix res = X.copy();
		times(res, 1 / (1 + 2 * t), X);
		return res;
	}

	/**
	 * Compute prox_th(X) where h = lambda * || X ||_F^2. For a 
	 * vector, || X ||_F is the l_2 norm of X, for a matrix
	 * || X ||_F is the Frobenius norm of X.
	 * 
	 * @param res result matrix
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real column matrix
	 */
	@Override
	public void compute(Matrix res, double t, Matrix X) {
		t *= lambda;
		// Prox_th(X) = 1 / (1 + 2t) u
		times(res, 1 / (1 + 2 * t), X);
	}

}
