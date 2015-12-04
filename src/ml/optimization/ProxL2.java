package ml.optimization;

import la.matrix.Matrix;
import static ml.utils.Matlab.*;
import static ml.utils.InPlaceOperator.*;

/**
 * Compute prox_th(X) where h = lambda * || X ||_F.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProxL2 implements ProximalMapping {
	
	private double lambda;
	
	public ProxL2(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Compute prox_th(X) where h = lambda * || X ||_F. For a 
	 * vector, || X ||_F is the l_2 norm of X, for a matrix
	 * || X ||_F is the Frobenius norm of X.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = lambda * || X ||_F
	 * 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		t *= lambda;
		// prox_th(X) = X - proj_{h*(X) <= t}(X)
		double norm = norm(X, "fro");
		if (norm <= t) {
			return zeros(size(X));
		} else {
			Matrix res = X.copy();
			times(res, 1 - t / norm, X);
			return res;
		}
	}

	/**
	 * Compute prox_th(X) where h = lambda * || X ||_F. For a 
	 * vector, || X ||_F is the l_2 norm of X, for a matrix
	 * || X ||_F is the Frobenius norm of X.
	 * 
	 * @param res result matrix
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 */
	@Override
	public void compute(Matrix res, double t, Matrix X) {
		t *= lambda;
		// prox_th(X) = X - proj_{h*(X) <= t}(X)
		double norm = norm(X, "fro");
		if (norm <= t) {
			res.clear();
		} else {
			times(res, 1 - t / norm, X);
		}
	}

}
