package ml.optimization;

import la.matrix.Matrix;
import static ml.utils.InPlaceOperator.assign;

/**
 * Compute prox_th(X) where h = 0.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class Prox implements ProximalMapping {

	/**
	 * Compute prox_th(X) where h = 0.
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = 0 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		return X;
	}

	/**
	 * res = prox_th(X) where h = 0.
	 * 
	 * @param res result matrix
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 */
	@Override
	public void compute(Matrix res, double t, Matrix X) {
		assign(res, X);
	}

}
