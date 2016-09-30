package ml.optimization;

import la.matrix.Matrix;

/**
 * Compute prox_th(X), which is defined by
 * prox_th(x) := argmin_u 1/2 * || u - x ||^2 + t * h(u).
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 24th, 2014
 */
public interface ProximalMapping {
	
	/**
	 * Compute prox_th(X).
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X)
	 */
	public Matrix compute(double t, Matrix X);

	/**
	 * res = prox_th(X).
	 * 
	 * @param res result matrix
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 */
	public void compute(Matrix res, double t, Matrix X);

}
