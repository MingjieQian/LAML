package ml.optimization;

import la.matrix.Matrix;
import static ml.utils.Matlab.*;
import static ml.utils.InPlaceOperator.*;

/**
 * Compute prox_th(X) where h = I_+(X).
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProxPlus implements ProximalMapping {

	/**
	 * Compute prox_th(X) where h = I_+(X).
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = I_+(X)
	 * 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		return subplus(X);
	}

	/**
	 * Compute prox_th(X) where h = I_+(X).
	 * 
	 * @param res result matrix
	 * 
	 * @param t a real scalar
	 * 
	 * @param X a real matrix
	 */
	@Override
	public void compute(Matrix res, double t, Matrix X) {
		subplus(res, X);
	}

}
