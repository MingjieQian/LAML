package ml.optimization;

import static la.utils.InPlaceOperator.assign;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;

/**
 * Compute prox_th(X) where h = lambda * || X ||_1.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 25th, 2014
 */
public class ProxL1 implements ProximalMapping {
	
	private double lambda;
	
	public ProxL1(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Compute prox_th(X) where h = lambda * || X ||_1.
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 * @return prox_th(X) where h = lambda * || X ||_1
	 * 
	 */
	@Override
	public Matrix compute(double t, Matrix X) {
		t *= lambda;
		Matrix res = X.copy();
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double v = 0;
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					v = resRow[j];
					if (v > t)
						resRow[j] = v - t;
					else if (v < -t) {
						resRow[j] = v + t;
					} else
						resRow[j] = 0;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				v = pr[k];
				if (v > t) {
					pr[k] = v - t;
				} else if (v < -t) {
					pr[k] = v + t;
				} else {
					pr[k] = 0;
				}
			}
			((SparseMatrix) res).clean();
		}
		return res;
	}

	/**
	 * res = prox_th(X) where h = lambda * || X ||_1.
	 * 
	 * @param res result matrix
	 * 
	 * @param t a nonnegative real scalar
	 * 
	 * @param X a real matrix
	 * 
	 */
	@Override
	public void compute(Matrix res, double t, Matrix X) {
		t *= lambda;
		assign(res, X);
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double v = 0;
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					v = resRow[j];
					if (v > t)
						resRow[j] = v - t;
					else if (v < -t) {
						resRow[j] = v + t;
					} else
						resRow[j] = 0;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				v = pr[k];
				if (v > t) {
					pr[k] = v - t;
				} else if (v < -t) {
					pr[k] = v + t;
				} else {
					pr[k] = 0;
				}
			}
			((SparseMatrix) res).clean();
		}
	}
}
