package ml.optimization;

import static ml.utils.InPlaceOperator.assign;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;

/***
 * Soft-thresholding (shrinkage) operator, which is defined as
 * S_{t}[x] = argmin_u 1/2 * || u - x ||^2 + t||u||_1</br>
 * which is actually prox_{t||.||_1}(x). The analytical form is</br>
 * S_{t}[x] =</br>
 * | x - t, if x > t</br>
 * | x + t, if x < -t</br>
 * | 0, otherwise</br>
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 24th, 2013
 */
public class ShrinkageOperator {

	/**
	 * Soft-thresholding (shrinkage) operator, which is defined as
	 * S_{t}[x] = argmin_u 1/2 * || u - x ||^2 + t||u||_1</br>
	 * which is actually prox_{t||.||_1}(x). The analytical form is</br>
	 * S_{t}[x] =</br>
	 * | x - t, if x > t</br>
	 * | x + t, if x < -t</br>
	 * | 0, otherwise</br>
	 * 
	 * @param X a real matrix
	 * 
	 * @param t threshold
	 * 
	 * @return argmin_u 1/2 * || u - x ||^2 + t||u||_1
	 */
	public static Matrix shrinkage(Matrix X, double t) {
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
	 * Soft-thresholding (shrinkage) operator, which is defined as
	 * S_{t}[x] = argmin_u 1/2 * || u - x ||^2 + t||u||_1</br>
	 * which is actually prox_{t||.||_1}(x). The analytical form is</br>
	 * S_{t}[x] =</br>
	 * | x - t, if x > t</br>
	 * | x + t, if x < -t</br>
	 * | 0, otherwise</br>
	 * 
	 * @param res result matrix
	 * 
	 * @param t threshold
	 * 
	 * @param X a real matrix
	 */
	public static void shrinkage(Matrix res, double t, Matrix X) {
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
	
	/**
	 * Soft-thresholding (shrinkage) operator, which is defined as
	 * S_{t}[x] = argmin_u 1/2 * || u - x ||^2 + t||u||_1</br>
	 * which is actually prox_{t||.||_1}(x). The analytical form is</br>
	 * S_{t}[x] =</br>
	 * | x - t, if x > t</br>
	 * | x + t, if x < -t</br>
	 * | 0, otherwise</br>
	 * 
	 * @param res result matrix
	 * 
	 * @param X a real matrix
	 * 
	 * @param t threshold
	 */
	public static void shrinkage(Matrix res, Matrix X, double t) {
		shrinkage(res, t, X);
	}
	
}
