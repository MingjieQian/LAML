package ml.utils;

import static ml.utils.Matlab.full;
import static ml.utils.Matlab.hilb;
import static ml.utils.Matlab.norm;
import static ml.utils.Matlab.sparse;
import static ml.utils.Printer.disp;
import static ml.utils.Printer.err;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Utility.exit;

import java.util.ArrayList;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;

/**
 * Memory allocation and garbage collection cost considerable
 * time. Thus a set of in-place functions without memory 
 * allocation are required for fast implementations. Most
 * operations assume that the arguments are dense matrices.
 *  
 * @author Mingjie Qian
 * @version 1.0 Dec. 20th, 2013
 */
public class InPlaceOperator {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int M = 5;
		int P = 5;
		int N = 5;
		Matrix A = hilb(M, P);
		A.setEntry(2, 0, 0);
		A.setEntry(2, 1, 0);
		A.setEntry(2, 2, 0);
		Matrix B = hilb(P, N);
		B.setEntry(1, 0, 0);
		B.setEntry(1, 1, 0);
		B.setEntry(1, 2, 0);
		B.setEntry(1, 4, 0);
		Matrix res = new DenseMatrix(M, N);

		fprintf("A * B:%n");
		disp(A.mtimes(B));
		fprintf("dense * dense:%n");
		mtimes(res, A, B);
		disp(res);
		fprintf("dense * sparse:%n");
		mtimes(res, A, sparse(B));
		disp(res);
		fprintf("sparse * dense:%n");
		mtimes(res, sparse(A), B);
		disp(res);
		fprintf("sparse * sparse:%n");
		mtimes(res, sparse(A), sparse(B));
		disp(res);

		Matrix ATB = A.transpose().mtimes(B);
		fprintf("A' * B:%n");
		disp(ATB);
		fprintf("dense' * dense:%n");
		mtimes(res, A, 'T', B);
		disp(norm(res.minus(ATB)));
		fprintf("dense' * sparse:%n");
		mtimes(res, A, 'T', sparse(B));
		disp(norm(res.minus(ATB)));
		fprintf("sparse' * dense:%n");
		mtimes(res, sparse(A), 'T', B);
		disp(norm(res.minus(ATB)));
		fprintf("sparse' * sparse:%n");
		mtimes(res, sparse(A), 'T', sparse(B));
		disp(norm(res.minus(ATB)));

		fprintf("A .* B:%n");
		disp(A.times(B));
		fprintf("dense .* dense:%n");
		times(res, A, B);
		disp(res);
		fprintf("dense .* sparse:%n");
		times(res, A, sparse(B));
		disp(res);
		fprintf("sparse .* dense:%n");
		times(res, sparse(A), B);
		disp(res);
		fprintf("sparse .* sparse:%n");
		times(res, sparse(A), sparse(B));
		disp(res);

		disp(A.times(B));
		res = A.copy();
		timesAssign(res, B);
		disp(res);
		res = A.copy();
		timesAssign(res, sparse(B));
		disp(res);

		int[] rIndices = new int[] {0, 1, 3, 1, 2, 2, 3, 2, 3};
		int[] cIndices = new int[] {0, 0, 0, 1, 1, 2, 2, 3, 3};
		double[] values = new double[] {10, 3.2, 3, 9, 7, 8, 8, 7, 7};
		int numRows = 5;
		int numColumns = 5;
		int nzmax = rIndices.length;

		A = new SparseMatrix(rIndices, cIndices, values, numRows, numColumns, nzmax);
		fprintf("A:%n");
		printMatrix(A);
		fprintf("A':%n");
		printMatrix(A.transpose());
		// printMatrix(A.transpose().transpose());

		printMatrix(A.plus(A.transpose()));
		plus(res, full(A), A.transpose());
		disp(res);

		printMatrix(A.minus(A.transpose()));
		minus(res, A, A.transpose());
		disp(res);

		B = A.transpose();
		double a = 0.5;
		double b = -1.5;

		fprintf("res = a * A + b * B%n");
		printMatrix(A.times(a).plus(B.times(b)));
		affine(res, a, A, b, B);
		disp(res);

		double c = 2.3;
		disp("res = A * B + c");
		res = A.mtimes(B).plus(c);
		disp(res);
		affine(res, A, B, c);
		disp(res);

		disp("*************************************");

		int dim = 5;
		Vector resV = new DenseVector(dim, 0);
		Vector V = new SparseVector(5);
		Vector U = new SparseVector(5);

		V.set(2, 3.5);
		V.set(4, 2.5);

		U.set(0, 0.5);
		U.set(2, 2.5);
		U.set(3, 1.5);

		double r = 0;
		affine(resV, a, V, b, U);
		disp(resV);

		r = norm(resV.minus(V.times(a).plus(U.times(b))));
		disp(r);
		
		disp("U:");
		disp(U);
		disp("a:");
		disp(a);
		disp("V:");
		disp(V);
		disp("U -= a * V for sparse V");
		Vector Ut = U.copy();
		minusAssign(Ut, a, V);
		disp("U:");
		disp(Ut);
		disp("U -= a * V for dense V");
		minusAssign(U, a, full(V));
		disp("U:");
		disp(U);

	}

	/**
	 * res = A * V.
	 * 
	 * @param res
	 * @param A
	 * @param V
	 */
	public static void operate(Vector res, Matrix A, Vector V) {
		int dim = V.getDim();
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (N != dim) {
			err("Dimension doesn't match.");
			exit(1);
		}
		if (res instanceof DenseVector) {
			double[] resPr = ((DenseVector) res).getPr();
			if (A instanceof DenseMatrix) {
				double[][] data = ((DenseMatrix) A).getData();
				if (V instanceof DenseVector) {
					ArrayOperator.operate(resPr, data, ((DenseVector) V).getPr());
				} else if (V instanceof SparseVector) {
					int[] ir = ((SparseVector) V).getIr();
					double[] pr = ((SparseVector) V).getPr();
					int nnz = ((SparseVector) V).getNNZ();
					int idx = 0;
					double[] row_i = null;
					for (int i = 0; i < M; i++) {
						row_i = data[i];
						double s = 0;
						for (int k = 0; k < nnz; k++) {
							idx = ir[k];
							s += row_i[idx] * pr[k];
						}
						resPr[i] = s;
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				if (V instanceof DenseVector) {
					double[] VPr = ((DenseVector) V).getPr();
					double s = 0;
					int c = 0;
					for (int r = 0; r < M; r++) {
						s = 0;
						for (int k = jr[r]; k < jr[r + 1]; k++) {
							c = ic[k];
							s += pr[valCSRIndices[k]] * VPr[c];
						}
						resPr[r] = s;
					}
				} else if (V instanceof SparseVector) {
					int[] ir = ((SparseVector) V).getIr();
					double[] VPr = ((SparseVector) V).getPr();
					int nnz = ((SparseVector) V).getNNZ();
					double s = 0;
					int kl = 0;
					int kr = 0;
					int cl = 0;
					int rr = 0;
					for (int i = 0; i < M; i++) {
						kl = jr[i];
						kr = 0;
						s = 0;
						while (true) {
							if (kl >= jr[i + 1] || kr >= nnz) {
								break;
							}
							cl = ic[kl];
							rr = ir[kr];
							if (cl < rr) {
								kl++;
							} else if (cl > rr) {
								kr++;
							} else {
								s += pr[valCSRIndices[kl]] * VPr[kr];
								kl++;
								kr++;
							}
						}
						resPr[i] = s;	
					}
				}
			}
		} else if (res instanceof SparseVector) {
			err("Sparse vector is not supported for res.");
			exit(1);
		}
	}

	/**
	 * res' = V' * A.
	 * 
	 * @param res
	 * @param V
	 * @param A
	 */
	public static void operate(Vector res, Vector V, Matrix A) {
		int dim = V.getDim();
		int M = A.getRowDimension();
		int N = A.getColumnDimension();
		if (M != dim) {
			err("Dimension doesn't match.");
			exit(1);
		}
		if (res instanceof DenseVector) {
			double[] resPr = ((DenseVector) res).getPr();
			if (A instanceof DenseMatrix) {
				clear(resPr);
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (V instanceof DenseVector) {
					double[] pr = ((DenseVector) V).getPr();
					double v = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						v = pr[i];
						for (int j = 0; j < N; j++) {
							resPr[j] += v * ARow[j];
						}
					}
				} else if (V instanceof SparseVector) {
					int[] ir = ((SparseVector) V).getIr();
					double[] pr = ((SparseVector) V).getPr();
					int nnz = ((SparseVector) V).getNNZ();
					double v = 0;
					for (int k = 0; k < nnz; k++) {
						int i = ir[k];
						ARow = AData[i];
						v = pr[k];
						for (int j = 0; j < N; j++) {
							resPr[j] += v * ARow[j];
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ir = ((SparseMatrix) A).getIr();
				int[] jc = ((SparseMatrix) A).getJc();
				double[] pr = ((SparseMatrix) A).getPr();
				if (V instanceof DenseVector) {
					clear(resPr);
					double[] VPr = ((DenseVector) V).getPr();
					for (int j = 0; j < N; j++) {
						for (int k = jc[j]; k < jc[j + 1]; k++) {
							resPr[j] += VPr[ir[k]] * pr[k];
						}
					}
				} else if (V instanceof SparseVector) {
					int[] VIr = ((SparseVector) V).getIr();
					double[] VPr = ((SparseVector) V).getPr();
					int nnz = ((SparseVector) V).getNNZ();
					double s = 0;
					int k1 = 0;
					int k2 = 0;
					int c = 0;
					int r = 0;
					for (int j = 0; j < N; j++) {
						k1 = 0;
						k2 = jc[j];
						s = 0;
						while (true) {
							if (k2 >= jc[j + 1] || k1 >= nnz) {
								break;
							}
							c = VIr[k1];
							r = ir[k2];
							if (r < c) {
								k2++;
							} else if (r > c) {
								k1++;
							} else {
								s += VPr[k1] * pr[k2];
								k1++;
								k2++;
							}
						}
						resPr[j] = s;
					}	
				}
			}
		} else if (res instanceof SparseVector) {
			err("Sparse vector is not supported for res.");
			exit(1);
		}
	}

	/**
	 * res = abs(A).
	 * 
	 * @param res
	 * @param A
	 */
	public static void abs(Matrix res, Matrix A) {
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < nRow; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < nCol; j++) {
						resRow[j] = Math.abs(ARow[j]);
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < nRow; i++) {
					resRow = resData[i];
					clear(resRow);
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						resRow[ic[k]] = Math.abs(pr[valCSRIndices[k]]);
					}
				}
			}
		} else {
			((SparseMatrix) res).assignSparseMatrix((SparseMatrix) Matlab.abs(A));
			/*err("");
			exit(1);*/
		}
	}

	/**
	 * res = subplus(A).
	 * 
	 * @param res
	 * @param A
	 */
	public static void subplus(Matrix res, Matrix A) {
		assign(res, A);
		subplusAssign(res);
	}
	
	/**
	 * res = subplus(V)
	 * 
	 * @param res
	 * @param V
	 */
	public static void subplus(double[] res, double[] V) {
		for (int i = 0; i < res.length; i++) {
			double v = V[i];
			res[i] = v >= 0 ? v : 0;
		}
	}
	
	/**
	 * res = subplus(res)
	 * 
	 * @param res
	 */
	public static void subplusAssign(double[] res) {
		subplus(res, res);
	}

	/**
	 * res = subplus(res).
	 * 
	 * @param res
	 */
	public static void subplusAssign(Matrix res) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					if (resRow[j] < 0)
						resRow[j] = 0;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				if (pr[k] < 0)
					pr[k] = 0;
			}
			((SparseMatrix) res).clean();
		}
	}

	/**
	 * res = A | B.
	 * 
	 * @param res
	 * @param A
	 * @param B
	 */
	public static void or(Matrix res, Matrix A, Matrix B) {
		double[][] resData = null;
		if (res instanceof DenseMatrix) {
			resData = ((DenseMatrix) res).getData();
		} else {
			System.err.println("res should be a dense matrix.");
			System.exit(1);
		}
		double[][] AData = null;
		if (A instanceof DenseMatrix) {
			AData = ((DenseMatrix) A).getData();
		} else {
			System.err.println("A should be a dense matrix.");
			System.exit(1);
		}
		double[][] BData = null;
		if (B instanceof DenseMatrix) {
			BData = ((DenseMatrix) B).getData();
		} else {
			System.err.println("B should be a dense matrix.");
			System.exit(1);
		}
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		double[] resRow = null;
		double[] ARow = null;
		double[] BRow = null;
		for (int i = 0; i < M; i++) {
			resRow = resData[i];
			ARow = AData[i];
			BRow = BData[i];
			for (int j = 0; j < N; j++) {
				resRow[j] = ARow[j] + BRow[j] >= 1 ? 1 : 0;
			}
		}
	}
	
	/**
	 * res = a * U + b * V.
	 * 
	 * @param res
	 * @param a
	 * @param U
	 * @param b
	 * @param V
	 */
	public static void affine(double[] res, double a, double[] U, double b, double[] V) {
		for (int i = 0; i < res.length; i++)
			res[i] = a * U[i] + b * V[i];
	}

	/**
	 * res = a * V + b * U.
	 * 
	 * @param res
	 * @param a
	 * @param V
	 * @param b
	 * @param U
	 */
	public static void affine(Vector res, double a, Vector V, double b, Vector U) {
		if (b == 0) {
			times(res, a, V);
			return;
		} else if (b == 1) {
			affine(res, a, V, '+', U);
			return;
		} else if (b == -1) {
			affine(res, a, V, '-', U);
			return;
		}
		if (a == 0) {
			times(res, b, U);
			return;
		} else if (a == 1) {
			affine(res, b, U, '+', V);
			return;
		} else if (a == -1) {
			affine(res, b, U, '-', V);
			return;
		}
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) { // res = a * V + b * U
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					for (int i = 0; i < dim; i++) {
						resData[i] = a * VData[i] + b * UData[i];
					}
				} else if (U instanceof SparseVector) {
					int[] ir = ((SparseVector) U).getIr();
					double[] pr = ((SparseVector) U).getPr();
					int nnz = ((SparseVector) U).getNNZ();
					int lastIdx = -1;
					int currentIdx = 0;
					for (int k = 0; k < nnz; k++) {
						currentIdx = ir[k];
						for (int i = lastIdx + 1; i < currentIdx; i++) {
							resData[i] = a * VData[i];
						}
						resData[currentIdx] = a * VData[currentIdx] + b * pr[k];
						lastIdx = currentIdx;
					}
					for (int i = lastIdx + 1; i < dim; i++) {
						resData[i] = a * VData[i];
					}
				}
			} else if (V instanceof SparseVector) { // res = a * V +b * U
				int[] ir1 = ((SparseVector) V).getIr();
				double[] pr1 = ((SparseVector) V).getPr();
				int nnz1 = ((SparseVector) V).getNNZ();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					int lastIdx = -1;
					int currentIdx = 0;
					for (int k = 0; k < nnz1; k++) {
						currentIdx = ir1[k];
						for (int i = lastIdx + 1; i < currentIdx; i++) {
							resData[i] = b * UData[i];
						}
						resData[currentIdx] = a * pr1[k] + b * UData[currentIdx];
						lastIdx = currentIdx;
					}
					for (int i = lastIdx + 1; i < dim; i++) {
						resData[i] = b * UData[i];
					}
				} else if (U instanceof SparseVector) { // res = a * V + b * U
					int[] ir2 = ((SparseVector) U).getIr();
					double[] pr2 = ((SparseVector) U).getPr();
					int nnz2 = ((SparseVector) U).getNNZ();
					clear(resData);
					if (!(nnz1 == 0 && nnz2 == 0)) {
						int k1 = 0;
						int k2 = 0;
						int r1 = 0;
						int r2 = 0;
						double v = 0;
						int i = -1;
						while (k1 < nnz1 || k2 < nnz2) {
							if (k2 == nnz2) { // V has been processed.
								i = ir1[k1];
								v = a * pr1[k1];
								k1++;
							} else if (k1 == nnz1) { // this has been processed.
								i = ir2[k2];
								v = b * pr2[k2];
								k2++;
							} else { // Both this and V have not been fully processed.
								r1 = ir1[k1];
								r2 = ir2[k2];
								if (r1 < r2) {
									i = r1;
									v = a * pr1[k1];
									k1++;
								} else if (r1 == r2) {
									i = r1;
									v = a * pr1[k1] + b * pr2[k2];
									k1++;
									k2++;
								} else {
									i = r2;
									v = b * pr2[k2];
									k2++;
								}
							}
							if (v != 0) {
								resData[i] = v;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = a * V + U if operator is '+',</br>
	 * res = a * V - U if operator is '-'.
	 * 
	 * @param res
	 * @param a
	 * @param V
	 * @param operator a {@code char} variable: '+' or '-'
	 * @param U
	 */
	public static void affine(Vector res, double a, Vector V, char operator, Vector U) {
		if (operator == '+') {
			if (a == 0) {
				assign(res, U);
				return;
			} else if (a == 1) {
				plus(res, V, U);
				return;
			} else if (a == -1) {
				minus(res, U, V);
				return;
			}
			int dim = res.getDim();
			if (res instanceof SparseVector) {

			} else if (res instanceof DenseVector) { // res = a * V + U
				double[] resData = ((DenseVector) res).getPr();
				if (V instanceof DenseVector) {
					double[] VData = ((DenseVector) V).getPr();
					if (U instanceof DenseVector) {
						double[] UData = ((DenseVector) U).getPr();
						for (int i = 0; i < dim; i++) {
							resData[i] = a * VData[i] + UData[i];
						}
					} else if (U instanceof SparseVector) {
						int[] ir = ((SparseVector) U).getIr();
						double[] pr = ((SparseVector) U).getPr();
						int nnz = ((SparseVector) U).getNNZ();
						int lastIdx = -1;
						int currentIdx = 0;
						for (int k = 0; k < nnz; k++) {
							currentIdx = ir[k];
							for (int i = lastIdx + 1; i < currentIdx; i++) {
								resData[i] = a * VData[i];
							}
							resData[currentIdx] = a * VData[currentIdx] + pr[k];
							lastIdx = currentIdx;
						}
						for (int i = lastIdx + 1; i < dim; i++) {
							resData[i] = a * VData[i];
						}
					}
				} else if (V instanceof SparseVector) { // res = a * V + U
					int[] ir1 = ((SparseVector) V).getIr();
					double[] pr1 = ((SparseVector) V).getPr();
					int nnz1 = ((SparseVector) V).getNNZ();
					if (U instanceof DenseVector) {
						double[] UData = ((DenseVector) U).getPr();
						int lastIdx = -1;
						int currentIdx = 0;
						for (int k = 0; k < nnz1; k++) {
							currentIdx = ir1[k];
							for (int i = lastIdx + 1; i < currentIdx; i++) {
								resData[i] = UData[i];
							}
							resData[currentIdx] = a * pr1[k] + UData[currentIdx];
							lastIdx = currentIdx;
						}
						for (int i = lastIdx + 1; i < dim; i++) {
							resData[i] = UData[i];
						}
					} else if (U instanceof SparseVector) { // res = a * V + U
						int[] ir2 = ((SparseVector) U).getIr();
						double[] pr2 = ((SparseVector) U).getPr();
						int nnz2 = ((SparseVector) U).getNNZ();
						clear(resData);
						if (!(nnz1 == 0 && nnz2 == 0)) {
							int k1 = 0;
							int k2 = 0;
							int r1 = 0;
							int r2 = 0;
							double v = 0;
							int i = -1;
							while (k1 < nnz1 || k2 < nnz2) {
								if (k2 == nnz2) { // V has been processed.
									i = ir1[k1];
									v = a * pr1[k1];
									k1++;
								} else if (k1 == nnz1) { // this has been processed.
									i = ir2[k2];
									v = pr2[k2];
									k2++;
								} else { // Both this and V have not been fully processed.
									r1 = ir1[k1];
									r2 = ir2[k2];
									if (r1 < r2) {
										i = r1;
										v = a * pr1[k1];
										k1++;
									} else if (r1 == r2) {
										i = r1;
										v = a * pr1[k1] + pr2[k2];
										k1++;
										k2++;
									} else {
										i = r2;
										v = pr2[k2];
										k2++;
									}
								}
								if (v != 0) {
									resData[i] = v;
								}
							}
						}
					}
				}
			}
		} else if (operator == '-') {
			if (a == 0) {
				uminus(res, U);
				return;
			} else if (a == 1) {
				minus(res, V, U);
				return;
			}
			int dim = res.getDim();
			if (res instanceof SparseVector) {

			} else if (res instanceof DenseVector) { // res = a * V - U
				double[] resData = ((DenseVector) res).getPr();
				if (V instanceof DenseVector) {
					double[] VData = ((DenseVector) V).getPr();
					if (U instanceof DenseVector) {
						double[] UData = ((DenseVector) U).getPr();
						for (int i = 0; i < dim; i++) {
							resData[i] = a * VData[i] - UData[i];
						}
					} else if (U instanceof SparseVector) {
						int[] ir = ((SparseVector) U).getIr();
						double[] pr = ((SparseVector) U).getPr();
						int nnz = ((SparseVector) U).getNNZ();
						int lastIdx = -1;
						int currentIdx = 0;
						for (int k = 0; k < nnz; k++) {
							currentIdx = ir[k];
							for (int i = lastIdx + 1; i < currentIdx; i++) {
								resData[i] = a * VData[i];
							}
							resData[currentIdx] = a * VData[currentIdx] - pr[k];
							lastIdx = currentIdx;
						}
						for (int i = lastIdx + 1; i < dim; i++) {
							resData[i] = a * VData[i];
						}
					}
				} else if (V instanceof SparseVector) {
					int[] ir1 = ((SparseVector) V).getIr();
					double[] pr1 = ((SparseVector) V).getPr();
					int nnz1 = ((SparseVector) V).getNNZ();
					if (U instanceof DenseVector) { // res = a * V - U
						double[] UData = ((DenseVector) U).getPr();
						int lastIdx = -1;
						int currentIdx = 0;
						for (int k = 0; k < nnz1; k++) {
							currentIdx = ir1[k];
							for (int i = lastIdx + 1; i < currentIdx; i++) {
								resData[i] = -UData[i];
							}
							resData[currentIdx] = a * pr1[k] - UData[currentIdx];
							lastIdx = currentIdx;
						}
						for (int i = lastIdx + 1; i < dim; i++) {
							resData[i] = -UData[i];
						}
					} else if (U instanceof SparseVector) { // res = a * V - U
						int[] ir2 = ((SparseVector) U).getIr();
						double[] pr2 = ((SparseVector) U).getPr();
						int nnz2 = ((SparseVector) U).getNNZ();
						clear(resData);
						if (!(nnz1 == 0 && nnz2 == 0)) {
							int k1 = 0;
							int k2 = 0;
							int r1 = 0;
							int r2 = 0;
							double v = 0;
							int i = -1;
							while (k1 < nnz1 || k2 < nnz2) {
								if (k2 == nnz2) { // V has been processed.
									i = ir1[k1];
									v = a * pr1[k1];
									k1++;
								} else if (k1 == nnz1) { // this has been processed.
									i = ir2[k2];
									v = -pr2[k2];
									k2++;
								} else { // Both this and V have not been fully processed.
									r1 = ir1[k1];
									r2 = ir2[k2];
									if (r1 < r2) {
										i = r1;
										v = a * pr1[k1];
										k1++;
									} else if (r1 == r2) {
										i = r1;
										v = a * pr1[k1] - pr2[k2];
										k1++;
										k2++;
									} else {
										i = r2;
										v = -pr2[k2];
										k2++;
									}
								}
								if (v != 0) {
									resData[i] = v;
								}
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = V + b * U.
	 * 
	 * @param res
	 * @param V
	 * @param b
	 * @param U
	 */
	public static void affine(Vector res, Vector V, double b, Vector U) {
		affine(res, b, U, '+', V);
	}
	
	/**
	 * res = U + a * V.
	 * 
	 * @param res
	 * @param U
	 * @param a
	 * @param V
	 */
	public static void affine(double[] res, double[] U, double a, double[] V) {
		for (int i = 0; i < res.length; i++)
			res[i] = U[i] + a * V[i];
	}

	/**
	 * res = -res.
	 * 
	 * @param res
	 */
	public static void uminusAssign(Vector res) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			double[] pr = ((SparseVector) res).getPr();
			int nnz = ((SparseVector) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] = -pr[k];
			}
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			for (int i = 0; i < dim; i++) {
				resData[i] = -resData[i];
			}
		}
	}

	/**
	 * res = -V.
	 * 
	 * @param res
	 * @param V
	 */
	public static void uminus(Vector res, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] = -VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = 0;
					}
					resData[currentIdx] = -pr[k];
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = 0;
				}
			}
		}
	}

	/**
	 * res = V .* U.
	 * 
	 * @param res
	 * @param V
	 * @param U
	 */
	public static void times(Vector res, Vector V, Vector U) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					for (int i = 0; i < dim; i++) {
						resData[i] = VData[i] * UData[i];
					}
				} else if (U instanceof SparseVector) {
					int[] ir = ((SparseVector) U).getIr();
					double[] pr = ((SparseVector) U).getPr();
					int nnz = ((SparseVector) U).getNNZ();
					int idx = -1;
					res.clear();
					for (int k = 0; k < nnz; k++) {
						idx = ir[k];
						resData[idx] = VData[idx] * pr[k];
					}
				}
			} else if (V instanceof SparseVector) {
				int[] ir1 = ((SparseVector) V).getIr();
				double[] pr1 = ((SparseVector) V).getPr();
				int nnz1 = ((SparseVector) V).getNNZ();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					int lastIdx = -1;
					int currentIdx = 0;
					for (int k = 0; k < nnz1; k++) {
						currentIdx = ir1[k];
						for (int i = lastIdx + 1; i < currentIdx; i++) {
							resData[i] = 0;
						}
						resData[currentIdx] = pr1[k] * UData[currentIdx];
						lastIdx = currentIdx;
					}
					for (int i = lastIdx + 1; i < dim; i++) {
						resData[i] = 0;
					}
				} else if (U instanceof SparseVector) {
					int[] ir2 = ((SparseVector) U).getIr();
					double[] pr2 = ((SparseVector) U).getPr();
					int nnz2 = ((SparseVector) U).getNNZ();
					res.clear();
					if (nnz1 != 0 && nnz2 != 0) {
						int k1 = 0;
						int k2 = 0;
						int r1 = 0;
						int r2 = 0;
						double v = 0;
						int i = -1;
						while (k1 < nnz1 && k2 < nnz2) {
							r1 = ir1[k1];
							r2 = ir2[k2];
							if (r1 < r2) {
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = pr1[k1] * pr2[k2];
								k1++;
								k2++;
								if (v != 0) {
									resData[i] = v;
								}
							} else {
								k2++;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = v * V.
	 * 
	 * @param res
	 * @param v
	 * @param V
	 */
	public static void times(Vector res, double v, Vector V) {
		if (v == 0) {
			res.clear();
			return;
		} else if (v == 1) {
			assign(res, V);
			return;
		} else if (v == -1) {
			uminus(res, V);
			return;
		}
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] = v * VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = 0;
					}
					resData[currentIdx] = v * pr[k];
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = 0;
				}
			}
		}
	}

	/**
	 * res *= V.
	 * 
	 * @param res
	 * @param V
	 */
	public static void timesAssign(Vector res, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] *= VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = 0;
					}
					resData[currentIdx] *= pr[k];
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = 0;
				}
			}
		}
	}

	/**
	 * res *= v.
	 * 
	 * @param res
	 * @param v
	 */
	public static void timesAssign(Vector res, double v) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			if (v == 0) {
				res.clear();
				return;
			}
			double[] pr = ((SparseVector) res).getPr();
			int nnz = ((SparseVector) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] *= v;
			}
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			for (int i = 0; i < dim; i++) {
				resData[i] *= v;
			}
		}
	}

	/**
	 * res = V.
	 * 
	 * @param res
	 * @param V
	 */
	public static void assign(Vector res, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			((SparseVector) res).assignSparseVector((SparseVector) V);
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				System.arraycopy(VData, 0, resData, 0, dim);
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = 0;
					}
					resData[currentIdx] = pr[k];
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = 0;
				}
			}
		}
	}

	/**
	 * res = V - U.
	 * 
	 * @param res
	 * @param V
	 * @param U
	 */
	public static void minus(Vector res, Vector V, Vector U) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					for (int i = 0; i < dim; i++) {
						resData[i] = VData[i] - UData[i];
					}
				} else if (U instanceof SparseVector) {
					int[] ir = ((SparseVector) U).getIr();
					double[] pr = ((SparseVector) U).getPr();
					int nnz = ((SparseVector) U).getNNZ();
					System.arraycopy(VData, 0, resData, 0, dim);
					for (int k = 0; k < nnz; k++) {
						resData[ir[k]] -= pr[k];
					}
				}
			} else if (V instanceof SparseVector) {
				int[] ir1 = ((SparseVector) V).getIr();
				double[] pr1 = ((SparseVector) V).getPr();
				int nnz1 = ((SparseVector) V).getNNZ();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					int lastIdx = -1;
					int currentIdx = 0;
					for (int k = 0; k < nnz1; k++) {
						currentIdx = ir1[k];
						for (int i = lastIdx + 1; i < currentIdx; i++) {
							resData[i] = -UData[i];
						}
						resData[currentIdx] = pr1[k] - UData[currentIdx];
						lastIdx = currentIdx;
					}
					for (int i = lastIdx + 1; i < dim; i++) {
						resData[i] = -UData[i];
					}
				} else if (U instanceof SparseVector) {
					int[] ir2 = ((SparseVector) U).getIr();
					double[] pr2 = ((SparseVector) U).getPr();
					int nnz2 = ((SparseVector) U).getNNZ();
					clear(resData);
					if (!(nnz1 == 0 && nnz2 == 0)) {
						int k1 = 0;
						int k2 = 0;
						int r1 = 0;
						int r2 = 0;
						double v = 0;
						int i = -1;
						while (k1 < nnz1 || k2 < nnz2) {
							if (k2 == nnz2) { // V has been processed.
								i = ir1[k1];
								v = pr1[k1];
								k1++;
							} else if (k1 == nnz1) { // this has been processed.
								i = ir2[k2];
								v = -pr2[k2];
								k2++;
							} else { // Both this and V have not been fully processed.
								r1 = ir1[k1];
								r2 = ir2[k2];
								if (r1 < r2) {
									i = r1;
									v = pr1[k1];
									k1++;
								} else if (r1 == r2) {
									i = r1;
									v = pr1[k1] - pr2[k2];
									k1++;
									k2++;
								} else {
									i = r2;
									v = -pr2[k2];
									k2++;
								}
							}
							if (v != 0) {
								resData[i] = v;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = V1 - V2.
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 */
	public static void minus(double[] res, double[] V1, double[] V2) {
		for (int i = 0; i < res.length; i++)
			res[i] = V1[i] - V2[i];
	}
	
	/**
	 * res = V - v.
	 * 
	 * @param res
	 * @param V
	 * @param v
	 */
	public static void minus(Vector res, Vector V, double v) {
		int dim = res.getDim();
		double minusv = -v;
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] = VData[i] - v;
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = minusv;
					}
					resData[currentIdx] = pr[k] - v;
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = minusv;
				}
			}
		}
	}

	/**
	 * res = v - V.
	 * 
	 * @param res
	 * @param v
	 * @param V
	 */
	public static void minus(Vector res, double v, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] = v - VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = v;
					}
					resData[currentIdx] = v - pr[k];
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = v;
				}
			}
		}
	}

	/**
	 * res -= v.
	 * 
	 * @param res
	 * @param v
	 */
	public static void minusAssign(Vector res, double v) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			for (int i = 0; i < dim; i++) {
				resData[i] -= v;
			}
		}
	}

	/**
	 * res -= V
	 * @param res
	 * @param V
	 */
	public static void minusAssign(Vector res, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			((SparseVector) res).assignSparseVector((SparseVector) res.minus(V));
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] -= VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				for (int k = 0; k < nnz; k++) {
					resData[ir[k]] -= pr[k];
				}
			}
		}
	}

	/**
	 * res -= a * V.
	 * 
	 * @param res
	 * @param a
	 * @param V
	 */
	public static void minusAssign(Vector res, double a, Vector V) {
		if (a == 0) {
			return;
		}
		if (a == 1) {
			minusAssign(res, V);
			return;
		} else if (a == -1) {
			plusAssign(res, V);
			return;
		}
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			plusAssign(res, -a, V);
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] -= a * VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				for (int k = 0; k < nnz; k++) {
					resData[ir[k]] -= a * pr[k];
				}
			}
		}
	}

	/**
	 * res = V + U.
	 * 
	 * @param res
	 * @param V
	 * @param U
	 */
	public static void plus(Vector res, Vector V, Vector U) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			((SparseVector) res).assignSparseVector((SparseVector) V.plus(U));
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					for (int i = 0; i < dim; i++) {
						resData[i] = VData[i] + UData[i];
					}
				} else if (U instanceof SparseVector) {
					int[] ir = ((SparseVector) U).getIr();
					double[] pr = ((SparseVector) U).getPr();
					int nnz = ((SparseVector) U).getNNZ();
					System.arraycopy(VData, 0, resData, 0, dim);
					for (int k = 0; k < nnz; k++) {
						resData[ir[k]] += pr[k];
					}
				}
			} else if (V instanceof SparseVector) {
				int[] ir1 = ((SparseVector) V).getIr();
				double[] pr1 = ((SparseVector) V).getPr();
				int nnz1 = ((SparseVector) V).getNNZ();
				if (U instanceof DenseVector) {
					double[] UData = ((DenseVector) U).getPr();
					int lastIdx = -1;
					int currentIdx = 0;
					for (int k = 0; k < nnz1; k++) {
						currentIdx = ir1[k];
						for (int i = lastIdx + 1; i < currentIdx; i++) {
							resData[i] = UData[i];
						}
						resData[currentIdx] = pr1[k] + UData[currentIdx];
						lastIdx = currentIdx;
					}
					for (int i = lastIdx + 1; i < dim; i++) {
						resData[i] = UData[i];
					}
				} else if (U instanceof SparseVector) {
					int[] ir2 = ((SparseVector) U).getIr();
					double[] pr2 = ((SparseVector) U).getPr();
					int nnz2 = ((SparseVector) U).getNNZ();
					clear(resData);
					if (!(nnz1 == 0 && nnz2 == 0)) {
						int k1 = 0;
						int k2 = 0;
						int r1 = 0;
						int r2 = 0;
						double v = 0;
						int i = -1;
						while (k1 < nnz1 || k2 < nnz2) {
							if (k2 == nnz2) { // V has been processed.
								i = ir1[k1];
								v = pr1[k1];
								k1++;
							} else if (k1 == nnz1) { // this has been processed.
								i = ir2[k2];
								v = pr2[k2];
								k2++;
							} else { // Both this and V have not been fully processed.
								r1 = ir1[k1];
								r2 = ir2[k2];
								if (r1 < r2) {
									i = r1;
									v = pr1[k1];
									k1++;
								} else if (r1 == r2) {
									i = r1;
									v = pr1[k1] + pr2[k2];
									k1++;
									k2++;
								} else {
									i = r2;
									v = pr2[k2];
									k2++;
								}
							}
							if (v != 0) {
								resData[i] = v;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = V1 + V2.
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param V1 a 1D {@code double} array
	 * 
	 * @param V2 a 1D {@code double} array
	 */
	public static void plus(double[] res, double[] V1, double[] V2) {
		for (int i = 0; i < res.length; i++)
			res[i] = V1[i] + V2[i];
	}
	
	/**
	 * res = V + v.
	 * 
	 * @param res
	 * @param V
	 * @param v
	 */
	public static void plus(Vector res, Vector V, double v) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] = VData[i] + v;
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = v;
					}
					resData[currentIdx] = pr[k] + v;
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = v;
				}
			}
		}
	}

	/**
	 * res = v + V.
	 * 
	 * @param res
	 * @param v
	 * @param V
	 */
	public static void plus(Vector res, double v, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] = v + VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				int lastIdx = -1;
				int currentIdx = 0;
				for (int k = 0; k < nnz; k++) {
					currentIdx = ir[k];
					for (int i = lastIdx + 1; i < currentIdx; i++) {
						resData[i] = v;
					}
					resData[currentIdx] = v + pr[k];
					lastIdx = currentIdx;
				}
				for (int i = lastIdx + 1; i < dim; i++) {
					resData[i] = v;
				}
			}
		}
	}

	/**
	 * res += v.
	 * 
	 * @param res
	 * @param v
	 */
	public static void plusAssign(Vector res, double v) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {

		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			for (int i = 0; i < dim; i++) {
				resData[i] += v;
			}
		}
	}

	/**
	 * res += V
	 * @param res
	 * @param V
	 */
	public static void plusAssign(Vector res, Vector V) {
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			((SparseVector) res).assignSparseVector((SparseVector) res.plus(V));
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] += VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				for (int k = 0; k < nnz; k++) {
					resData[ir[k]] += pr[k];
				}
			}
		}
	}

	/**
	 * res += a * V.
	 * 
	 * @param res
	 * @param a
	 * @param V
	 */
	public static void plusAssign(Vector res, double a, Vector V) {
		if (a == 0) {
			return;
		}
		if (a == 1) {
			plusAssign(res, V);
			return;
		} else if (a == -1) {
			minusAssign(res, V);
			return;
		}
		int dim = res.getDim();
		if (res instanceof SparseVector) {
			if (V instanceof DenseVector) {
				ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
				int[] ir1 = ((SparseVector) res).getIr();
				double[] pr1 = ((SparseVector) res).getPr();
				int nnz1 = ((SparseVector) res).getNNZ();
				double[] pr2 = ((DenseVector) V).getPr();
				int idx = -1;
				int i = -1;
				double v = 0;
				int lastIdx = -1;
				for (int k = 0; k < nnz1; k++) {
					idx = ir1[k];
					for (int r = lastIdx + 1; r < idx; r++) {
						i = r;
						v = a * pr2[i];
						if (v != 0)
							list.add(Pair.of(i, v));
					}
					i = idx;
					v = pr1[k] + a * pr2[i];
					if (v != 0)
						list.add(Pair.of(i, v));
					lastIdx = idx;
				}
				for (int r = lastIdx + 1; r < dim; r++) {
					i = r;
					v = a * pr2[i];
					if (v != 0)
						list.add(Pair.of(i, v));
				}
				int nnz = list.size();
				int[] ir_res = new int[nnz];
				double[] pr_res = new double[nnz];
				int k = 0;
				for (Pair<Integer, Double> pair : list) {
					ir_res[k] = pair.first;
					pr_res[k] = pair.second;
					k++;
				}
				((SparseVector) res).assignSparseVector(new SparseVector(ir_res, pr_res, nnz, dim));
			} else if (V instanceof SparseVector) {
				ArrayList<Pair<Integer, Double>> list = new ArrayList<Pair<Integer, Double>>();
				int[] ir1 = ((SparseVector) res).getIr();
				double[] pr1 = ((SparseVector) res).getPr();
				int nnz1 = ((SparseVector) res).getNNZ();
				int[] ir2 = ((SparseVector) V).getIr();
				double[] pr2 = ((SparseVector) V).getPr();
				int nnz2 = ((SparseVector) V).getNNZ();
				if (!(nnz1 == 0 && nnz2 == 0)) {
					int k1 = 0;
					int k2 = 0;
					int r1 = 0;
					int r2 = 0;
					double v = 0;
					int i = -1;
					while (k1 < nnz1 || k2 < nnz2) {
						if (k2 == nnz2) { // V has been processed.
							i = ir1[k1];
							v = pr1[k1];
							k1++;
						} else if (k1 == nnz1) { // this has been processed.
							i = ir2[k2];
							v = a * pr2[k2];
							k2++;
						} else { // Both this and V have not been fully processed.
							r1 = ir1[k1];
							r2 = ir2[k2];
							if (r1 < r2) {
								i = r1;
								v = pr1[k1];
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = pr1[k1] + a * pr2[k2];
								k1++;
								k2++;
							} else {
								i = r2;
								v = a * pr2[k2];
								k2++;
							}
						}
						if (v != 0) {
							list.add(Pair.of(i, v));
						}
					}
				}
				int nnz = list.size();
				int[] ir_res = new int[nnz];
				double[] pr_res = new double[nnz];
				int k = 0;
				for (Pair<Integer, Double> pair : list) {
					ir_res[k] = pair.first;
					pr_res[k] = pair.second;
					k++;
				}
				((SparseVector) res).assignSparseVector(new SparseVector(ir_res, pr_res, nnz, dim));
			}
		} else if (res instanceof DenseVector) {
			double[] resData = ((DenseVector) res).getPr();
			if (V instanceof DenseVector) {
				double[] VData = ((DenseVector) V).getPr();
				for (int i = 0; i < dim; i++) {
					resData[i] += a * VData[i];
				}
			} else if (V instanceof SparseVector) {
				int[] ir = ((SparseVector) V).getIr();
				double[] pr = ((SparseVector) V).getPr();
				int nnz = ((SparseVector) V).getNNZ();
				for (int k = 0; k < nnz; k++) {
					resData[ir[k]] += a * pr[k];
				}
			}
		}
	}

	/**
	 * res = A * B if operator is ' ',</br>
	 * res = A<sup>T</sup> * B if operator is 'T'.
	 * 
	 * @param res
	 * @param A
	 * @param operator a {@code char} variable: 'T' or ' '
	 * @param B
	 */
	public static void mtimes(Matrix res, Matrix A, char operator, Matrix B) {
		if (operator == ' ') {
			mtimes(res, A, B);
		} else if (operator == 'T') {

			if (res instanceof SparseMatrix) {
				((SparseMatrix) res).assignSparseMatrix(sparse(A.transpose().mtimes(B)));
			} else if (res instanceof DenseMatrix) {
				double[][] resData = ((DenseMatrix) res).getData();
				// double[] rowA = null;
				int NB = B.getColumnDimension();
				int N = A.getRowDimension();
				int M = A.getColumnDimension();
				if (A instanceof DenseMatrix) {
					double[][] AData = ((DenseMatrix) A).getData();
					if (B instanceof DenseMatrix) {

						double[][] BData = ((DenseMatrix) B).getData();
						// double[] columnB = new double[B.getRowDimension()];
						// double[] columnA = new double[A.getRowDimension()];
						double[] resRow = null;
						double[] BRow = null;
						// double s = 0;
						double A_ki = 0;
						for (int i = 0; i < M; i++) {
							resRow = resData[i];
							clear(resRow);
							for (int k = 0; k < N; k++) {
								BRow = BData[k];
								A_ki = AData[k][i];
								for (int j = 0; j < NB; j++) {
									resRow[j] += A_ki * BRow[j];
								}
							}
						}

						/*for (int j = 0; j < NB; j++) {
							for (int r = 0; r < B.getRowDimension(); r++) {
								columnB[r] = BData[r][j];
							}

							for (int i = 0; i < M; i++) {
								for (int r = 0; r < A.getRowDimension(); r++) {
									columnA[r] = AData[r][i];
								}
								s = 0;
								for (int k = 0; k < N; k++) {
									// Using AData[k][j] costs 16.8 seconds
									// Referring AData[k][j] involves one integer multiplication!
									// s += rowData[k] * AData[k][j];
									// Using columnA[j] costs 3.4 seconds
									s += columnA[k] * columnB[k];
								}
								resData[i][j] = s;
							}
						}*/

					} else if (B instanceof SparseMatrix) {

						int[] ir = null;
						int[] jc = null;
						double[] pr = null;
						ir = ((SparseMatrix) B).getIr();
						jc = ((SparseMatrix) B).getJc();
						pr = ((SparseMatrix) B).getPr();
						int r = -1;
						double s = 0;
						double[] columnA = new double[A.getRowDimension()];
						for (int i = 0; i < M; i++) {
							for (int t = 0; t < N; t++) {
								columnA[t] = AData[t][i];
							}
							for (int j = 0; j < NB; j++) {
								s = 0;
								for (int k = jc[j]; k < jc[j + 1]; k++) {
									r = ir[k];
									// A[r][j] = pr[k]
									s += columnA[r] * pr[k];
								}
								resData[i][j] = s;
							}
						}

					}
				} else if (A instanceof SparseMatrix) {

					if (B instanceof DenseMatrix) {
						int[] ir = ((SparseMatrix) A).getIr();
						int[] jc = ((SparseMatrix) A).getJc();
						double[] pr = ((SparseMatrix) A).getPr();
						// int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
						double[][] BData = ((DenseMatrix) B).getData();
						int c = -1;
						double s = 0;
						for (int i = 0; i < M; i++) {
							for (int j = 0; j < NB; j++) {
								s = 0;
								for (int k = jc[i]; k < jc[i + 1]; k++) {
									c = ir[k];
									s += pr[k] * BData[c][j];
								}
								resData[i][j] = s;
							}
						}
					} else if (B instanceof SparseMatrix) {
						double[] resRow = null;
						int[] ir1 = ((SparseMatrix) A).getIr();
						int[] jc1 = ((SparseMatrix) A).getJc();
						double[] pr1 = ((SparseMatrix) A).getPr();
						int[] ir2 = ((SparseMatrix) B).getIr();
						int[] jc2 = ((SparseMatrix) B).getJc();
						double[] pr2 = ((SparseMatrix) B).getPr();
						// rowIdx of the right sparse matrix
						int rr = -1;
						// colIdx of the left sparse matrix
						int cl = -1;
						double s = 0;
						int kl = 0;
						int kr = 0;
						for (int i = 0; i < M; i++) {
							resRow = resData[i];
							for (int j = 0; j < NB; j++) {
								s = 0;
								kl = jc1[i];
								kr = jc2[j];
								while (true) {
									if (kl >= jc1[i + 1] || kr >= jc2[j + 1]) {
										break;
									}
									cl = ir1[kl];
									rr = ir2[kr];
									if (cl < rr) {
										kl++;
									} else if (cl > rr) {
										kr++;
									} else {
										s += pr1[kl] * pr2[kr];
										kl++;
										kr++;
									}
								}
								resRow[j] = s;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = A * B.
	 * @param res result matrix
	 * @param A	a real matrix
	 * @param B a real matrix
	 */
	public static void mtimes(Matrix res, Matrix A, Matrix B) {
		if (res instanceof SparseMatrix) {
			((SparseMatrix) res).assignSparseMatrix(sparse(A.mtimes(B)));
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] rowA = null;
			int NB = B.getColumnDimension();
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				if (B instanceof DenseMatrix) {

					double[][] BData = ((DenseMatrix) B).getData();
					double[] columnB = new double[B.getRowDimension()];
					double s = 0;
					for (int j = 0; j < NB; j++) {
						for (int r = 0; r < B.getRowDimension(); r++) {
							columnB[r] = BData[r][j];
						}
						for (int i = 0; i < M; i++) {
							rowA = AData[i];
							s = 0;
							for (int k = 0; k < N; k++) {
								// Using AData[k][j] costs 16.8 seconds
								// Referring AData[k][j] involves one integer multiplication!
								// s += rowData[k] * AData[k][j];
								// Using columnA[j] costs 3.4 seconds
								s += rowA[k] * columnB[k];
							}
							resData[i][j] = s;
						}
					}

				} else if (B instanceof SparseMatrix) {

					int[] ir = null;
					int[] jc = null;
					double[] pr = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr = ((SparseMatrix) B).getPr();
					int r = -1;
					double s = 0;

					for (int i = 0; i < M; i++) {
						rowA = AData[i];
						for (int j = 0; j < NB; j++) {
							s = 0;
							for (int k = jc[j]; k < jc[j + 1]; k++) {
								r = ir[k];
								// A[r][j] = pr[k]
								s += rowA[r] * pr[k];
							}
							resData[i][j] = s;
						}
					}

				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				double[] pr = ((SparseMatrix) A).getPr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					int c = -1;
					double s = 0;
					for (int i = 0; i < M; i++) {
						for (int j = 0; j < NB; j++) {
							s = 0;
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								c = ic[k];
								s += pr[valCSRIndices[k]] * BData[c][j];
							}
							resData[i][j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					double[] resRow = null;
					int[] ir = null;
					int[] jc = null;
					double[] pr2 = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();
					// rowIdx of the right sparse matrix
					int rr = -1;
					// colIdx of the left sparse matrix
					int cl = -1;
					double s = 0;
					int kl = 0;
					int kr = 0;
					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = 0;
							kl = jr[i];
							kr = jc[j];
							while (true) {
								if (kl >= jr[i + 1] || kr >= jc[j + 1]) {
									break;
								}
								cl = ic[kl];
								rr = ir[kr];
								if (cl < rr) {
									kl++;
								} else if (cl > rr) {
									kr++;
								} else {
									s += pr[valCSRIndices[kl]] * pr2[kr];
									kl++;
									kr++;
								}
							}
							resRow[j] = s;
						}
					}
				}
			}
		}
	}



	/**
	 * res = A .* B
	 * @param res
	 * @param A
	 * @param B
	 */
	public static void times(Matrix res, Matrix A, Matrix B) {
		if (res instanceof SparseMatrix) {
			((SparseMatrix) res).assignSparseMatrix(sparse(A.times(B)));
		} else if (res instanceof DenseMatrix) {
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			double[][] resData = ((DenseMatrix) res).getData();
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] BRow = null;
					double[] ARow = null;
					double[] resRow = null;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						BRow = BData[i];
						resRow = resData[i];
						for (int j = 0; j < N; j++) {
							resRow[j] = ARow[j] * BRow[j];
						}
					}
				} else if (B instanceof SparseMatrix) {
					ArrayOperator.clearMatrix(resData);
					int[] ir = null;
					int[] jc = null;
					double[] pr = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr = ((SparseMatrix) B).getPr();
					int r = -1;
					for (int j = 0; j < B.getColumnDimension(); j++) {
						for (int k = jc[j]; k < jc[j + 1]; k++) {
							r = ir[k];
							// A[r][j] = pr[k]
							resData[r][j] = AData[r][j] * pr[k];
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				if (B instanceof DenseMatrix) {
					times(res, B, A);
				} else if (B instanceof SparseMatrix) {
					int[] ir1 = null;
					int[] jc1 = null;
					double[] pr1 = null;
					ir1 = ((SparseMatrix) A).getIr();
					jc1 = ((SparseMatrix) A).getJc();
					pr1 = ((SparseMatrix) A).getPr();
					int[] ir2 = null;
					int[] jc2 = null;
					double[] pr2 = null;
					ir2 = ((SparseMatrix) B).getIr();
					jc2 = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();

					ArrayOperator.clearMatrix(resData);

					int k1 = 0;
					int k2 = 0;
					int r1 = -1;
					int r2 = -1;
					int i = -1;
					double v = 0;

					for (int j = 0; j < N; j++) {
						k1 = jc1[j];
						k2 = jc2[j];

						// If the j-th column of A or this is empty, we don't need to compute.
						if (k1 == jc1[j + 1] || k2 == jc2[j + 1])
							continue;

						while (k1 < jc1[j + 1] && k2 < jc2[j + 1]) {

							r1 = ir1[k1];
							r2 = ir2[k2];
							if (r1 < r2) {
								k1++;
							} else if (r1 == r2) {
								i = r1;
								v = pr1[k1] * pr2[k2];
								k1++;
								k2++;
								if (v != 0) {
									resData[i][j] = v;
								}
							} else {
								k2++;
							}

						}

					}
				}
			}
		}
	}

	/**
	 * res = a * V
	 * @param res
	 * @param a
	 * @param V
	 */
	public static void times(double[] res, double a, double[] V) {
		for (int i = 0; i < res.length; i++) {
			res[i] = a * V[i];
		}
	}

	/**
	 * res = res .* A
	 * @param res
	 * @param A
	 */
	public static void timesAssign(Matrix res, Matrix A) {
		if (res instanceof SparseMatrix) {
			((SparseMatrix) res).assignSparseMatrix(sparse(res.times(A)));
		} else if (res instanceof DenseMatrix) {
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					ARow = AData[i];
					resRow = resData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] *= ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.clearVector(resRow);
						continue;
					}

					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = 0;
						}
						resRow[currentColumnIdx] *= pr[valCSRIndices[k]];
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = 0;
					}
				}
			}
		}
	}

	/**
	 * res = v * res
	 * @param res
	 * @param v
	 */
	public static void timesAssign(Matrix res, double v) {
		if (v == 0) {
			res.clear();
		}
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] *= v;
			}
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] *= v;
				}
			}
		}
	}

	/**
	 * res = v * A
	 * @param res
	 * @param v
	 * @param A
	 */
	public static void times(Matrix res, double v, Matrix A) {
		if (v == 1) {
			assign(res, A);
			return;
		}
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = ARow[j] * v;
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, 0);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = 0;
						}
						resRow[currentColumnIdx] = pr[valCSRIndices[k]] * v;
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = 0;
					}
				}
			}
		}
	}

	/**
	 * res = v.
	 * 
	 * @param res
	 * @param v
	 *//*
	public static void assign(Matrix res, double v) {
		if (v == 0) {
			clear(res);
			return;
		}

	}*/

	/**
	 * Clear the input matrix.
	 * 
	 * @param res a real matrix
	 */
	public static void clear(Matrix res) {
		res.clear();
	}

	/**
	 * Clear the input 2D {@code double} array.
	 * 
	 * @param res a 2D {@code double} array
	 */
	public static void clear(double[][] res) {
		ArrayOperator.clearMatrix(res);
	}

	/**
	 * Clear the input vector.
	 * 
	 * @param res a real vector
	 */
	public static void clear(Vector res) {
		res.clear();
	}

	/**
	 * Clear the input 1D {@code double} array.
	 * 
	 * @param res a 1D {@code double} array
	 */
	public static void clear(double[] res) {
		ArrayOperator.clearVector(res);
	}

	/**
	 * res = A
	 * @param res
	 * @param A
	 */
	public static void assign(Matrix res, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {
			((SparseMatrix) res).assignSparseMatrix(sparse(A));
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, 0);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = 0;
						}
						resRow[currentColumnIdx] = pr[valCSRIndices[k]];
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = 0;
					}
				}
			}
		}
	}

	/**
	 * res = V
	 * @param res
	 * @param V
	 */
	public static void assign(double[] res, double[] V) {
		System.arraycopy(V, 0, res, 0, res.length);
	}

	/**
	 * Assign a 1D {@code double} array by a real scalar.
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 */
	public static void assign(double[] res, double v) {
		for (int i = 0; i < res.length; i++)
			res[i] = v;
	}

	/**
	 * res = -res
	 * @param res
	 */
	public static void uminusAssign(Matrix res) {
		if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] = -pr[k];
			}
		} else if (res instanceof DenseMatrix) {
			int M = res.getRowDimension();
			int N = res.getColumnDimension();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] = -resRow[j];
				}
			}
		}
	}

	/**
	 * res = -A
	 * @param res
	 * @param A
	 */
	public static void uminus(Matrix res, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = -ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, 0);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = 0;
						}
						resRow[currentColumnIdx] = -pr[valCSRIndices[k]];
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = 0;
					}
				}
			}
		}
	}

	/**
	 * res = -V
	 * @param res
	 * @param V
	 */
	public static void uminus(double[] res, double[] V) {
		for (int i = 0; i < res.length; i++) {
			res[i] = -V[i];
		}
	}

	/**
	 * res = v \ res
	 * @param res
	 * @param v
	 */
	public static void divide(Matrix res, double v) {

	}

	/**
	 * res = res / v
	 * @param res
	 * @param v
	 */
	public static void rdivideAssign(Matrix res, double v) {
		int nRow = res.getRowDimension();
		int nCol = res.getColumnDimension();

		if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < nRow; i++) {
				resRow = resData[i];
				for (int j = 0; j < nCol; j++) {
					resRow[j] /= v;
				}
			}
		} else if (res instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) res).getPr();
			for (int k = 0; k < pr.length; k++) {
				pr[k] /= v;
			}
		}
	}

	/**
	 * res = A * B + v * C
	 * @param res
	 * @param A
	 * @param B
	 * @param v
	 * @param C
	 */
	public static void affine(Matrix res, Matrix A, Matrix B, double v, Matrix C) {
		if (res instanceof SparseMatrix) {
			err("Sparse matrix for res is not supported.");
			exit(1);
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			int NB = B.getColumnDimension();
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] columnB = new double[B.getRowDimension()];

					double s = 0;
					for (int j = 0; j < NB; j++) {
						for (int r = 0; r < A.getRowDimension(); r++) {
							columnB[r] = BData[r][j];
						}
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							s = v * C.getEntry(i, j);
							for (int k = 0; k < N; k++) {
								// Using AData[k][j] costs 16.8 seconds
								// Referring AData[k][j] involves one integer multiplication!
								// s += rowData[k] * AData[k][j];
								// Using columnA[j] costs 3.4 seconds
								s += ARow[k] * columnB[k];
							}
							resData[i][j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					int[] ir = null;
					int[] jc = null;
					double[] pr = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr = ((SparseMatrix) B).getPr();
					int r = -1;
					double s = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = v * C.getEntry(i, j);
							for (int k = jc[j]; k < jc[j + 1]; k++) {
								r = ir[k];
								// A[r][j] = pr[k]
								s += ARow[r] * pr[k];
							}
							resRow[j] = s;
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					int c = -1;
					double s = 0;
					// double v = 0;
					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = v * C.getEntry(i, j);
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								c = ic[k];
								s += pr[valCSRIndices[k]] * BData[c][j];
							}
							resRow[j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					/*
					 * When this and A are all sparse matrices,
					 * the result is also a sparse matrix.
					 */
					int[] ir = null;
					int[] jc = null;
					double[] pr2 = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();
					// rowIdx of the right sparse matrix
					int rr = -1;
					// colIdx of the left sparse matrix
					int cl = -1;
					double s = 0;
					int kl = 0;
					int kr = 0;

					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = v * C.getEntry(i, j);
							kl = jr[i];
							kr = jc[j];
							while (true) {
								if (kl >= jr[i + 1] || kr >= jc[j + 1]) {
									break;
								}
								cl = ic[kl];
								rr = ir[kr];
								if (cl < rr) {
									kl++;
								} else if (cl > rr) {
									kr++;
								} else {
									s += pr[valCSRIndices[kl]] * pr2[kr];
									kl++;
									kr++;
								}
							}
							resRow[j] = s;	
						}
					}
				}
			}
		}
	}

	/**
	 * res = A * B + C if operator is '+',</br>
	 * res = A * B - C if operator is '-'.
	 * 
	 * @param res
	 * @param A
	 * @param B
	 * @param operator a {@code char} variable: '+' or '-'
	 * @param C
	 */
	public static void affine(Matrix res, Matrix A, Matrix B, char operator, Matrix C) {
		if (res instanceof SparseMatrix) {
			err("Sparse matrix for res is not supported.");
			exit(1);
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			int NB = B.getColumnDimension();
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] columnB = new double[B.getRowDimension()];

					double s = 0;
					for (int j = 0; j < NB; j++) {
						for (int r = 0; r < A.getRowDimension(); r++) {
							columnB[r] = BData[r][j];
						}
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							if (operator == '+')
								s = C.getEntry(i, j);
							else if (operator == '-')
								s = -C.getEntry(i, j);
							// s = v * C.getEntry(i, j);
							for (int k = 0; k < N; k++) {
								// Using AData[k][j] costs 16.8 seconds
								// Referring AData[k][j] involves one integer multiplication!
								// s += rowData[k] * AData[k][j];
								// Using columnA[j] costs 3.4 seconds
								s += ARow[k] * columnB[k];
							}
							resData[i][j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					int[] ir = null;
					int[] jc = null;
					double[] pr = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr = ((SparseMatrix) B).getPr();
					int r = -1;
					double s = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							if (operator == '+')
								s = C.getEntry(i, j);
							else if (operator == '-')
								s = -C.getEntry(i, j);
							for (int k = jc[j]; k < jc[j + 1]; k++) {
								r = ir[k];
								// A[r][j] = pr[k]
								s += ARow[r] * pr[k];
							}
							resRow[j] = s;
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					int c = -1;
					double s = 0;
					// double v = 0;
					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							if (operator == '+')
								s = C.getEntry(i, j);
							else if (operator == '-')
								s = -C.getEntry(i, j);
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								c = ic[k];
								s += pr[valCSRIndices[k]] * BData[c][j];
							}
							resRow[j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					/*
					 * When this and A are all sparse matrices,
					 * the result is also a sparse matrix.
					 */
					int[] ir = null;
					int[] jc = null;
					double[] pr2 = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();
					// rowIdx of the right sparse matrix
					int rr = -1;
					// colIdx of the left sparse matrix
					int cl = -1;
					double s = 0;
					int kl = 0;
					int kr = 0;

					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							if (operator == '+')
								s = C.getEntry(i, j);
							else if (operator == '-')
								s = -C.getEntry(i, j);
							kl = jr[i];
							kr = jc[j];
							while (true) {
								if (kl >= jr[i + 1] || kr >= jc[j + 1]) {
									break;
								}
								cl = ic[kl];
								rr = ir[kr];
								if (cl < rr) {
									kl++;
								} else if (cl > rr) {
									kr++;
								} else {
									s += pr[valCSRIndices[kl]] * pr2[kr];
									kl++;
									kr++;
								}
							}
							resRow[j] = s;
						}
					}
				}
			}
		}
	}

	/**
	 * res = A * B + v
	 * @param res
	 * @param A
	 * @param B
	 * @param v
	 */
	public static void affine(Matrix res, Matrix A, Matrix B, double v) {
		if (res instanceof SparseMatrix) {
			err("Sparse matrix for res is not supported.");
			exit(1);
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			int NB = B.getColumnDimension();
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] columnB = new double[B.getRowDimension()];

					double s = 0;
					for (int j = 0; j < NB; j++) {
						for (int r = 0; r < A.getRowDimension(); r++) {
							columnB[r] = BData[r][j];
						}
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							s = v;
							for (int k = 0; k < N; k++) {
								// Using AData[k][j] costs 16.8 seconds
								// Referring AData[k][j] involves one integer multiplication!
								// s += rowData[k] * AData[k][j];
								// Using columnA[j] costs 3.4 seconds
								s += ARow[k] * columnB[k];
							}
							resData[i][j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					int[] ir = null;
					int[] jc = null;
					double[] pr = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr = ((SparseMatrix) B).getPr();
					int r = -1;
					double s = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = v;
							for (int k = jc[j]; k < jc[j + 1]; k++) {
								r = ir[k];
								// A[r][j] = pr[k]
								s += ARow[r] * pr[k];
							}
							resRow[j] = s;
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					int c = -1;
					double s = 0;
					// double v = 0;
					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = v;
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								c = ic[k];
								s += pr[valCSRIndices[k]] * BData[c][j];
							}
							resRow[j] = s;
						}
					}
				} else if (B instanceof SparseMatrix) {
					/*
					 * When this and A are all sparse matrices,
					 * the result is also a sparse matrix.
					 */
					int[] ir = null;
					int[] jc = null;
					double[] pr2 = null;
					ir = ((SparseMatrix) B).getIr();
					jc = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();
					// rowIdx of the right sparse matrix
					int rr = -1;
					// colIdx of the left sparse matrix
					int cl = -1;
					double s = 0;
					int kl = 0;
					int kr = 0;

					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = v;
							kl = jr[i];
							kr = jc[j];
							while (true) {
								if (kl >= jr[i + 1] || kr >= jc[j + 1]) {
									break;
								}
								cl = ic[kl];
								rr = ir[kr];
								if (cl < rr) {
									kl++;
								} else if (cl > rr) {
									kr++;
								} else {
									s += pr[valCSRIndices[kl]] * pr2[kr];
									kl++;
									kr++;
								}
							}
							resRow[j] = s;	
						}
					}
				}
			}
		}
	}

	/**
	 * res = a * A + b * B
	 * @param res
	 * @param a
	 * @param A
	 * @param b
	 * @param B
	 */
	public static void affine(Matrix res, double a, Matrix A, double b, Matrix B) {
		if (b == 0) {
			times(res, a, A);
			return;
		} else if (b == 1) {
			affine(res, a, A, '+', B);
			return;
		} else if (b == -1) {
			affine(res, a, A, '-', B);
			return;
		}
		if (a == 0) {
			times(res, b, B);
			return;
		} else if (a == 1) {
			affine(res, b, B, '+', A);
			return;
		} else if (a == -1) {
			affine(res, b, B, '-', A);
			return;
		}
		if (res instanceof DenseMatrix) { // res = a * A + b * B
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] BRow = null;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						BRow = BData[i];
						resRow = resData[i];
						for (int j = 0; j < N; j++) {
							resRow[j] = a * ARow[j] + b * BRow[j];
						}
					}
				} else if (B instanceof SparseMatrix) {
					int[] ic = ((SparseMatrix) B).getIc();
					int[] jr = ((SparseMatrix) B).getJr();
					int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
					double[] pr = ((SparseMatrix) B).getPr();
					int j = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						resRow = resData[i];
						times(resRow, a, ARow);
						for (int k = jr[i]; k < jr[i + 1]; k++) {
							j = ic[k];
							resRow[j] += b * pr[valCSRIndices[k]];
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) A).getData();
					double[] BRow = null;
					int[] ic = ((SparseMatrix) A).getIc();
					int[] jr = ((SparseMatrix) A).getJr();
					int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
					double[] pr = ((SparseMatrix) A).getPr();
					int j = 0;
					for (int i = 0; i < M; i++) {
						BRow = BData[i];
						resRow = resData[i];
						times(resRow, b, BRow);
						for (int k = jr[i]; k < jr[i + 1]; k++) {
							j = ic[k];
							resRow[j] += a * pr[valCSRIndices[k]];
						}
					}
				} else if (B instanceof SparseMatrix) {
					res.clear();
					// res = a * A + b * B where both A and B are sparse matrices
					int[] ir1 = null;
					int[] jc1 = null;
					double[] pr1 = null;
					ir1 = ((SparseMatrix) A).getIr();
					jc1 = ((SparseMatrix) A).getJc();
					pr1 = ((SparseMatrix) A).getPr();
					int[] ir2 = null;
					int[] jc2 = null;
					double[] pr2 = null;
					ir2 = ((SparseMatrix) B).getIr();
					jc2 = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();

					int k1 = 0;
					int k2 = 0;
					int r1 = -1;
					int r2 = -1;
					int i = -1;
					double v = 0;

					for (int j = 0; j < N; j++) {
						k1 = jc1[j];
						k2 = jc2[j];

						// Both A and B's j-th columns are empty.
						if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
							continue;

						while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {

							if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
								i = ir1[k1];
								v = a * pr1[k1];
								k1++;
							} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
								i = ir2[k2];
								v = b * pr2[k2];
								k2++;
							} else { // Both A and B's j-th columns have not been fully processed.
								r1 = ir1[k1];
								r2 = ir2[k2];				
								if (r1 < r2) {
									i = r1;
									v = a * pr1[k1];
									k1++;
								} else if (r1 == r2) {
									i = r1;
									v = a * pr1[k1] + b * pr2[k2];
									k1++;
									k2++;
								} else {
									i = r2;
									v = b * pr2[k2];
									k2++;
								}
							}
							if (v != 0)
								resData[i][j] = v;
						}
					}
				}
			}
		}
	}

	/**
	 * res = a * A + B if operator is '+',</br>
	 * res = a * A - B if operator is '-'.
	 * 
	 * @param res
	 * @param a
	 * @param A
	 * @param operator a {@code char} variable: '+' or '-'
	 * @param B
	 */
	public static void affine(Matrix res, double a, Matrix A, char operator, Matrix B) {
		if (operator == '+') {
			if (a == 0) {
				assign(res, B);
				return;
			} else if (a == 1) {
				plus(res, A, B);
				return;
			} else if (a == -1) {
				minus(res, B, A);
				return;
			}
			if (res instanceof DenseMatrix) { // res = a * A + B
				int M = A.getRowDimension();
				int N = A.getColumnDimension();
				double[][] resData = ((DenseMatrix) res).getData();
				double[] resRow = null;
				if (A instanceof DenseMatrix) {
					double[][] AData = ((DenseMatrix) A).getData();
					double[] ARow = null;
					if (B instanceof DenseMatrix) {
						double[][] BData = ((DenseMatrix) B).getData();
						double[] BRow = null;
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							BRow = BData[i];
							resRow = resData[i];
							for (int j = 0; j < N; j++) {
								resRow[j] = a * ARow[j] + BRow[j];
							}
						}
					} else if (B instanceof SparseMatrix) {
						int[] ic = ((SparseMatrix) B).getIc();
						int[] jr = ((SparseMatrix) B).getJr();
						int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
						double[] pr = ((SparseMatrix) B).getPr();
						int j = 0;
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							resRow = resData[i];
							times(resRow, a, ARow);
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								j = ic[k];
								resRow[j] += pr[valCSRIndices[k]];
							}
						}
					}
				} else if (A instanceof SparseMatrix) {
					if (B instanceof DenseMatrix) {
						double[][] BData = ((DenseMatrix) A).getData();
						double[] BRow = null;
						int[] ic = ((SparseMatrix) A).getIc();
						int[] jr = ((SparseMatrix) A).getJr();
						int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
						double[] pr = ((SparseMatrix) A).getPr();
						int j = 0;
						for (int i = 0; i < M; i++) {
							BRow = BData[i];
							resRow = resData[i];
							assign(resRow, BRow);
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								j = ic[k];
								resRow[j] += a * pr[valCSRIndices[k]];
							}
						}
					} else if (B instanceof SparseMatrix) {
						res.clear();
						// res = a * A + B where both A and B are sparse matrices
						int[] ir1 = null;
						int[] jc1 = null;
						double[] pr1 = null;
						ir1 = ((SparseMatrix) A).getIr();
						jc1 = ((SparseMatrix) A).getJc();
						pr1 = ((SparseMatrix) A).getPr();
						int[] ir2 = null;
						int[] jc2 = null;
						double[] pr2 = null;
						ir2 = ((SparseMatrix) B).getIr();
						jc2 = ((SparseMatrix) B).getJc();
						pr2 = ((SparseMatrix) B).getPr();

						int k1 = 0;
						int k2 = 0;
						int r1 = -1;
						int r2 = -1;
						int i = -1;
						double v = 0;

						for (int j = 0; j < N; j++) {
							k1 = jc1[j];
							k2 = jc2[j];

							// Both A and B's j-th columns are empty.
							if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
								continue;

							while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {

								if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
									i = ir1[k1];
									v = a * pr1[k1];
									k1++;
								} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
									i = ir2[k2];
									v = pr2[k2];
									k2++;
								} else { // Both A and B's j-th columns have not been fully processed.
									r1 = ir1[k1];
									r2 = ir2[k2];				
									if (r1 < r2) {
										i = r1;
										v = a * pr1[k1];
										k1++;
									} else if (r1 == r2) {
										i = r1;
										v = a * pr1[k1] + pr2[k2];
										k1++;
										k2++;
									} else {
										i = r2;
										v = pr2[k2];
										k2++;
									}
								}
								if (v != 0)
									resData[i][j] = v;
							}
						}
					}
				}
			}
		} else if (operator == '-') { // res = a * A - B
			if (a == 0) {
				uminus(res, B);
				return;
			} else if (a == 1) {
				minus(res, A, B);
				return;
			}
			if (res instanceof DenseMatrix) {
				int M = A.getRowDimension();
				int N = A.getColumnDimension();
				double[][] resData = ((DenseMatrix) res).getData();
				double[] resRow = null;
				if (A instanceof DenseMatrix) {
					double[][] AData = ((DenseMatrix) A).getData();
					double[] ARow = null;
					if (B instanceof DenseMatrix) {
						double[][] BData = ((DenseMatrix) B).getData();
						double[] BRow = null;
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							BRow = BData[i];
							resRow = resData[i];
							for (int j = 0; j < N; j++) {
								resRow[j] = a * ARow[j] - BRow[j];
							}
						}
					} else if (B instanceof SparseMatrix) {
						int[] ic = ((SparseMatrix) B).getIc();
						int[] jr = ((SparseMatrix) B).getJr();
						int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
						double[] pr = ((SparseMatrix) B).getPr();
						int j = 0;
						for (int i = 0; i < M; i++) {
							ARow = AData[i];
							resRow = resData[i];
							times(resRow, a, ARow);
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								j = ic[k];
								resRow[j] -= pr[valCSRIndices[k]];
							}
						}
					}
				} else if (A instanceof SparseMatrix) {
					if (B instanceof DenseMatrix) {
						double[][] BData = ((DenseMatrix) A).getData();
						double[] BRow = null;
						int[] ic = ((SparseMatrix) A).getIc();
						int[] jr = ((SparseMatrix) A).getJr();
						int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
						double[] pr = ((SparseMatrix) A).getPr();
						int j = 0;
						for (int i = 0; i < M; i++) {
							BRow = BData[i];
							resRow = resData[i];
							uminus(resRow, BRow);
							for (int k = jr[i]; k < jr[i + 1]; k++) {
								j = ic[k];
								resRow[j] += a * pr[valCSRIndices[k]];
							}
						}
					} else if (B instanceof SparseMatrix) {
						res.clear();
						// res = a * A - B where both A and B are sparse matrices
						int[] ir1 = null;
						int[] jc1 = null;
						double[] pr1 = null;
						ir1 = ((SparseMatrix) A).getIr();
						jc1 = ((SparseMatrix) A).getJc();
						pr1 = ((SparseMatrix) A).getPr();
						int[] ir2 = null;
						int[] jc2 = null;
						double[] pr2 = null;
						ir2 = ((SparseMatrix) B).getIr();
						jc2 = ((SparseMatrix) B).getJc();
						pr2 = ((SparseMatrix) B).getPr();

						int k1 = 0;
						int k2 = 0;
						int r1 = -1;
						int r2 = -1;
						int i = -1;
						double v = 0;

						for (int j = 0; j < N; j++) {
							k1 = jc1[j];
							k2 = jc2[j];

							// Both A and B's j-th columns are empty.
							if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
								continue;

							while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {

								if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
									i = ir1[k1];
									v = a * pr1[k1];
									k1++;
								} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
									i = ir2[k2];
									v = -pr2[k2];
									k2++;
								} else { // Both A and B's j-th columns have not been fully processed.
									r1 = ir1[k1];
									r2 = ir2[k2];				
									if (r1 < r2) {
										i = r1;
										v = a * pr1[k1];
										k1++;
									} else if (r2 < r1) {
										i = r2;
										v = -pr2[k2];
										k2++;
									} else { // if (r1 == r2)
										i = r1;
										v = a * pr1[k1] - pr2[k2];
										k1++;
										k2++;
									}
								}
								if (v != 0)
									resData[i][j] = v;
							}
						}
					}
				}
			}
		}
	}

	/**
	 * res = A + b * B
	 * @param res
	 * @param A
	 * @param b
	 * @param B
	 */
	public static void affine(Matrix res, Matrix A, double b, Matrix B) {
		affine(res, b, B, '+', A);
	}

	/**
	 * res = a * A + b
	 * @param res
	 * @param a
	 * @param A
	 * @param b
	 */
	public static void affine(Matrix res, double a, Matrix A, double b) {
		if (a == 1) {
			plus(res, A, b);
			return;
		}
		if (b == 0) {
			times(res, a, A);
			return;
		}
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = a * ARow[j] + b;
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, b);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = b;
						}
						resRow[currentColumnIdx] = a * pr[valCSRIndices[k]] + b;
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = b;
					}
				}
			}
		}
	}

	/**
	 * res = A + B
	 * @param res
	 * @param A
	 * @param B
	 */
	public static void plus(Matrix res, Matrix A, Matrix B) {
		if (res instanceof SparseMatrix) {
			((SparseMatrix) res).assignSparseMatrix(sparse(A.plus(B)));
		} else if (res instanceof DenseMatrix) {
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] BRow = null;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						BRow = BData[i];
						resRow = resData[i];
						for (int j = 0; j < N; j++) {
							resRow[j] = ARow[j] + BRow[j];
						}
					}
				} else if (B instanceof SparseMatrix) {
					int[] ic = ((SparseMatrix) B).getIc();
					int[] jr = ((SparseMatrix) B).getJr();
					int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
					double[] pr = ((SparseMatrix) B).getPr();
					int j = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						resRow = resData[i];
						assign(resRow, ARow);
						for (int k = jr[i]; k < jr[i + 1]; k++) {
							j = ic[k];
							resRow[j] += pr[valCSRIndices[k]];
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				if (B instanceof DenseMatrix) {
					plus(res, B, A);
				} else if (B instanceof SparseMatrix) {
					res.clear();
					// res = A + B where both A and B are sparse matrices
					int[] ir1 = null;
					int[] jc1 = null;
					double[] pr1 = null;
					ir1 = ((SparseMatrix) A).getIr();
					jc1 = ((SparseMatrix) A).getJc();
					pr1 = ((SparseMatrix) A).getPr();
					int[] ir2 = null;
					int[] jc2 = null;
					double[] pr2 = null;
					ir2 = ((SparseMatrix) B).getIr();
					jc2 = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();

					int k1 = 0;
					int k2 = 0;
					int r1 = -1;
					int r2 = -1;
					int i = -1;
					double v = 0;

					for (int j = 0; j < N; j++) {
						k1 = jc1[j];
						k2 = jc2[j];

						// Both A and B's j-th columns are empty.
						if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
							continue;

						while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {

							if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
								i = ir1[k1];
								v = pr1[k1];
								k1++;
							} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
								i = ir2[k2];
								v = pr2[k2];
								k2++;
							} else { // Both A and B's j-th columns have not been fully processed.
								r1 = ir1[k1];
								r2 = ir2[k2];				
								if (r1 < r2) {
									i = r1;
									v = pr1[k1];
									k1++;
								} else if (r1 == r2) {
									i = r1;
									v = pr1[k1] + pr2[k2];
									k1++;
									k2++;
								} else {
									i = r2;
									v = pr2[k2];
									k2++;
								}
							}
							if (v != 0)
								resData[i][j] = v;
						}
					}
				}
			}
		}
	}

	/**
	 * res = A + v;
	 * @param res
	 * @param A
	 * @param v
	 */
	public static void plus(Matrix res, Matrix A, double v) {
		if (v == 0) {
			assign(res, A);
			return;
		}
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = ARow[j] + v;
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, v);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = v;
						}
						resRow[currentColumnIdx] = pr[valCSRIndices[k]] + v;
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = v;
					}
				}
			}
		}
	}

	/**
	 * res = res + A
	 * @param res
	 * @param A
	 */
	public static void plusAssign(Matrix res, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] += ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				int j = 0;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						j = ic[k];
						resRow[j] += pr[valCSRIndices[k]];
					}
				}
			}
		}
	}

	/**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument by the second argument
	 * and assign the result to the first argument, i.e., res += v.
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param v a real scalar
	 * 
	 *//*
	public static void plusAssign(double[] res, double v) {
		for (int i = 0; i < res.length; i++)
			res[i] += v;
	}*/

	/**
	 * Element-wise addition and assignment operation.
	 * It adds the first argument by the second argument
	 * and assign the result to the first argument, i.e., res += V2.
	 * 
	 * @param res a 1D {@code double} array
	 * 
	 * @param V a 1D {@code double} array
	 * 
	 *//*
	public static void plusAssign(double[] res, double[] V) {
		for (int i = 0; i < res.length; i++)
			res[i] += V[i];
	}*/

	/**
	 * res = res + v
	 * @param res
	 * @param v
	 */
	public static void plusAssign(Matrix res, double v) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof SparseMatrix) {
			/*double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] += v;
			}*/
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] += v;
				}
			}
		}
	}

	/**
	 * res += a * A.
	 * 
	 * @param res
	 * @param a
	 * @param A
	 */
	public static void plusAssign(Matrix res, double a, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (a == 0) {
			return;
		} else if (a == 1) {
			plusAssign(res, A);
			return;
		} else if (a == -1) {
			minusAssign(res, A);
			return;
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] += a * ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				int j = 0;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						j = ic[k];
						resRow[j] += a * pr[valCSRIndices[k]];
					}
				}
			}
		}
	}

	/**
	 * res = A - B
	 * @param res
	 * @param A
	 * @param B
	 */
	public static void minus(Matrix res, Matrix A, Matrix B) {
		if (res instanceof SparseMatrix) {
			((SparseMatrix) res).assignSparseMatrix(sparse(A.minus(B)));
		} else if (res instanceof DenseMatrix) {
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) B).getData();
					double[] BRow = null;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						BRow = BData[i];
						resRow = resData[i];
						for (int j = 0; j < N; j++) {
							resRow[j] = ARow[j] - BRow[j];
						}
					}
				} else if (B instanceof SparseMatrix) {
					int[] ic = ((SparseMatrix) B).getIc();
					int[] jr = ((SparseMatrix) B).getJr();
					int[] valCSRIndices = ((SparseMatrix) B).getValCSRIndices();
					double[] pr = ((SparseMatrix) B).getPr();
					int j = 0;
					for (int i = 0; i < M; i++) {
						ARow = AData[i];
						resRow = resData[i];
						assign(resRow, ARow);
						for (int k = jr[i]; k < jr[i + 1]; k++) {
							j = ic[k];
							resRow[j] -= pr[valCSRIndices[k]];
						}
					}
				}
			} else if (A instanceof SparseMatrix) {
				if (B instanceof DenseMatrix) {
					double[][] BData = ((DenseMatrix) A).getData();
					double[] BRow = null;
					int[] ic = ((SparseMatrix) A).getIc();
					int[] jr = ((SparseMatrix) A).getJr();
					int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
					double[] pr = ((SparseMatrix) A).getPr();
					int j = 0;
					for (int i = 0; i < M; i++) {
						BRow = BData[i];
						resRow = resData[i];
						uminus(resRow, BRow);
						for (int k = jr[i]; k < jr[i + 1]; k++) {
							j = ic[k];
							resRow[j] += pr[valCSRIndices[k]];
						}
					}
				} else if (B instanceof SparseMatrix) {
					res.clear();
					// res = A - B where both A and B are sparse matrices
					int[] ir1 = null;
					int[] jc1 = null;
					double[] pr1 = null;
					ir1 = ((SparseMatrix) A).getIr();
					jc1 = ((SparseMatrix) A).getJc();
					pr1 = ((SparseMatrix) A).getPr();
					int[] ir2 = null;
					int[] jc2 = null;
					double[] pr2 = null;
					ir2 = ((SparseMatrix) B).getIr();
					jc2 = ((SparseMatrix) B).getJc();
					pr2 = ((SparseMatrix) B).getPr();

					int k1 = 0;
					int k2 = 0;
					int r1 = -1;
					int r2 = -1;
					int i = -1;
					double v = 0;

					for (int j = 0; j < N; j++) {
						k1 = jc1[j];
						k2 = jc2[j];

						// Both A and B's j-th columns are empty.
						if (k1 == jc1[j + 1] && k2 == jc2[j + 1])
							continue;

						while (k1 < jc1[j + 1] || k2 < jc2[j + 1]) {

							if (k2 == jc2[j + 1]) { // B's j-th column has been processed.
								i = ir1[k1];
								v = pr1[k1];
								k1++;
							} else if (k1 == jc1[j + 1]) { // A's j-th column has been processed.
								i = ir2[k2];
								v = -pr2[k2];
								k2++;
							} else { // Both A and B's j-th columns have not been fully processed.
								r1 = ir1[k1];
								r2 = ir2[k2];				
								if (r1 < r2) {
									i = r1;
									v = pr1[k1];
									k1++;
								} else if (r2 < r1) {
									i = r2;
									v = -pr2[k2];
									k2++;
								} else { // if (r1 == r2)
									i = r1;
									v = pr1[k1] - pr2[k2];
									k1++;
									k2++;
								}
							}
							if (v != 0)
								resData[i][j] = v;
						}
					}
				}
			}
		}
	}

	/**
	 * res = A - v
	 * @param res
	 * @param A
	 * @param v
	 */
	public static void minus(Matrix res, Matrix A, double v) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = ARow[j] - v;
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, -v);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = -v;
						}
						resRow[currentColumnIdx] = pr[valCSRIndices[k]] - v;
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = -v;
					}
				}
			}
		}
	}

	/**
	 * res = v - A
	 * @param res
	 * @param v
	 * @param A
	 */
	public static void minus(Matrix res, double v, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = v - ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					if (jr[i] ==  jr[i + 1]) {
						ArrayOperator.assignVector(resRow, v);
						continue;
					}
					int lastColumnIdx = -1;
					int currentColumnIdx = 0;
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						currentColumnIdx = ic[k];
						for (int c = lastColumnIdx + 1; c < currentColumnIdx; c++) {
							resRow[c] = v;
						}
						resRow[currentColumnIdx] = v - pr[valCSRIndices[k]];
						lastColumnIdx = currentColumnIdx;
					}
					for (int c = lastColumnIdx + 1; c < N; c++) {
						resRow[c] = v;
					}
				}
			}
		}
	}

	/**
	 * res = res - A
	 * @param res
	 * @param A
	 */
	public static void minusAssign(Matrix res, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] -= ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				int j = 0;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						j = ic[k];
						resRow[j] -= pr[valCSRIndices[k]];
					}
				}
			}
		}
	}

	/**
	 * res = res - v
	 * @param res
	 * @param v
	 */
	public static void minusAssign(Matrix res, double v) {
		if (v == 0) {
			return;
		}
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (res instanceof SparseMatrix) {
			/*double[] pr = ((SparseMatrix) res).getPr();
			int nnz = ((SparseMatrix) res).getNNZ();
			for (int k = 0; k < nnz; k++) {
				pr[k] -= v;
			}*/
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] -= v;
				}
			}
		}
	}

	/**
	 * res -= a * A.
	 * 
	 * @param res
	 * @param a
	 * @param A
	 */
	public static void minusAssign(Matrix res, double a, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (a == 0) {
			return;
		} else if (a == 1) {
			minusAssign(res, A);
			return;
		} else if (a == -1) {
			plusAssign(res, A);
			return;
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] -= a * ARow[j];
					}
				}
			} else if (A instanceof SparseMatrix) {
				int[] ic = ((SparseMatrix) A).getIc();
				int[] jr = ((SparseMatrix) A).getJr();
				int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
				double[] pr = ((SparseMatrix) A).getPr();
				int j = 0;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					for (int k = jr[i]; k < jr[i + 1]; k++) {
						j = ic[k];
						resRow[j] -= a * pr[valCSRIndices[k]];
					}
				}
			}
		}
	}

	/**
	 * res = log(A).
	 * @param res
	 * @param A
	 */
	public static void log(Matrix res, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = Math.log(ARow[j]);
					}
				}
			}
		}
	}

	/**
	 * res = log(V).
	 * 
	 * @param res
	 * @param V
	 */
	public static void log(double[] res, double[] V) {
		for (int i = 0; i < res.length; i++) {
			res[i] = Math.log(V[i]);
		}
	}

	/**
	 * res = log(res).
	 * 
	 * @param res
	 */
	public static void logAssign(double[] res) {
		for (int i = 0; i < res.length; i++) {
			res[i] = Math.log(res[i]);
		}
	}

	/**
	 * res = log(res).
	 * @param res
	 */
	public static void logAssign(Matrix res) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();

		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] = Math.log(resRow[j]);
				}
			}
		}
	}

	/**
	 * res = exp(A).
	 * @param res
	 * @param A
	 */
	public static void exp(Matrix res, Matrix A) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();
		if (M != A.getRowDimension() || N != A.getColumnDimension()) {
			System.err.println("Dimension doesn't match.");
			System.exit(1);
		}
		if (res instanceof SparseMatrix) {

		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			if (A instanceof DenseMatrix) {
				double[][] AData = ((DenseMatrix) A).getData();
				double[] ARow = null;
				for (int i = 0; i < M; i++) {
					resRow = resData[i];
					ARow = AData[i];
					for (int j = 0; j < N; j++) {
						resRow[j] = Math.exp(ARow[j]);
					}
				}
			}
		}
	}

	/**
	 * res = exp(res).
	 * 
	 * @param res
	 */
	public static void expAssign(Matrix res) {
		int M = res.getRowDimension();
		int N = res.getColumnDimension();

		if (res instanceof SparseMatrix) {
			err("The expAssign routine doesn't support sparse matrix.");
			exit(1);
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
			for (int i = 0; i < M; i++) {
				resRow = resData[i];
				for (int j = 0; j < N; j++) {
					resRow[j] = Math.exp(resRow[j]);
				}
			}
		}
	}

	/**
	 * Calculate the sigmoid of a matrix A by rows. Specifically, supposing 
	 * that the input activation matrix is [a11, a12; a21, a22], the output 
	 * value is 
	 * <p>
	 * [exp(a11) / exp(a11) + exp(a12), exp(a12) / exp(a11) + exp(a12); 
	 * </br>
	 * exp(a21) / exp(a21) + exp(a22), exp(a22) / exp(a21) + exp(a22)].
	 * 
	 * @param res resulted matrix
	 * 
	 * @param A a real matrix
	 * 
	 */
	public static void sigmoid(Matrix res, Matrix A) {
		if (res instanceof DenseMatrix) {
			assign(res, A);
			double[][] data = ((DenseMatrix) res).getData();
			int M = A.getRowDimension();
			int N = A.getColumnDimension();
			double[] row_i = null;
			double old = 0;
			double current = 0;
			double max = 0;
			double sum = 0;
			double v = 0;
			for (int i = 0; i < M; i++) {
				row_i = data[i];
				old = row_i[0];
				current = 0;
				max = old;
				for (int j = 1; j < N; j++) {
					current = row_i[j];
					if (max < current)
						max = current;
					old = current;
				}
				sum = 0;
				for (int j = 0; j < N; j++) {
					v = Math.exp(row_i[j] - max);
					sum += v;
					row_i[j] = v;
				}
				for (int j = 0; j < N; j++) {
					row_i[j] /= sum; 
				}
			}
		} else {
			err("Sorry, sparse matrix is not support for res.");
			exit(1);
		}
	}
}
