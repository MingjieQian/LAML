package ml.regression;

import static ml.utils.ArrayOperator.allocate1DArray;
import static ml.utils.ArrayOperator.allocate2DArray;
import static ml.utils.ArrayOperator.innerProduct;
import static ml.utils.ArrayOperator.sum;
import static ml.utils.Matlab.full;
import static ml.utils.Printer.display;
import static ml.utils.Printer.errf;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printf;
import static ml.utils.Time.tic;
import static ml.utils.Time.toc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import ml.options.Options;

/***
 * A Java implementation of linear regression, which solves the 
 * following convex optimization problem:
 * </p>
 * min_W 2\1 || Y - X * W ||_F^2 + lambda * || W ||_F^2</br>
 * where X is an n-by-p data matrix with each row bing a p
 * dimensional data vector and Y is an n-by-ny dependent
 * variable matrix.
 * 
 * @author Mingjie Qian
 * @version 1.0 May 2nd, 2015
 */
public class LinearRegression extends Regression {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = {
				{1, 2, 3, 2},
				{4, 2, 3, 6},
				{5, 1, 4, 1}
				}; 

		double[][] depVars = {
				{3, 2},
				{2, 3},
				{1, 4}
				};

		Options options = new Options();
		options.maxIter = 600;
		options.lambda = 0.1;
		options.verbose = !true;
		options.calc_OV = !true;
		options.epsilon = 1e-5;

		Regression LR = new LinearRegression(options);
		LR.feedData(data);
		LR.feedDependentVariables(depVars);

		tic();
		LR.train();
		fprintf("Elapsed time: %.3f seconds\n\n", toc());

		fprintf("Projection matrix:\n");
		display(LR.W);
		
		fprintf("Bias vector:\n");
		display(((LinearRegression)LR).B);

		Matrix Yt = LR.predict(data);
		fprintf("Predicted dependent variables:\n");
		display(Yt);

	}

	/**
	 * Regularization parameter.
	 */
	private double lambda;

	/**
	 * If compute objective function values during
	 * the iterations or not.
	 */
	private boolean calc_OV;

	/**
	 * If show computation detail during iterations or not.
	 */
	private boolean verbose;

	/**
	 * Bias vector.
	 */
	public double[] B;

	public LinearRegression() {
	}

	public LinearRegression(double epsilon) {
		super(epsilon);

	}

	public LinearRegression(int maxIter, double epsilon) {
		super(maxIter, epsilon);

	}

	public LinearRegression(Options options) {
		super(options);
		lambda = options.lambda;
		calc_OV = options.calc_OV;
		verbose = options.verbose;
	}

	private double train(Matrix X, double[] y, double[] w0, double b0) {
		
		double[] w = w0;
		double b = b0;
		double[] y_hat = new double[n];
		double[] e = new double[n];
		double[] OFVs = null;
		boolean debug = !true;
		int blockSize = 10;
		if (calc_OV && verbose) {
			OFVs = allocate1DArray(maxIter + 1, 0);
			double ofv = 0;
			ofv = computeOFV(y, w, b);
			OFVs[0] = ofv;
			fprintf("Iter %d: %.10g\n", 0, ofv);
		}

		int cnt = 0;
		double ofv_old = 0;
		double ofv_new = 0;

		if (X instanceof SparseMatrix) {

			int[] ic = ((SparseMatrix) X).getIc();
			int[] ir = ((SparseMatrix) X).getIr();
			int[] jc = ((SparseMatrix) X).getJc();
			int[] jr = ((SparseMatrix) X).getJr();
			double[] pr = ((SparseMatrix) X).getPr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();

			// Compute y_hat and cache e
			for (int r = 0; r < n; r++) {
				double s = b;
				// Compute <W, X[r, :]>
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					int j = ic[k];
					s += w[j] * pr[valCSRIndices[k]];
				}
				y_hat[r] = s;
				e[r] = y[r] - s; // Why recalculated e[r] != e[r] updated till the end in last iteration
			}

			while (true) {

				// Update b
				ofv_old = 0;
				if (debug) {
					ofv_old = computeOFV(y, w, b);
					printf("f(b): %f\n", ofv_old);
				}
				double b_new = (b * n + sum(e)) / (n + lambda);
				for (int i = 0; i < n; i++)
					e[i] -= (b_new - b);
				b = b_new;
				// println("b = " + b);

				if (debug) {
					ofv_new = computeOFV(y, w, b);
					printf("b updated: %f\n", ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating b\n");
					}
				}

				// Update w
				for (int j = 0; j < p; j++) {

					ofv_old = 0;
					/*if (debug) {
					ofv_old = computOFV();
					printf("f(w[%d]): %f\n", j, ofv_old);
				}*/
					/*
					 * v1 = \sum_i h^2(x_i)
					 */
					double v1 = 0;
					/*
					 * v2 = \sum_i h(x_i) * e[i]
					 */
					double v2 = 0;
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						int i = ir[k];
						double xj = pr[k];
						double hj = xj;
						v1 += hj * hj;
						v2 += hj * e[i];
					}
					// Update w[j]
					double wj_new = (w[j] * v1 + v2) / (v1 + lambda);
					if (Double.isInfinite(wj_new)) {
						int a = 1;
						a = a + 1;
					}
					// e[i] for X[i,j] != 0
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						int i = ir[k];
						double xj = pr[k];
						// double t = X.getEntry(i, j);
						e[i] -= (wj_new - w[j]) * xj;
					}
					w[j] = wj_new;

					if (debug) {
						ofv_new = computeOFV(y, w, b);
						printf("w[%d] updated: %f\n", j, ofv_new);
						if (ofv_old < ofv_new) {
							errf("Error when updating w[%d]\n", j);
						}
					}

				}

				cnt++;
				if (verbose) {
					if (calc_OV) {
						double ofv = computeOFV(y, w, b);
						OFVs[cnt] = ofv;
						if (cnt % blockSize == 0)
							fprintf(".Iter %d: %.8g\n", cnt, ofv);
						else
							fprintf(".");
					} else {
						if (cnt % blockSize == 0)
							fprintf(".Iter %d\n", cnt);
						else
							fprintf(".");
					}
				}
				if (cnt >= maxIter) {
					break;
				}
			}

		} else if (X instanceof DenseMatrix) {
			
			double[][] data = X.getData();
			
			// Compute y_hat and cache e
			for (int r = 0; r < n; r++) {
				double s = b;
				// Compute <W, X[r, :]>
				s += innerProduct(w, data[r]);
				y_hat[r] = s;
				e[r] = y[r] - s; // Why recalculated e[r] != e[r] updated till the end in last iteration
			}

			while (true) {

				// Update b
				ofv_old = 0;
				if (debug) {
					ofv_old = computeOFV(y, w, b);
					printf("f(b): %f\n", ofv_old);
				}
				double b_new = (b * n + sum(e)) / (n + lambda);
				for (int i = 0; i < n; i++)
					e[i] -= (b_new - b);
				b = b_new;
				// println("b = " + b);

				if (debug) {
					ofv_new = computeOFV(y, w, b);
					printf("b updated: %f\n", ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating b\n");
					}
				}

				// Update w
				for (int j = 0; j < p; j++) {

					ofv_old = 0;
					/*if (debug) {
					ofv_old = computOFV();
					printf("f(w[%d]): %f\n", j, ofv_old);
				}*/
					/*
					 * v1 = \sum_i h^2(x_i)
					 */
					double v1 = 0;
					/*
					 * v2 = \sum_i h(x_i) * e[i]
					 */
					double v2 = 0;
					for (int i = 0; i < n; i++) {
						double xj = data[i][j];
						double hj = xj;
						v1 += hj * hj;
						v2 += hj * e[i];
					}
					/*for (int k = jc[j]; k < jc[j + 1]; k++) {
						int i = ir[k];
						double xj = pr[k];
						double hj = xj;
						v1 += hj * hj;
						v2 += hj * e[i];
					}*/
					// Update w[j]
					double wj_new = (w[j] * v1 + v2) / (v1 + lambda);
					if (Double.isInfinite(wj_new)) {
						int a = 1;
						a = a + 1;
					}
					// e[i] for X[i,j] != 0
					for (int i = 0; i < n; i++) {
						double xj = data[i][j];
						e[i] -= (wj_new - w[j]) * xj;
					}
					/*for (int k = jc[j]; k < jc[j + 1]; k++) {
						int i = ir[k];
						double xj = pr[k];
						// double t = X.getEntry(i, j);
						e[i] -= (wj_new - w[j]) * xj;
					}*/
					w[j] = wj_new;

					if (debug) {
						ofv_new = computeOFV(y, w, b);
						printf("w[%d] updated: %f\n", j, ofv_new);
						if (ofv_old < ofv_new) {
							errf("Error when updating w[%d]\n", j);
						}
					}

				}

				cnt++;
				if (verbose) {
					if (calc_OV) {
						double ofv = computeOFV(y, w, b);
						OFVs[cnt] = ofv;
						if (cnt % blockSize == 0)
							fprintf(".Iter %d: %.8g\n", cnt, ofv);
						else
							fprintf(".");
					} else {
						if (cnt % blockSize == 0)
							fprintf(".Iter %d\n", cnt);
						else
							fprintf(".");
					}
				}
				if (cnt >= maxIter) {
					break;
				}
			}
			
		}

		return b;

	}

	@Override
	public void train() {
		double[][] ws = allocate2DArray(ny, p, 0);
		B = allocate1DArray(ny, 0);
		for (int k = 0; k < ny; k++) {
			B[k] = train(X, full(Y.getColumnVector(k)).getPr(), ws[k], B[k]);
		}
		W = new DenseMatrix(ws).transpose();
	}

	@Override
	public void train(Matrix W0) {
		double[][] ws = W0.transpose().getData();
		B = allocate1DArray(ny, 0);
		for (int k = 0; k < ny; k++) {
			B[k] = train(X, full(Y.getColumnVector(k)).getPr(), ws[k], B[k]);
		}
		W = new DenseMatrix(ws).transpose();
	}

	@Override
	public Matrix train(Matrix X, Matrix Y) {
		String Method = "Linear Regression";
		System.out.printf("Training %s...\n", Method);

		double[][] ws = allocate2DArray(ny, p, 0);
		B = allocate1DArray(ny, 0);
		for (int k = 0; k < ny; k++) {
			B[k] = train(X, full(Y.getColumnVector(k)).getPr(), ws[k], B[k]);
		}
		W = new DenseMatrix(ws).transpose();
		return W;
	}

	private double computeOFV(double[] y, double[] w, double b) {
		double ofv = 0;
		ofv += lambda * b * b;
		ofv += lambda * innerProduct(w, w);
		int[] ic = ((SparseMatrix) X).getIc();
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		// compute y_hat
		for (int r = 0; r < n; r++) {
			double s = b;
			// Compute <W, X[r, :]>
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
				if (Double.isNaN(s)) {
					int a = 1;
					a = a + 1;
				}
			}
			double e = (y[r] - s);
			ofv += e * e;
		}
		return ofv;
	}

	@Override
	public Matrix train(Matrix X, Matrix Y, Matrix W0) {
		double[][] ws = W0.transpose().getData();
		B = allocate1DArray(ny, 0);
		for (int k = 0; k < ny; k++) {
			B[k] = train(X, full(Y.getColumnVector(k)).getPr(), ws[k], B[k]);
		}
		W = new DenseMatrix(ws).transpose();
		return W;
	}

	@Override
	public void loadModel(String filePath) {

		// System.out.println("Loading regression model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (Matrix)ois.readObject();
			B = (double[])ois.readObject();
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

	}

	@Override
	public void saveModel(String filePath) {

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.writeObject(B);
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
