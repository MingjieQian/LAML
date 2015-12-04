package ml.temporal;

import static ml.utils.ArrayOperator.plusAssign;
import static ml.utils.ArrayOperator.timesAssign;
import static ml.utils.InPlaceOperator.assign;
import static ml.utils.InPlaceOperator.clear;
import static ml.utils.InPlaceOperator.log;
import static ml.utils.InPlaceOperator.minus;
import static ml.utils.InPlaceOperator.mtimes;
import static ml.utils.InPlaceOperator.plus;
import static ml.utils.InPlaceOperator.sigmoid;
import static ml.utils.InPlaceOperator.times;
import static ml.utils.InPlaceOperator.timesAssign;
import static ml.utils.Matlab.abs;
import static ml.utils.Matlab.eps;
import static ml.utils.Matlab.innerProduct;
import static ml.utils.Matlab.max;
import static ml.utils.Matlab.norm;
import static ml.utils.Matlab.ones;
import static ml.utils.Matlab.sigmoid;
import static ml.utils.Matlab.speye;
import static ml.utils.Matlab.sum;
import static ml.utils.Matlab.sumAll;

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
import ml.optimization.AcceleratedProximalGradient;
import ml.optimization.BoundConstrainedPLBFGS;
import ml.optimization.LBFGS;
import ml.optimization.Prox;
import ml.optimization.ProxL1;
import ml.optimization.ProxL2;
import ml.optimization.ProxL2Square;
import ml.optimization.ProxLInfinity;
import ml.options.Options;
import ml.utils.Matlab;
import ml.utils.Printer;

public class TemporalLogisticRegression {

	public static void main(String[] args) {

	}
	
	/**
	 * Regularization type.
	 * 0:  No regularization
	 * 1:  L1 regularization
	 * 2:  L2^2 regularization
	 * 3:  L2 regularization
	 * 4:  Infinity norm regularization
	 */
	public int regularizationType = 1;
	
	/**
	 * Training data matrix (nExample x nFeature),
	 * each row is a feature vector. The data 
	 * matrix should not include bias dummy features.
	 */
	protected Matrix X;
	
	/**
	 * An n x 1 matrix for the observation time.
	 */
	protected Matrix T;
	
	/**
	 * Label matrix for training (nExample x nClass).
	 * Y_{i,k} = 1 if x_i belongs to class k, and 0 otherwise.
	 */
	protected Matrix Y;
	
	/**
	 * Projection matrix (nFeature x nClass), column i is the projector for class i.
	 */
	public Matrix W;
	
	/**
	 * The biases in the linear model.
	 */
	public double[] b;
	
	public double rho;
	
	public Options options = new Options();
	
	private int nClass = 2;
	
	/**
	 * Number of features, without bias dummy features,
	 * i.e., for SVM.
	 */
	protected int nFeature;
	
	/**
	 * Number of examples.
	 */
	protected int nExample;
	
	public TemporalLogisticRegression(double lambda) {
		options.lambda = lambda;
	}
	
	public TemporalLogisticRegression(Options options) {
		this.options = options;
	}
	
	public TemporalLogisticRegression() {
	}

	public void initialize(double rho0) {
		this.rho = rho0;
	}
	
	public void initialize(double... params) {
		if (params.length == 3) {
			rho = params[1];
		} else {
			rho = params[0];
		}
	}
	
	public void feedData(Matrix X) {
		this.X = X;
		nExample = X.getRowDimension();
		nFeature = X.getColumnDimension();
	}
	
	public void feedTime(Matrix T) {
		this.T = T;
	}
	
	public void feedScore(int[] labels) {
		int n = labels.length;
		Y = new DenseMatrix(n, 2, 0);
		for (int i = 0; i < n; i++) {
			if (labels[i] == 1)
				Y.setEntry(i, 0, 1);
			else if (labels[i] == 0)
				Y.setEntry(i, 1, 1);
		}
	}
	
	public void feedScore(double[] scores) {
		int n = scores.length;
		Y = new DenseMatrix(n, 2, 0);
		for (int i = 0; i < n; i++) {
			if (scores[i] == 1.0)
				Y.setEntry(i, 0, 1);
			else if (scores[i] == 0.0)
				Y.setEntry(i, 1, 1);
		}
	}
	
	public void feedScore(Matrix V) {
		int n = V.getRowDimension();
		Y = new DenseMatrix(n, 2, 0);
		for (int i = 0; i < n; i++) {
			if (V.getEntry(i, 0) == 1.0)
				Y.setEntry(i, 0, 1);
			else if (V.getEntry(i, 0) == 0.0)
				Y.setEntry(i, 1, 1);
		}
	}

	public void train() {
		
		double lambda = options.lambda;
		double epsilon = options.epsilon;
		int maxIter = options.maxIter;
		// maxIter = 3;
		// DenseMatrix W = (DenseMatrix) rand(nFeature + 1, nClass);
		DenseMatrix W = (DenseMatrix) ones(nFeature + 1, nClass);
		Matrix A = new DenseMatrix(nExample, nClass);
		Matrix V = A.copy();
		Matrix G = W.copy();
		Matrix XW = new DenseMatrix(nExample, nClass);
		Matrix  VMinusY = new DenseMatrix(nExample, nClass);
		Matrix YLogV = new DenseMatrix(nExample, nClass);
		Matrix VPlusEps = new DenseMatrix(nExample, nClass);
		Matrix LogV = new DenseMatrix(nExample, nClass);
		
		// Matrix XT = getXT(X);
		Matrix C = speye(nExample);
		for (int i = 0; i < nExample; i++) {
			C.setEntry(i, i, Y.getEntry(i, 0) == 0.0 ? 5 : 1);
		}
		
		double fval = 0;
		double hval = 0;
		double fval_pre = 0;
		Matrix Grad4Rho = new DenseMatrix(0.0);
		Matrix Rho = new DenseMatrix(rho);
		int cnt = 0;
		while(true) {

			// Update W

			/* Minimize the cross-entropy error function defined by
			 * E (W) = −ln p (T|w1,w2,...,wK) / nSample
			 * Gradient: G = X * (V - Y) / nSample
			 */
			// W = repmat(zeros(nFea, 1), new int[]{1, K});
			// DenseMatrix W = new DenseMatrix(nFeature + 1, nClass);
			
			// A = X.transpose().multiply(W);
			computeActivation(A, X, W, T, rho);
			sigmoid(V, A);
			// G = W.copy();

			// G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nSample);
			minus(VMinusY, V, Y);
			// VMinusY = C.mtimes(VMinusY);
			mtimes(VMinusY, C, VMinusY);
			computeGradient(G, X, VMinusY, rho);
			timesAssign(G, 1.0 / nExample);


			// fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample;
			plus(VPlusEps, V, eps);
			log(LogV, VPlusEps);
			times(YLogV, Y, LogV);
			mtimes(YLogV, C, YLogV);
			// YLogV = C.mtimes(YLogV);
			fval = -sum(sum(YLogV)) / nExample;

			boolean flags[] = null;
			AcceleratedProximalGradient.type = 1;
			switch (regularizationType) {
			case 1:
				AcceleratedProximalGradient.prox = new ProxL1(lambda);
				break;
			case 2:
				AcceleratedProximalGradient.prox = new ProxL2Square(lambda);
				break;
			case 3:
				AcceleratedProximalGradient.prox = new ProxL2(lambda);
				break;
			case 4:
				AcceleratedProximalGradient.prox = new ProxLInfinity(lambda);
				break;
			default:
			}
			while (true) {
				if (regularizationType == 0) {
					flags = LBFGS.run(G, fval, epsilon, W);
				} else {
					flags = AcceleratedProximalGradient.run(G, fval, hval, epsilon, W);
				}

				/*disp("W:");
			disp(W);*/
				if (flags[0])
					break;
				// A = X.transpose().multiply(W);
				computeActivation(A, X, W, T, rho);
				/*disp("A:");
			disp(A);*/
				// V = sigmoid(A);
				sigmoid(V, A);
				// fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample;
				plus(VPlusEps, V, eps);
				/*disp("V:");
			disp(V);*/
				log(LogV, VPlusEps);
				times(YLogV, Y, LogV);
				// YLogV = C.mtimes(YLogV);
				mtimes(YLogV, C, YLogV);
				/*disp("YLogV:");
			disp(YLogV);*/
				fval = -sum(sum(YLogV)) / nExample;

				switch (regularizationType) {
				case 1:
					hval = lambda * sumAll(abs(W));
					break;
				case 2:
					double norm = norm(W, "fro");
					hval = lambda * norm * norm;
					break;
				case 3:
					hval = lambda * norm(W, "fro");
					break;
				case 4:
					hval = lambda * max(max(abs(W))[0])[0];
					break;
				default:
				}

				// fprintf("fval: %.4f\n", fval);
				if (flags[1]) {
					// G = rdivide(X.multiply(V.subtract(Y)), nSample);
					minus(VMinusY, V, Y);
					mtimes(VMinusY, C, VMinusY);
					// VMinusY = C.mtimes(VMinusY);
					computeGradient(G, X, VMinusY, rho);
					timesAssign(G, 1.0 / nExample);
				}
			}

			// Update rho
			// Grad4Rho.clear();
			Rho.setEntry(0, 0, rho);
			computeXW(XW, X, W, T);
			minus(VMinusY, V, Y);
			mtimes(VMinusY, C, VMinusY);
			Grad4Rho.setEntry(0, 0, innerProduct(VMinusY, XW) / nExample);

			hval = 0;

			// Compute gval
			assign(A, XW);
			timesAssign(A, rho);
			sigmoid(V, A);
			plus(VPlusEps, V, eps);
			log(LogV, VPlusEps);
			times(YLogV, Y, LogV);
			// YLogV = C.mtimes(YLogV);
			mtimes(YLogV, C, YLogV);
			double gval = -sum(sum(YLogV)) / nExample;

			AcceleratedProximalGradient.prox = new Prox();
			
			double l = -10;
			double u = 10;

			while (true) {
				// flags = AcceleratedProximalGradient.run(Grad4Rho, gval, hval, epsilon, Rho);
				flags = BoundConstrainedPLBFGS.run(Grad4Rho, gval, l, u, epsilon, Rho);
				rho = Rho.getEntry(0, 0);
				
				if (flags[0])
					break;

				assign(A, XW);
				timesAssign(A, rho);
				sigmoid(V, A);
				plus(VPlusEps, V, eps);
				log(LogV, VPlusEps);
				times(YLogV, Y, LogV);
				// YLogV = C.mtimes(YLogV);
				mtimes(YLogV, C, YLogV);
				gval = -sum(sum(YLogV)) / nExample;

				if (flags[1]) {
					minus(VMinusY, V, Y);
					mtimes(VMinusY, C, VMinusY);
					// VMinusY = C.mtimes(VMinusY);
					Grad4Rho.setEntry(0, 0, innerProduct(VMinusY, XW) / nExample);
				}

			}

			cnt++;
			fval = gval + lambda * norm(W, 1);
			Printer.fprintf("Iter %d - fval: %.4f\n", cnt, fval);
			if ( cnt > 1 && Math.abs(fval_pre - fval) < Matlab.eps)
				//break;
				fval_pre = fval;
			if (cnt >= maxIter)
				break;
		}
		
		double[][] WData = W.getData();
		double[][] thisWData = new double[nFeature][];
		for (int feaIdx = 0; feaIdx < nFeature; feaIdx++) {
			thisWData[feaIdx] = WData[feaIdx];
		}
		this.W = new DenseMatrix(thisWData);
		b = WData[nFeature];
	}

	public Matrix predict(Matrix Xt, Matrix Tt) {
		DenseMatrix ActivationMatrix = (DenseMatrix) Xt.mtimes(W);
		double[][] Activation = ActivationMatrix.getData();
		for (int i = 0; i < Xt.getRowDimension(); i++) {
			plusAssign(Activation[i], b);
			Activation[i][1] += Tt.getEntry(i, 0);
			timesAssign(Activation[i], rho);
		}
		return sigmoid(ActivationMatrix).getColumnMatrix(0);
	}
	
	/**
	 * G = [A ones(nExample, 1)]' * B.
	 * 
	 * @param A
	 * @param B
	 * @param rho 
	 */
	private void computeGradient(Matrix res, Matrix A, Matrix B, double rho) {
		if (res instanceof SparseMatrix) {
			
		} else if (res instanceof DenseMatrix) {
			double[][] resData = ((DenseMatrix) res).getData();
			double[] resRow = null;
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
					resRow = resData[M];
					clear(resRow);
					for (int k = 0; k < N; k++) {
						BRow = BData[k];
						for (int j = 0; j < NB; j++) {
							resRow[j] += BRow[j];
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
					double[] columnA = new double[A.getRowDimension()];
					for (int i = 0; i < M; i++) {
						for (int t = 0; t < N; t++) {
							columnA[t] = AData[t][i];
						}
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = 0;
							for (int k = jc[j]; k < jc[j + 1]; k++) {
								r = ir[k];
								// A[r][j] = pr[k]
								s += columnA[r] * pr[k];
							}
							resRow[j] = s;
						}
					}

					resRow = resData[M];
					for (int j = 0; j < NB; j++) {
						s = 0;
						for (int k = jc[j]; k < jc[j + 1]; k++) {
							// r = ir[k];
							// A[r][j] = pr[k]
							s += pr[k];
						}
						resRow[j] = s;
					}

				}
			} else if (A instanceof SparseMatrix) {
				
				if (B instanceof DenseMatrix) {
					int[] ir = ((SparseMatrix) A).getIr();
					int[] jc = ((SparseMatrix) A).getJc();
					double[] pr = ((SparseMatrix) A).getPr();
					// int[] valCSRIndices = ((SparseMatrix) A).getValCSRIndices();
					double[][] BData = ((DenseMatrix) B).getData();
					double[] BRow = null;
					int c = -1;
					double s = 0;
					for (int i = 0; i < M; i++) {
						resRow = resData[i];
						for (int j = 0; j < NB; j++) {
							s = 0;
							for (int k = jc[i]; k < jc[i + 1]; k++) {
								c = ir[k];
								s += pr[k] * BData[c][j];
							}
							resRow[j] = s;
						}
					}
					resRow = resData[M];
					clear(resRow);
					for (int k = 0; k < N; k++) {
						BRow = BData[k];
						for (int j = 0; j < NB; j++) {
							resRow[j] += BRow[j];
						}
					}
				} else if (B instanceof SparseMatrix) {
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
					resRow = resData[M];
					for (int j = 0; j < NB; j++) {
						s = 0;
						for (int k = jc2[j]; k < jc2[j + 1]; k++) {
							// r = ir[k];
							// A[r][j] = pr[k]
							s += pr2[k];
						}
						resRow[j] = s;
					}
				}
			}
			timesAssign(res, rho);
		}
	}
	
	private void computeActivation(Matrix A, Matrix X, Matrix W, Matrix T, double rho) {
		// A = [X ones(nExample, 1)] * W
		double[][] AData = ((DenseMatrix) A).getData();
		double[] ARow = null;
		double[][] WData = ((DenseMatrix) W).getData();
		double[] WColumn = new double[W.getRowDimension()];
		double[] WRow = null;
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			double s = 0;
			for (int j = 0; j < nClass; j++) {
				for (int r = 0; r < W.getRowDimension(); r++) {
					WColumn[r] = WData[r][j];
				}
				for (int i = 0; i < nExample; i++) {
					XRow = XData[i];
					s = 0;
					for (int k = 0; k < nFeature; k++) {
						s += XRow[k] * WColumn[k];
					}
					AData[i][j] = s + WColumn[nFeature];
				}
			}
		} else if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			double[] pr = ((SparseMatrix) X).getPr();
			int feaIdx = -1;
			double v = 0;
			for (int i = 0; i < nExample; i++) {
				ARow = AData[i];
				clear(ARow);
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					feaIdx = ic[k];
					WRow = WData[feaIdx];
					v = pr[valCSRIndices[k]];
					for (int j = 0; j < nClass; j++) {
						ARow[j] += v * WRow[j];
					}
				}
				WRow = WData[nFeature];
				for (int j = 0; j < nClass; j++) {
					ARow[j] += WRow[j];
				}
			}
		}
		// Add t_i for A[i][1];
		for (int i = 0; i < nExample; i++) {
			AData[i][1] += T.getEntry(i, 0);
		}
		timesAssign(A, rho);
	}
	
	private void computeXW(Matrix A, Matrix X, Matrix W, Matrix T) {
		// A = [X ones(nExample, 1)] * W
		double[][] AData = ((DenseMatrix) A).getData();
		double[] ARow = null;
		double[][] WData = ((DenseMatrix) W).getData();
		double[] WColumn = new double[W.getRowDimension()];
		double[] WRow = null;
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			double s = 0;
			for (int j = 0; j < nClass; j++) {
				for (int r = 0; r < W.getRowDimension(); r++) {
					WColumn[r] = WData[r][j];
				}
				for (int i = 0; i < nExample; i++) {
					XRow = XData[i];
					s = 0;
					for (int k = 0; k < nFeature; k++) {
						s += XRow[k] * WColumn[k];
					}
					AData[i][j] = s + WColumn[nFeature];
				}
			}
		} else if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			double[] pr = ((SparseMatrix) X).getPr();
			int feaIdx = -1;
			double v = 0;
			for (int i = 0; i < nExample; i++) {
				ARow = AData[i];
				clear(ARow);
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					feaIdx = ic[k];
					WRow = WData[feaIdx];
					v = pr[valCSRIndices[k]];
					for (int j = 0; j < nClass; j++) {
						ARow[j] += v * WRow[j];
					}
				}
				WRow = WData[nFeature];
				for (int j = 0; j < nClass; j++) {
					ARow[j] += WRow[j];
				}
			}
		}
		// Add t_i for A[i][1];
		for (int i = 0; i < nExample; i++) {
			AData[i][1] += T.getEntry(i, 0);
		}
	}

	public void loadModel(String filePath) {

		// System.out.println("Loading regression model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (Matrix)ois.readObject();
			b = (double[])ois.readObject();
			rho = ois.readDouble();
			options.lambda = ois.readDouble();
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

	public void saveModel(String filePath) {

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.writeObject(b);
			// oos.writeObject(new Double(rho));
			oos.writeDouble(rho);
			oos.writeDouble(options.lambda);
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
