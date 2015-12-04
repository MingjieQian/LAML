package ml.classification;

import static ml.utils.ArrayOperator.plusAssign;
import static ml.utils.InPlaceOperator.assign;
import static ml.utils.InPlaceOperator.clear;
import static ml.utils.InPlaceOperator.log;
import static ml.utils.InPlaceOperator.minus;
import static ml.utils.InPlaceOperator.plus;
import static ml.utils.InPlaceOperator.times;
import static ml.utils.InPlaceOperator.timesAssign;
import static ml.utils.Matlab.eps;
import static ml.utils.Matlab.sigmoid;
import static ml.utils.Matlab.sum;
import static ml.utils.Printer.disp;
import static ml.utils.Printer.fprintf;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Printer.printVector;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import la.io.DataSet;
import la.io.InvalidInputDataException;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import ml.optimization.BoundConstrainedPLBFGS;
import ml.utils.ArrayOperator;

/**
 * Multi-class logistic regression by using limited-memory BFGS method.
 * <p/>
 * We aim to minimize the cross-entropy error function defined by
 * <p/>
 * E(W) = -ln{p(T|w1, w2,..., wK)} / N = -sum_n{sum_k{t_{nk}ln(v_nk)}} / N,
 * <p/>where \nabla E(W) = X * (V - T) / N and v_nk = P(C_k|x_n).
 * 
 * @version 1.0 Jan. 23rd, 2014
 * 
 * @author Mingjie Qian
 */
public class LogisticRegressionByBoundConstrainedPLBFGS extends Classifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7298354335228192961L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = {
				{3.5, 5.3, 0.2, -1.2},
				{4.4, 2.2, 0.3, 0.4},
				{1.3, 0.5, 4.1, 3.2}
			    };
		/*double[][] data = { 
				{3.5, 4.4, 1.3, 2.3},
				{5.3, 2.2, 0.5, 4.5},
				{0.2, 0.3, 4.1, -3.1},
				{-1.2, 0.4, 3.2, 1.6}
				};*/

		int[] labels = new int[] {1, 2, 3};
		
		Classifier logReg = new LogisticRegressionByBoundConstrainedPLBFGS();
		logReg.epsilon = 1e-5;
		logReg.feedData(data);
		logReg.feedLabels(labels);
		
		// Get elapsed time in seconds
		long start = System.currentTimeMillis();
		logReg.train();
		System.out.format("Elapsed time: %.3f seconds.%n", (System.currentTimeMillis() - start) / 1000F);

		fprintf("W:%n");
		printMatrix(logReg.W);
		fprintf("b:%n");
		printVector(logReg.b);
		
		double[][] dataTest = data;
		
		fprintf("Ground truth:%n");
		printMatrix(logReg.Y);
		fprintf("Predicted probability matrix:%n");
		Matrix Prob_pred = logReg.predictLabelScoreMatrix(dataTest);
		disp(Prob_pred);
		fprintf("Predicted label matrix:%n");
		Matrix Y_pred = logReg.predictLabelMatrix(dataTest);
		printMatrix(Y_pred);
		int[] pred_labels = logReg.predict(dataTest);
		getAccuracy(pred_labels, labels);
		
		start = System.currentTimeMillis();
		String filePath = "heart_scale";
		logReg = new LogisticRegressionByBoundConstrainedPLBFGS();
		DataSet dataSet = null;
		try {
			dataSet = DataSet.readDataSetFromFile(filePath);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		logReg.feedData(dataSet.X);
		logReg.feedLabels(dataSet.Y);
		logReg.train();

		Matrix XTest = dataSet.X;
		pred_labels = logReg.predict(XTest);
		getAccuracy(pred_labels, logReg.labels);

		System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);

	}
	
	public LogisticRegressionByBoundConstrainedPLBFGS() {
	}

	@Override
	public void loadModel(String filePath) {
		
		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (Matrix)ois.readObject();
			b = (double[])ois.readObject(); 
			IDLabelMap = (int[])ois.readObject();
			nClass = IDLabelMap.length;
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
			oos.writeObject(b);
			oos.writeObject(IDLabelMap);
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	@Override
	public void train() {
		
		Matrix A = null;
		Matrix V = null;
		Matrix G = null;
		// Matrix XT = getXT(X);
		
		double fval = 0;
		
		/* Minimize the cross-entropy error function defined by
		 * E (W) = −ln p (T|w1,w2, · · · ,wK) / nSample
		 * Gradient: G = X * (V - Y) / nSample
		 */
		// W = repmat(zeros(nFea, 1), new int[]{1, K});
		DenseMatrix W = new DenseMatrix(nFeature + 1, nClass);
		A = new DenseMatrix(nExample, nClass);
		// A = X.transpose().multiply(W);
		computeActivation(A, X, W);
		V = sigmoid(A);
		G = W.copy();
		
		// G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nSample);
		Matrix  VMinusY = new DenseMatrix(nExample, nClass);
		minus(VMinusY, V, Y);
		// mtimes(G, XT, VMinusY);
		computeGradient(G, X, VMinusY);
		timesAssign(G, 1.0 / nExample);
		
		
		// fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample;
		Matrix YLogV = new DenseMatrix(nExample, nClass);
		Matrix VPlusEps = new DenseMatrix(nExample, nClass);
		Matrix LogV = new DenseMatrix(nExample, nClass);
		plus(VPlusEps, V, eps);
		log(LogV, VPlusEps);
		times(YLogV, Y, LogV);
		fval = -sum(sum(YLogV)) / nExample;
		
		boolean flags[] = null;
		while (true) {
			flags = BoundConstrainedPLBFGS.run(G, fval, 0, 1, epsilon, W);
			if (flags[0])
				break;
			// A = X.transpose().multiply(W);
			computeActivation(A, X, W);
			V = sigmoid(A);
			// fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nSample;
			plus(VPlusEps, V, eps);
			log(LogV, VPlusEps);
			times(YLogV, Y, LogV);
			fval = -sum(sum(YLogV)) / nExample;
			if (flags[1]) {
				// G = rdivide(X.multiply(V.subtract(Y)), nSample);
				minus(VMinusY, V, Y);
				// mtimes(G, XT, VMinusY);
				computeGradient(G, X, VMinusY);
				timesAssign(G, 1.0 / nExample);
			}
		}
		double[][] WData = W.getData();
		double[][] thisWData = new double[nFeature][];
		for (int feaIdx = 0; feaIdx < nFeature; feaIdx++) {
			thisWData[feaIdx] = WData[feaIdx];
		}
		this.W = new DenseMatrix(thisWData);
		b = WData[nFeature];
		
	}

	/**
	 * G = [A ones(nSample, 1)]' * B.
	 * 
	 * @param A
	 * @param B
	 */
	private void computeGradient(Matrix res, Matrix A, Matrix B) {
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
		}
	}

	@SuppressWarnings("unused")
	private Matrix getXT(Matrix X) {
		// XT = [X'; ones(1, nSample)]
		if (X instanceof DenseMatrix) {
			double[][] resData = ArrayOperator.allocate2DArray(nFeature + 1, nExample, 0);
			double[] resRow = null;
			double[][] XData = ((DenseMatrix) X).getData();
			for (int feaIdx = 0; feaIdx < nFeature; feaIdx++) {
				resRow = resData[feaIdx];
				for (int sampleIdx = 0; sampleIdx < nExample; sampleIdx++) {
					resRow[sampleIdx] = XData[sampleIdx][feaIdx];
				}
			}
			resRow = resData[nFeature];
			assign(resRow, 1.0);
			return new DenseMatrix(resData);
		} else if (X instanceof SparseMatrix) {
			SparseMatrix res = (SparseMatrix) X.transpose();
			res.appendAnEmptyRow();
			for (int sampleIdx = 0; sampleIdx < nExample; sampleIdx++) {
				res.setEntry(nFeature, sampleIdx, 1.0);
			}
			return res;
		}
		return null;
	}

	private void computeActivation(Matrix A, Matrix X, Matrix W) {
		// A = [X ones(nSample, 1)] * W
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
	}

	@Override
	public Matrix predictLabelScoreMatrix(Matrix Xt) {
		DenseMatrix ScoreMatrix = (DenseMatrix) Xt.mtimes(W);
		double[][] scoreData = ScoreMatrix.getData();
		for (int i = 0; i < Xt.getRowDimension(); i++) {
			plusAssign(scoreData[i], b);
		}
		return sigmoid(ScoreMatrix);
	}

}
