package ml.classification;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static ml.utils.ArrayOperator.allocateVector;
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
import ml.utils.ArrayOperator;
import ml.utils.Printer;

/***
 * A Java implementation of fast linear multi-class SVM by 
 * Crammer-Singer formulation. It uses sparse representation of
 * the feature vectors, and updates the weight vectors and dual 
 * variables by dual coordinate descent. For heart_scale data, 
 * for C = 1.0 and eps = 1e-2, the average running time is 0.08 
 * seconds using an Intel(R) Core(TM) i7 CPU M620 @ 2.67GHz with 
 * 4.00GB memory and 64-bit Windows 7 operating system.
 * 
 * <p>
 * The memory complexity is O(l*d_s) + O(d*K) and the 
 * computation complexity is O(l*d_s), where d_s is the average 
 * number of non-zero features, d is the feature size, l is 
 * the training sample size, and K is the number of classes.
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Nov. 30th, 2013
 */
public class LinearMCSVM extends Classifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4808466628014511429L;

	/**
	 * @param args
	 * @throws InvalidInputDataException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, InvalidInputDataException {
		
		double C = 1.0;
		double eps = 1e-4;
		Classifier linearMCSVM = new LinearMCSVM(C, eps);

		double[][] data = { 
				{3.5, 4.4, 1.3, 2.3},
				{5.3, 2.2, 0.5, 4.5},
				{0.2, 0.3, 4.1, -3.1},
				{-1.2, 0.4, 3.2, 1.6}
				};

		int[] labels = new int[] {1, 2, 3, 4};
		
		linearMCSVM.feedData(data);
		linearMCSVM.feedLabels(labels);
		linearMCSVM.train();
		fprintf("W:%n");
		printMatrix(linearMCSVM.W);
		fprintf("b:%n");
		printVector(linearMCSVM.b);
		int[] pred_labels = linearMCSVM.predict(data);
		getAccuracy(pred_labels, labels);
		
		// Get elapsed time in seconds
		long start = System.currentTimeMillis();
		
		String filePath = "heart_scale";
		C = 1;
		eps = 0.01;
		linearMCSVM = new LinearMCSVM(C, eps);
		DataSet dataSet = DataSet.readDataSetFromFile(filePath);
		linearMCSVM.feedData(dataSet.X);
		linearMCSVM.feedLabels(dataSet.Y);
		linearMCSVM.train();
		
		fprintf("W:%n");
		printMatrix(linearMCSVM.W);
		fprintf("b:%n");
		printVector(linearMCSVM.b);
		
		Matrix XTest = dataSet.X;
		pred_labels = linearMCSVM.predict(XTest);
		getAccuracy(pred_labels, linearMCSVM.labels);
		
		System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);
		
	}

	/**
	 * Parameter for loss term.
	 */
	double C;
	
	/**
	 * Convergence tolerance.
	 */
	double eps;
	
	public LinearMCSVM() {
		C = 1.0;
		eps = 1e-2;
	}
	
	public LinearMCSVM(double C, double eps) {
		super();
		this.C = C;
		this.eps = eps;
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
		
		// Vector[] Xs = getVectors(X);
		
		double[] pr_CSR = null;
		if (X instanceof SparseMatrix) {
			double[] pr = ((SparseMatrix) X).getPr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			int nnz = ((SparseMatrix) X).getNNZ();
			pr_CSR = ArrayOperator.allocate1DArray(nnz);
			for (int k = 0; k < nnz; k++) {
				pr_CSR[k] = pr[valCSRIndices[k]];
			}
		}
		
		double[][] Ws = new double[nClass][];
		for (int c = 0; c < nClass; c++) {
			Ws[c] = allocateVector(nFeature + 1, 0);
		}
		double[] Q = computeQ(X, pr_CSR);
		
		double[][] Alpha = new DenseMatrix(nExample, nClass, 0).getData();
		
		int Np = nExample * nClass;
		
		double M = Double.NEGATIVE_INFINITY;
		double m = Double.POSITIVE_INFINITY;
		double Grad = 0;
		double alpha_old = 0;
		double alpha_new = 0;
		double PGrad = 0;
		int i, p, q;
		int C = nClass;
		int[] y = labelIDs;
		double delta = 0;
		int cnt = 1;
		while (true) {
			M = Double.NEGATIVE_INFINITY;
			m = Double.POSITIVE_INFINITY;
			for (int k = 0; k < Np; k++) {
				q = k % C;
				i = (k - q) / C;
				p = y[i];
				if (q == p) {
					continue;
				}
				// G = K(i, :) * (Beta(:, p) - Beta(:, q)) - 1;
				// G = Xs{i}' * (W(:, p) - W(:, q)) - 1;
				// Grad = computeGradient(Xs[i], Ws[p], Ws[q]);
				// Grad = computeGradient(X, i, Ws[p], Ws[q]);
				Grad = computeGradient(X, i, pr_CSR, Ws[p], Ws[q]);
				alpha_old = Alpha[i][q];
				if (alpha_old == 0) {
					PGrad = min(Grad, 0);
				} else if (alpha_old == C) {
					PGrad = max(Grad, 0);
				} else {
					PGrad = Grad;
				}
				M = max(M, PGrad);
				m = min(m, PGrad);
				if (PGrad != 0) {
					alpha_new = min(max(alpha_old - Grad / Q[i], 0), C);
					Alpha[i][q] = alpha_new;
					delta = alpha_new - alpha_old;
					// Beta[i][p] += delta;
					// Beta[i][q] -= delta;
					// W(:, p) = W(:, p) + (Alpha(i, q) - alpha) * Xs{i};
		            // W(:, q) = W(:, q) - (Alpha(i, q) - alpha) * Xs{i};
					// updateW(Ws[p], Ws[q], delta, Xs[i]);
					updateW(Ws[p], Ws[q], delta, X, i, pr_CSR);
				}
				
			}
			if (cnt % 20 == 0)
				Printer.fprintf(".");
			if (cnt % 400 == 0)
				Printer.fprintf("%n");
			cnt++;
			if (Math.abs(M - m) <= eps) {
				Printer.fprintf("%n");
				break;
			}
			
		}
		
		double[][] weights = new double[nClass][];
		b = new double[nClass];
		for (int c = 0; c < nClass; c++) {
			weights[c] = new double[nFeature];
			System.arraycopy(Ws[c], 0, weights[c], 0, nFeature);
			b[c] = Ws[c][nFeature];
		}
		this.W = new DenseMatrix(weights).transpose();
		
		/*W = XT.mtimes(new DenseMatrix(Beta));
		b = operate(allocateVector(l, 1.0), Beta);*/
		// b = ((DenseMatrix) new DenseMatrix(1, l, 1.0).mtimes(new DenseMatrix(Beta))).getData()[0];
		
	}

	private double[] computeQ(Matrix X, double[] pr_CSR) {
		int l = X.getRowDimension();
		double[] Q = new double[l];
		double s = 0;
		double v = 0;
		int M = X.getRowDimension();
		int N = X.getColumnDimension();
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			for (int i = 0; i < M; i++) {
				XRow = XData[i];
				s = 1;
				for (int j = 0; j < N; j++) {
					v = XRow[j];
					s += v * v;
				}
				Q[i] = 2 * s;
			}
		} else if (X instanceof SparseMatrix) {
			// int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			// int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			// double[] pr = ((SparseMatrix) X).getPr();
			for (int i = 0; i < M; i++) {
				s = 1;
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					// v = pr[valCSRIndices[k]];
					v = pr_CSR[k];
					s += v * v;
				}
				Q[i] = 2 * s;
			}
		}
		return Q;
	}

	/**
	 * Update the p-th and q-th weight vectors of W by the formula</br>
	 * W(:, p) = W(:, p) + delta_alpha_iq * [X(i, :) 1]'</br>
     * W(:, q) = W(:, q) - delta_alpha_iq * [X(i, :) 1]'</br>
	 * @param Wp
	 * @param Wq
	 * @param delta
	 * @param X training samples
	 * @param i sample index
	 * @param pr_CSR 
	 */
	private void updateW(double[] Wp, double[] Wq, double delta, Matrix X, int i, double[] pr_CSR) {
		int N = X.getColumnDimension();
		double v = 0;
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			XRow = XData[i];
			for (int j = 0; j < N; j++) {
				// W[j] += v * XRow[j];
				v = delta * XRow[j];
				Wp[j] += v;
				Wq[j] -= v;
			}
			// W[N] += v;
			Wp[N] += delta;
			Wq[N] -= delta;
		} else if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			// int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			// double[] pr = ((SparseMatrix) X).getPr();
			int idx = 0;
			for (int k = jr[i]; k < jr[i + 1]; k++) {
				// W[ic[k]] += v * pr[valCSRIndices[k]];
				idx = ic[k];
				// v = delta * pr[valCSRIndices[k]];
				v = delta * pr_CSR[k];
				Wp[idx] += v;
				Wq[idx] -= v;
			}
			// W[N] += v;
			Wp[N] += delta;
			Wq[N] -= delta;
		}
	}

	/**
	 * Compute the gradient by the formula
	 * G = [X(i, :) 1]' * (W(:, p) - W(:, q)) - 1.
	 * 
	 * @param X training samples
	 * 
	 * @param i sample index
	 * 
	 * @param pr_CSR pr array in CSR format
	 * 
	 * @param Wp weight vector for the p-th class
	 * 
	 * @param Wq weight vector for the p-th class
	 * 
	 * @return gradient
	 */
	private double computeGradient(Matrix X, int i, double[] pr_CSR, double[] Wp, double[] Wq) {
		int N = X.getColumnDimension();
		double res = 0;
		double s = -1;
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			XRow = XData[i];
			for (int j = 0; j < N; j++) {
				s += (Wp[j] - Wq[j]) * XRow[j];
			}
			res = s + Wp[N] - Wq[N];
		} else if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			// int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			// double[] pr = ((SparseMatrix) X).getPr();
			int idx = 0;
			for (int k = jr[i]; k < jr[i + 1]; k++) {
				idx = ic[k];
				// s += (Wp[idx] - Wq[idx]) * pr[valCSRIndices[k]];
				s += (Wp[idx] - Wq[idx]) * pr_CSR[k];
			}
			res = s + Wp[N] - Wq[N];
		}
		return res;
	}

	@Override
	public Matrix predictLabelScoreMatrix(Matrix Xt) {
		int n = Xt.getRowDimension();
		Matrix ScoreMatrix = null;
		Matrix Bias = new DenseMatrix(n, 1, 1.0).mtimes(new DenseMatrix(b, 2));
		ScoreMatrix = Xt.mtimes(W).plus(Bias);
		return ScoreMatrix;
	}

}
