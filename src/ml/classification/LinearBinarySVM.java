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
 * A Java implementation of fast linear binary SVM. It uses 
 * sparse representation of the feature vectors, and updates 
 * the weight vectors and dual variables by dual coordinate 
 * descent. For heart_scale data, for C = 1.0 and eps = 1e-2, 
 * the average running time is 0.04 seconds using an Intel(R) 
 * Core(TM) i7 CPU M620 @ 2.67GHz with 4.00GB memory and 
 * 64-bit Windows 7 operating system, even a little faster 
 * than liblinear, which costs 0.06 seconds in average with 
 * the same parameter.
 * 
 * <p>
 * The memory complexity is O(l*d_s) + O(d) + O(l*d_s) and the 
 * computation complexity is O(l*d_s), where d_s is the average 
 * number of non-zero features, d is the feature size, and l is 
 * the training sample size.
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Nov. 30th, 2013
 */
public class LinearBinarySVM extends Classifier {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2374085637018946130L;
	
	/**
	 * @param args
	 * @throws InvalidInputDataException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, InvalidInputDataException {

		double C = 1.0;
		double eps = 1e-4;
		Classifier linearBinarySVM = new LinearBinarySVM(C, eps);

		int[] pred_labels = null;
		double[][] data = { 
				{3.5, 4.4, 1.3, 2.3},
				{5.3, 2.2, 0.5, 4.5},
				{0.2, 0.3, 4.1, -3.1},
				{-1.2, 0.4, 3.2, 1.6}
				};

		int[] labels = new int[] {1, 1, -1, -1};
		
		linearBinarySVM.feedData(data);
		linearBinarySVM.feedLabels(labels);
		linearBinarySVM.train();
		fprintf("W:%n");
		printMatrix(linearBinarySVM.W);
		fprintf("b:%n");
		printVector(linearBinarySVM.b);
		pred_labels = linearBinarySVM.predict(data);
		getAccuracy(pred_labels, labels);
		
		// Get elapsed time in seconds
		long start = System.currentTimeMillis();

		String trainDataFilePath = "heart_scale";
		C = 1;
		eps = 0.01;
		linearBinarySVM = new LinearBinarySVM(C, eps);
		DataSet dataSet = DataSet.readDataSetFromFile(trainDataFilePath);
		linearBinarySVM.feedData(dataSet.X);
		linearBinarySVM.feedLabels(dataSet.Y);
		linearBinarySVM.train();

		Matrix XTest = dataSet.X;
		pred_labels = linearBinarySVM.predict(XTest);
		getAccuracy(pred_labels, linearBinarySVM.labels);

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

	public LinearBinarySVM() {
		C = 1.0;
		eps = 1e-2;
	}
	
	public LinearBinarySVM(double C, double eps) {
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
		
		double[] Y = new double[this.nExample];
		for (int i = 0; i < nExample; i++) {
			Y[i] = -2 * (labelIDs[i] - 0.5);
		}
		
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
		
		double[] W = new double[nFeature + 1];
		double[] Q = computeQ(X, pr_CSR);
		
		double[] alphas = allocateVector(nExample, 0);
		
		double M = Double.NEGATIVE_INFINITY;
		double m = Double.POSITIVE_INFINITY;
		double Grad = 0;
		// double alpha = 0;
		double alpha_old = 0;
		double alpha_new = 0;
		double PGrad = 0;
		int cnt = 1;
		while (true) {
			M = Double.NEGATIVE_INFINITY;
			m = Double.POSITIVE_INFINITY;
			for (int i = 0; i < nExample; i++) {
				// Grad = Y[i] * innerProduct(W, Xs[i]) - 1;
				Grad = Y[i] * innerProduct(W, X, i, pr_CSR) - 1;
				alpha_old = alphas[i];
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
					// W <- W + (alpha_i_new - alpha_i_old) * Y_i * X_i
					// updateW(W, Y[i] * (alphas[i] - alpha), Xs[i]);
					updateW(W, Y[i] * (alpha_new - alpha_old), X, i, pr_CSR);
					alphas[i] = alpha_new;
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
		
		double[] weights = new double[X.getColumnDimension()];
		System.arraycopy(W, 0, weights, 0, X.getColumnDimension());
		this.W = new DenseMatrix(weights, 1);
		b = new double[1];
		b[0] = W[X.getColumnDimension()];
		
		/*double[] YTimesAlphas = times(Y, alphas);
		this.W = XT.mtimes(new DenseMatrix(YTimesAlphas, 1));
		double[] ones = allocateVector(nSample, 1);
		b = new double[1];
		b[0] = ArrayOperation.innerProduct(ones, YTimesAlphas);*/
		
	}
	
	/**
	 * Compute the inner product <W, [X(i, :) 1]>.
	 * 
	 * @param W
	 * @param X
	 * @param i
	 * @param pr_CSR 
	 * @return <W, [X(i, :) 1]>
	 */
	private double innerProduct(double[] W, Matrix X, int i, double[] pr_CSR) {
		// int M = X.getRowDimension();
		int N = X.getColumnDimension();
		double res = 0;
		double s = 0;
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			XRow = XData[i];
			for (int j = 0; j < N; j++) {
				s += W[j] * XRow[j];
			}
			res = s + W[N];
		} else if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			// int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			// double[] pr = ((SparseMatrix) X).getPr();
			for (int k = jr[i]; k < jr[i + 1]; k++) {
				// s += W[ic[k]] * pr[valCSRIndices[k]];
				s += W[ic[k]] * pr_CSR[k];
			}
			res = s + W[N];
		}
		return res;
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
				Q[i] = s;
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
				Q[i] = s;
			}
		}
		return Q;
	}
	
	/**
	 * Update W in place, i.e., W <- W + v * [X(i, :) 1]'.
	 * @param W
	 * @param v
	 * @param X
	 * @param i
	 * @param pr_CSR 
	 */
	private void updateW(double[] W, double v, Matrix X, int i, double[] pr_CSR) {
		// int M = X.getRowDimension();
		int N = X.getColumnDimension();
		if (X instanceof DenseMatrix) {
			double[][] XData = ((DenseMatrix) X).getData();
			double[] XRow = null;
			XRow = XData[i];
			for (int j = 0; j < N; j++) {
				W[j] += v * XRow[j];
			}
			W[N] += v;
		} else if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			// int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			// double[] pr = ((SparseMatrix) X).getPr();
			for (int k = jr[i]; k < jr[i + 1]; k++) {
				// W[ic[k]] += v * pr[valCSRIndices[k]];
				W[ic[k]] += v * pr_CSR[k];
			}
			W[N] += v;
		}
	}

	@Override
	public Matrix predictLabelScoreMatrix(Matrix Xt) {
		int n = Xt.getRowDimension();
		double[][] ScoreData = ((DenseMatrix) Xt.mtimes(W).plus(b[0])).getData();
		DenseMatrix ScoreMatrix = new DenseMatrix(n, 2);
		double[][] scores = ScoreMatrix.getData();
		for (int i = 0; i < n; i++) {
			scores[i][0] = ScoreData[i][0];
			scores[i][1] = -ScoreData[i][0];
		}
		return ScoreMatrix;
	}

}
