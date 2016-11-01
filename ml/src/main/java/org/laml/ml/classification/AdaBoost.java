package org.laml.ml.classification;

import static org.laml.la.utils.Matlab.diag;
import static org.laml.la.utils.Matlab.exp;
import static org.laml.la.utils.Matlab.full;
import static org.laml.la.utils.Matlab.minus;
import static org.laml.la.utils.Matlab.mtimes;
import static org.laml.la.utils.Matlab.ne;
import static org.laml.la.utils.Matlab.ones;
import static org.laml.la.utils.Matlab.plus;
import static org.laml.la.utils.Matlab.rdivide;
import static org.laml.la.utils.Matlab.sign;
import static org.laml.la.utils.Matlab.sumAll;
import static org.laml.la.utils.Matlab.times;
import static org.laml.la.utils.Matlab.zeros;
import static org.laml.la.utils.Printer.display;
import static org.laml.la.utils.Printer.fprintf;
import static org.laml.la.utils.Time.tic;
import static org.laml.la.utils.Time.toc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;

import org.laml.la.matrix.DenseMatrix;
import org.laml.la.matrix.Matrix;

/***
 * A Java implementation for AdaBoost.
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 31st, 2014
 */
public class AdaBoost extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1100546985050582205L;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = { {3.5, 4.4, 1.3},
							{5.3, 2.2, 0.5},
							{0.2, 0.3, 4.1},
							{5.3, 2.2, -1.5},
							{-1.2, 0.4, 3.2} };
		int[] labels = {1, 1, -1, -1, -1};
		
		Matrix X = new DenseMatrix(data);
		
		double epsilon = 1e-5;
		Classifier logReg = new LogisticRegression(epsilon);
		logReg.feedData(X);
		logReg.feedLabels(labels);
		logReg.train();
		
		Matrix Xt = X;
		double accuracy = Classifier.getAccuracy(labels, logReg.predict(Xt));
		fprintf("Accuracy for logistic regression: %.2f%%\n", 100 * accuracy);
		
		int T = 10;
		Classifier[] weakClassifiers = new Classifier[T];
		for (int t = 0; t < 10; t++) {
			weakClassifiers[t] = new LogisticRegression(epsilon); 
		}
		Classifier adaBoost = new AdaBoost(weakClassifiers);
		
		adaBoost.feedData(X);
		adaBoost.feedLabels(labels);
		tic();
		adaBoost.train();
		System.out.format("Elapsed time: %.2f seconds.%n", toc());
		
		Xt = X.copy();
		display(adaBoost.predictLabelScoreMatrix(Xt));
		display(full(adaBoost.predictLabelMatrix(Xt)));
		display(adaBoost.predict(Xt));
		accuracy = Classifier.getAccuracy(labels, adaBoost.predict(Xt));
		fprintf("Accuracy for AdaBoost with logistic regression: %.2f%%\n", 100 * accuracy);
		
		// Save the model
		String modelFilePath = "AdaBoostModel";
		adaBoost.saveModel(modelFilePath);
		
		// Load the model
		Classifier adaBoost2 = new AdaBoost();
		adaBoost2.loadModel(modelFilePath);
		
		display(adaBoost2.predictLabelScoreMatrix(Xt));
		display(full(adaBoost2.predictLabelMatrix(Xt)));
		display(adaBoost2.predict(Xt));
		accuracy = Classifier.getAccuracy(labels, adaBoost2.predict(Xt));
		fprintf("Accuracy: %.2f%%\n", 100 * accuracy);

	}

	/**
	 * Number of iterations, or the number of weak classifiers.
	 */
	int T;
	
	/**
	 * The sequence of weak classifiers during training.
	 */
	Classifier[] weakClassifiers;
	
	/**
	 * Weights on the outputs of the trained weak classifiers.
	 */
	double[] alphas;
	
	/**
	 * Constructor.
	 * @param weakClassifiers a sequence of weak classifiers to be
	 *                        trained during the boosting procedure
	 */
	public AdaBoost(Classifier[] weakClassifiers) {
		
		T = weakClassifiers.length;
		this.weakClassifiers = weakClassifiers;
		alphas = new double[T];
		
	}

	/**
	 * Default constructor.
	 */
	public AdaBoost() {
	}

	@Override
	public void loadModel(String filePath) {
		
		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			weakClassifiers = (Classifier[])ois.readObject();
			T = weakClassifiers.length;
			alphas = (double[])ois.readObject();
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
			oos.writeObject(weakClassifiers);
			oos.writeObject(alphas);
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
		
		int d = this.nFeature;
		int m = this.nExample;
		Matrix Dt = times(1.0/m, ones(m, 1));
		Matrix Xt = zeros(d, m);
		// Matrix Yt = zeros(m, 1); // Vector composed of 1 or -1
		// Matrix I = eye(2);
		// Matrix alphas = zeros(1, T);
		Matrix errs = zeros(1, T);
		Matrix ets = zeros(1, T);
		Matrix Coef = diag(diag(new double[]{1, -1}));
		Random generator = new Random();
		Matrix Y_true_labels = mtimes(Y, Coef);
		Matrix Yt_LabelMatrix =  zeros(m, this.nClass);
		
		
		for (int t = 1; t <= T; t++) {
			
			// Sample Xt from X w.r.t Dt
			if (t == 1) {
				Xt = X.copy();
				Yt_LabelMatrix = Y.copy();
				// Yt = Y_true_labels;
			} else {
				if (sumAll(Dt) == 0) {
					Xt = X.copy();
					Yt_LabelMatrix = Y.copy();
				} else {
					for (int i = 1; i <= m; i++) {
						double r_i = generator.nextDouble();
						// Select Xt_i
						double s = 0;
						int j = 1;
						while (j < m) {
							if (s <= r_i && r_i < s + Dt.getEntry(j - 1, 0))
								break;
							else {
								s = s + Dt.getEntry(j - 1, 0);
								j++;
							}
						}
						Xt.setRowMatrix(i - 1, X.getRowMatrix(j - 1));
						Yt_LabelMatrix.setRowMatrix(i - 1, Y.getRowMatrix(j - 1));
						// Yt.setEntry(i - 1, 0, Y_true_labels.getEntry(j - 1, 0));
					}
				}
			}
			
			// Train h_t
			weakClassifiers[t - 1].feedData(Xt);
			weakClassifiers[t - 1].feedLabels(Yt_LabelMatrix);
			weakClassifiers[t - 1].train();
			
			Matrix Y_pred_labels = mtimes(
					weakClassifiers[t - 1].predictLabelMatrix(X), Coef);
			
			Matrix I_err = ne(Y_true_labels, Y_pred_labels);
			double et = 0;
			for (int i = 0; i < m; i++) {
				if (I_err.getEntry(i, 0) == 1)
					et += Dt.getEntry(i, 0);
			}
			// et = sumAll(logicalIndexing(Dt, I_err));
			ets.setEntry(0, t - 1, et);
			errs.setEntry(0, t - 1, sumAll(I_err)/m);
			double alpha_t = 0.5 * Math.log((1 - et) / et);
			// alphas.setEntry(0, t - 1, alpha_t);
			alphas[t - 1] = alpha_t;
			Dt = times(Dt, exp(times(-alpha_t, times(Y_true_labels, Y_pred_labels))));
			double zt = sumAll(Dt);
			if (zt > 0)
				Dt = rdivide(Dt, zt);
			
		}
		
	}
	
	@Override
	public Matrix predictLabelMatrix(Matrix Xt) {
		
		int m = Xt.getRowDimension();
		Matrix Y_score = zeros(m, 1);
		Matrix Coef = diag(diag(new double[]{1, -1}));
		for (int t = 1; t <= T; t++) {
			Y_score = plus(Y_score, times(alphas[t - 1], 
					mtimes(weakClassifiers[t - 1].predictLabelMatrix(Xt), Coef)));
		}
		Matrix H_final_pred = sign(Y_score);
		Matrix Temp = minus(0.5, times(0.5, H_final_pred));
		int[] labelIndices = new int[m];
		for (int i = 0; i < m; i++) {
			labelIndices[i] = (int)Temp.getEntry(i, 0);
		}
		return labelIndexArray2LabelMatrix(labelIndices, nClass);
		
	}

	@Override
	public Matrix predictLabelScoreMatrix(Matrix Xt) {
		
		int m = Xt.getRowDimension();
		Matrix lableScoreMatrix = zeros(m, 2);
		Matrix Y_score = zeros(m, 1);
		Matrix Coef = diag(diag(new double[]{1, -1}));
		for (int t = 1; t <= T; t++) {
			Y_score = plus(Y_score, times(alphas[t - 1], 
					mtimes(weakClassifiers[t - 1].predictLabelMatrix(Xt), Coef)));
		}
		lableScoreMatrix.setColumnMatrix(0, Y_score);
		lableScoreMatrix.setColumnMatrix(1, times(-1, Y_score));
		return lableScoreMatrix;
	}

}
