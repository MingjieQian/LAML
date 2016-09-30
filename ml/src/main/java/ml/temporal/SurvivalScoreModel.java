package ml.temporal;

import static la.utils.ArrayOperator.allocate1DArray;
import static la.utils.Matlab.full;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;

/**
 * A abstract class for survival score models.
 * 
 * @author Mingjie Qian
 * @version 1.0 May 13rd, 2015
 */
public abstract class SurvivalScoreModel {
	
	/**
	 * An n x p feature matrix.
	 */
	public Matrix X;
	
	/**
	 * An n x 1 matrix for the observation time.
	 */
	public Matrix T;
	
	/**
	 * An n x 1 matrix for ground truth survival score vector.
	 */
	public Matrix Y;
	
	public Matrix W;
	
	public int n;
	
	public int p;
	
	public void feedData(Matrix X) {
		this.X = X;
		n = X.getRowDimension();
		p = X.getColumnDimension();
	}
	
	public void feedTime(Matrix T) {
		this.T = T;
	}
	
	public void feedScore(Matrix Y) {
		this.Y = Y;
	}
	
	public void feedScore(double[] scores) {
		this.Y = new DenseMatrix(scores, 1);
	}
	
	public void feedScore(int[] labels) {
		double[] scores = new double[labels.length];
		for (int i = 0; i < labels.length; i++) {
			scores[i] = labels[i];
		}
		this.Y = new DenseMatrix(scores, 1);
	}
	
	/**
	 * Initialize the model parameters.
	 * 
	 * @param params
	 */
	public abstract void initialize(double... params);
	
	/**
	 * Train this survival score model.
	 */
	public abstract void train();
	
	public abstract Matrix predict(Matrix Xt, Matrix Tt);
	
	public abstract void loadModel(String filePath);
	
	public abstract void saveModel(String filePath);
	
	public static double[][] computeROC(int[] scoreArray, Matrix Yhat, int numROCPoints) {
		double[] labels = new double[scoreArray.length];
		for (int i = 0; i < scoreArray.length; i++) {
			labels[i] = scoreArray[i];
		}
		Matrix Yt = new DenseMatrix(labels, 1); // n x 1
		return computeROC(Yt, Yhat, numROCPoints);
	}
	
	public static double[][] computeROC(double[] labels, Matrix Yhat, int numROCPoints) {
		Matrix Yt = new DenseMatrix(labels, 1); // n x 1
		return computeROC(Yt, Yhat, numROCPoints);
	}
	
	public static double[][] computeROC(Matrix Yt, Matrix Yhat, int numROCPoints) {
		double[] Y = full(Yt.getColumnVector(0)).getPr();
		int n = Y.length;
		double[] predY = full(Yhat.getColumnVector(0)).getPr();
		double threshold = 0;
		// int numROCPoints = 15;
		double min = min(predY);
		double max = max(predY);
		double d = (max - min) / (numROCPoints - 1);
		double[] FPRs = allocate1DArray(numROCPoints);
		double[] TPRs = allocate1DArray(numROCPoints);
		// Compute FNR and FPR at different threshold
		for (int k = 0; k < numROCPoints; k++) {
			threshold = min + k * d;

			double FP = 0;
			double TP = 0;
			double N = 0;
			double P = 0;
			for (int i = 0; i < n; i++) {
				if (Y[i] == 1) {
					P++;
					if (predY[i] >= threshold) {
						TP++;
					} else {
						
					}
				} else {
					N++;
					if (predY[i] >= threshold) {
						FP++;
					} else {
						
					}
				}
				
			}
			
			double FPR = FP / N;
			double TPR = TP / P;
			FPRs[k] = FPR;
			TPRs[k] = TPR;
		}
		
		return new double[][] {FPRs, TPRs};
	}
	
	public static double min(double[] V) {
		double res = V[0];
		for (int i = 1; i < V.length; i++) {
			if (res > V[i]) {
				res = V[i];
			}
		}
		return res;
	}
	
	public static double max(double[] V) {
		double res = V[0];
		for (int i = 1; i < V.length; i++) {
			if (res < V[i]) {
				res = V[i];
			}
		}
		return res;
	}

}
