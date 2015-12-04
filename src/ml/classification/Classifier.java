package ml.classification;

import java.io.Serializable;
import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import static ml.utils.Matlab.*;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;

public abstract class Classifier implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 2859398998295434078L;
	
	/**
	 * Number of classes.
	 */
	protected int nClass;
	
	/**
	 * Number of features, without bias dummy features,
	 * i.e., for SVM.
	 */
	protected int nFeature;
	
	/**
	 * Number of examples.
	 */
	protected int nExample;
	
	/**
	 * Training data matrix (nExample x nFeature),
	 * each row is a feature vector. The data 
	 * matrix should not include bias dummy features.
	 */
	protected Matrix X;
	
	/**
	 * Label matrix for training (nExample x nClass).
	 * Y_{i,k} = 1 if x_i belongs to class k, and 0 otherwise.
	 */
	protected Matrix Y;
	
	/**
	 * LabelID array for training data, starting from 0.
	 * The label ID array for the training data is latent,
	 * and we don't need to know them. They are only meaningful
	 * for reconstructing the integer labels by using IDLabelMap
	 * structure. 
	 */
	protected int[] labelIDs;
	
	/**
	 * Label array for training data with original integer code.
	 */
	protected int[] labels;
	
	/**
	 * Projection matrix (nFeature x nClass), column i is the projector for class i.
	 */
	protected Matrix W;
	
	/**
	 * The biases in the linear model.
	 */
	protected double[] b;
	
	/**
	 * Convergence tolerance.
	 */
	protected double epsilon;
	
	/**
	 * An ID to integer label mapping array. IDs start from 0.
	 */
	protected int[] IDLabelMap;
	
	/**
	 * Default constructor for a classifier.
	 */
	public Classifier() {
		nClass = 0;
		nFeature = 0;
		nExample = 0;
		X = null;
		W = null;
		epsilon = 1e-4;
	}
	
	/**
	 * Load the model for a classifier.
	 * 
	 * @param filePath file path to load the model
	 * 
	 */
	public abstract void loadModel(String filePath);
	
	/**
	 * Save the model for a classifier.
	 * 
	 * @param filePath file path to save the model
	 * 
	 */
	public abstract void saveModel(String filePath);
	
	/**
	 * Feed training data with original data matrix for this classifier.
	 * 
	 * @param X original data matrix without bias dummy features, each row
	 *          is a feature vector
	 * 
	 */
	public void feedData(Matrix X) {
		this.X = X;
		nFeature = X.getColumnDimension();
		nExample = X.getRowDimension();
	}
	
	/**
	 * Feed training data for this classification method.
	 * 
	 * @param data an n x d 2D {@code double} array with each
	 *             column being a data sample
	 */
	public void feedData(double[][] data) {
		feedData(new DenseMatrix(data));
	}
	
	/**
	 * Infer the number of classes from a given label sequence.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 *               
	 * @return number of classes
	 * 
	 */
	public static int calcNumClass(int[] labels) {
		TreeMap<Integer, Integer> IDLabelMap = new TreeMap<Integer, Integer>();
		int ID = 0;
		int label = -1;
		for (int i = 0; i < labels.length; i++) {
			label = labels[i];
			if (!IDLabelMap.containsValue(label)) {
				IDLabelMap.put(ID++, label);
			}
		}
		int nClass = IDLabelMap.size();
		return nClass;
	}
	
	/**
	 * Get an ID to integer label mapping array. IDs start from 0.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 *               
	 * @return ID to integer label mapping array
	 * 
	 */
	public static int[] getIDLabelMap(int[] labels) {
		TreeMap<Integer, Integer> IDLabelMap = new TreeMap<Integer, Integer>();
		int ID = 0;
		int label = -1;
		for (int i = 0; i < labels.length; i++) {
			label = labels[i];
			if (!IDLabelMap.containsValue(label)) {
				IDLabelMap.put(ID++, label);
			}
		}
		int nClass = IDLabelMap.size();
		int[] IDLabelArray = new int[nClass];
		for (int idx : IDLabelMap.keySet()) {
			IDLabelArray[idx] = IDLabelMap.get(idx);
		}
		return IDLabelArray;
	}
	
	/**
	 * Get a mapping from labels to IDs. IDs start from 0.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 *               
	 * @return a mapping from labels to IDs
	 * 
	 */
	public static TreeMap<Integer, Integer> getLabelIDMap(int[] labels) {
		TreeMap<Integer, Integer> labelIDMap = new TreeMap<Integer, Integer>();
		int ID = 0;
		int label = -1;
		for (int i = 0; i < labels.length; i++) {
			label = labels[i];
			if (!labelIDMap.containsKey(label)) {
				labelIDMap.put(label, ID++);
			}
		}
		return labelIDMap;
	}
	
	/**
	 * Feed labels of training data to the classifier.
	 * 
	 * @param labels any integer array holding the original
	 *               integer labels
	 * 
	 */
	public void feedLabels(int[] labels) {
		nClass = calcNumClass(labels);
		IDLabelMap = getIDLabelMap(labels);
		TreeMap<Integer, Integer> labelIDMap = getLabelIDMap(labels);
		int[] labelIDs = new int[labels.length];
		for (int i = 0; i < labels.length; i++) {
			labelIDs[i] = labelIDMap.get(labels[i]);
		}
		int[] labelIndices = labelIDs;
		Y = labelIndexArray2LabelMatrix(labelIndices, nClass);
		this.labels = labels;
		this.labelIDs = labelIndices;
	}
	
	/**
	 * Feed labels for training data from a matrix.
	 * Note that if we feed the classifier with only
	 * label matrix, then we don't have original integer
	 * labels actually. In this case, label IDs will be
	 * inferred according to the label matrix. The first
	 * observed label index will be assigned ID 0, the second
	 * observed label index will be assigned ID 1, and so on.
	 * And labels will be the label indices in the given
	 * label matrix
	 * 
	 * @param Y an N x K label matrix, where N is the number of
	 *          training examples, and K is the number of classes
	 * 
	 */
	public void feedLabels(Matrix Y) {
		this.Y = Y;
		nClass = Y.getColumnDimension();
		if (nExample != Y.getRowDimension()) {
			System.err.println("Number of labels error!");
			System.exit(1);
		}
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Y);
		labels = labelIndices;
		IDLabelMap = getIDLabelMap(labels);
		labelIDs = labelIndices;
	}
	
	/**
	 * Feed labels for this classification method.
	 * 
	 * @param labels an n x c 2D {@code double} array
	 * 
	 */
	public void feedLabels(double[][] labels) {
		feedLabels(new DenseMatrix(labels));
	}
	
	/**
	 * Train the classifier.
	 */
	public abstract void train();
	
	/**
	 * Predict the labels for the test data formated as an original data matrix.
	 * The original data matrix should not include bias dummy features.
	 * 
	 * @param Xt test data matrix with each row being a feature vector
	 * 
	 * @return predicted label array with original integer label code
	 * 
	 */
	public int[] predict(Matrix Xt) {
		Matrix Yt = predictLabelScoreMatrix(Xt);
		/*
		 * Because column vectors of W are arranged according to the 
		 * order of observation of class labels, in this case, label
		 * indices predicted from the label score matrix are identical
		 * to the latent label IDs, and labels can be inferred by the
		 * IDLabelMap structure.
		 */
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
		int[] labels = new int[labelIndices.length];
		for (int i = 0; i < labelIndices.length; i++) {
			labels[i] = IDLabelMap[labelIndices[i]];
		}
		return labels;
	}
	
	/**
	 * Predict the labels for the test data formated as an original 2D
	 * {@code double} array. The original data matrix should not
	 * include bias dummy features.
	 * 
	 * @param Xt an n x d 2D {@code double} array with each
	 *           row being a data sample
	 *           
	 * @return predicted label array with original integer label code
	 * 
	 */
	public int[] predict(double[][] Xt) {
		return predict(new DenseMatrix(Xt));
	}
	
	/**
	 * Predict the label score matrix given test data formated as an
	 * original data matrix.
	 * 
	 * Note that if a method of an abstract class is declared as
	 * abstract, it is implemented as an interface function in Java.
	 * Thus subclass needs to implement this abstract method rather
	 * than to override it.
	 * 
	 * @param Xt test data matrix with each row being a feature vector
	 * 
	 * @return predicted N x K label score matrix, where N is the number of
	 *         test examples, and K is the number of classes
	 * 
	 */
	public abstract Matrix predictLabelScoreMatrix(Matrix Xt);
	
	/**
	 * Predict the label score matrix given test data formated as an
	 * original data matrix.
	 * 
	 * @param Xt an n x d 2D {@code double} array with each
	 *           row being a data sample
	 *           
	 * @return predicted N x K label score matrix, where N is the number of
	 *         test examples, and K is the number of classes
	 *         
	 */
	public Matrix predictLabelScoreMatrix(double[][] Xt) {
		return predictLabelScoreMatrix(new DenseMatrix(Xt));
	}
	
	/**
	 * Predict the label matrix given test data formated as an
	 * original data matrix.
	 * 
	 * Note that if a method of an abstract class is declared as
	 * abstract, it is implemented as an interface function in Java.
	 * Thus subclasses need to implement this abstract method rather
	 * than to override it.
	 * 
	 * @param Xt test data matrix with each row being a feature vector
	 * 
	 * @return predicted N x K label matrix, where N is the number of
	 *         test examples, and K is the number of classes
	 * 
	 */
	public Matrix predictLabelMatrix(Matrix Xt) {
		Matrix Yt = predictLabelScoreMatrix(Xt);
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Yt);
		return labelIndexArray2LabelMatrix(labelIndices, nClass);
	}
	
	/**
	 * Predict the label matrix given test data formated as an
	 * original 2D {@code double} array.
	 * 
	 * @param Xt an n x d 2D {@code double} array with each
	 *           row being a data sample
	 *           
	 * @return predicted N x K label matrix, where N is the number of
	 *         test examples, and K is the number of classes
	 *         
	 */
	public Matrix predictLabelMatrix(double[][] Xt) {
		return predictLabelMatrix(new DenseMatrix(Xt));
	}

	/**
	 * Get accuracy for a classification task.
	 * 
	 * @param pre_labels predicted labels
	 * 
	 * @param labels true labels
	 * 
	 * @return accuracy
	 * 
	 */
	public static double getAccuracy(int[] pre_labels, int[] labels) {
		if (pre_labels.length != labels.length) {
			System.err.println("Number of predicted labels " +
					"and number of true labels mismatch.");
			System.exit(1);
		}
		int N = labels.length;
		int cnt_correct = 0;
		for ( int i = 0; i < N; i ++ ) {
			if ( pre_labels[i] == labels[i] )
				cnt_correct ++;
		}
		double accuracy = (double)cnt_correct / (double)N;
		System.out.println(String.format("Accuracy: %.2f%%\n", accuracy * 100));
		return accuracy;
	}
	
	/**
	 * Get projection matrix for this classifier.
	 * 
	 * @return a d x c projection matrix
	 * 
	 */
	public Matrix getProjectionMatrix() {
		return W;
	}
	
	/**
	 * Get ground truth label matrix for training data.
	 * 
	 * @return an n x c label matrix
	 * 
	 */
	public Matrix getTrainingLabelMatrix() {
		return Y;
	}
	
	/**
	 * Convert a label matrix to a label index array. Label indices start from 0.
	 * 
	 * @param Y label matrix
	 * 
	 * @return a label index array
	 * 
	 */
	public static int[] labelScoreMatrix2LabelIndexArray(Matrix Y) {
		
		int[] labelIndices = new int[Y.getRowDimension()];

		if (Y instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) Y).getIc();
			int[] jr = ((SparseMatrix) Y).getJr();
			int[] valCSRIndices = ((SparseMatrix) Y).getValCSRIndices();
			double pr[] = ((SparseMatrix) Y).getPr();
			for (int i = 0; i < Y.getRowDimension(); i++) {
				double max = Double.NEGATIVE_INFINITY;
				labelIndices[i] = 0;
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					if (max < pr[valCSRIndices[k]]) {
						max = pr[valCSRIndices[k]];
						labelIndices[i] = ic[k];
					}
				}
			}
		} else {
			double[][] YData = ((DenseMatrix) Y).getData();
			for (int i = 0; i < Y.getRowDimension(); i++) {
				double max = Double.NEGATIVE_INFINITY;
				labelIndices[i] = 0;
				for (int j = 0; j < Y.getColumnDimension(); j++) {
					if (max < YData[i][j]) {
						max = YData[i][j];
						labelIndices[i] = j;
					}
				}
			}
		}

		return labelIndices;

	}
	
	/**
	 * Convert a label index array to a label matrix. Label indices start from 0.
	 * 
	 * @param labelIndices a label index array
	 * 
	 * @param nClass number of classes
	 * 
	 * @return label matrix
	 * 
	 */
	public static Matrix labelIndexArray2LabelMatrix(int[] labelIndices, int nClass) {
		int[] rIndices = new int[labelIndices.length];
		int[] cIndices = new int[labelIndices.length];
		double[] values = new double[labelIndices.length];
		for (int i = 0; i < labelIndices.length; i++) {
			rIndices[i] = i;
			cIndices[i] = labelIndices[i];
			values[i] = 1;
		}
		return new SparseMatrix(rIndices, cIndices, values, labelIndices.length, nClass, labelIndices.length);
	}
	
	/**
	 * Each row of X is a feature vector without the bias term.
	 * 
	 * @param X Original data matrix each row of which is a feature 
	 *          vector without bias
	 * 
	 * @return a vector array where each vector has a bias term 1
	 */
	public static Vector[] getVectors(Matrix X) {
		int l = X.getRowDimension();
		int d = X.getColumnDimension();
		Vector[] res = new Vector[l];
		if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			double[] pr_X = ((SparseMatrix) X).getPr();
			int k = 0;
			int idx = 0;
			int nnz = 0;
			for (int i = 0; i < l; i++) {
				nnz = jr[i + 1] - jr[i] + 1;
				int[] ir = new int[nnz];
				double [] pr = new double[nnz];
				for (k = jr[i], idx = 0; k < jr[i + 1]; k++, idx++) {
					ir[idx] = ic[k];
					pr[idx] = pr_X[valCSRIndices[k]];
				}
				ir[idx] = d;
				pr[idx] = 1;
				res[i] = new SparseVector(ir, pr, nnz, d + 1);
			}
		} else {
			double[][] data = ((DenseMatrix) X).getData();
			double[] dataVector = new double[d + 1];
			for (int i = 0; i < l; i++) {
				System.arraycopy(data[i], 0, dataVector, 0, d);
				dataVector[d] = 1;
				res[i] = new DenseVector(dataVector);
			}
		}
		return res;
	}
	
	/**
	 * Construct a new feature matrix where each row is appended by 
	 * a bias term (default bias is 1).
	 * 
	 * @param X Original data matrix each row of which is a feature 
	 *          vector without bias
	 *           
	 * @return the new feature matrix with bias 1
	 */
	@SuppressWarnings("unused")
	private static Matrix addBias(Matrix X) {
		
		int l = X.getRowDimension();
		int d = X.getColumnDimension();
		Vector[] XVs = new Vector[l];
		Matrix res = null;
		if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			double[] pr_X = ((SparseMatrix) X).getPr();
			int k = 0;
			int idx = 0;
			int nnz = 0;
			for (int i = 0; i < l; i++) {
				nnz = jr[i + 1] - jr[i] + 1;
				int[] ir = new int[nnz];
				double [] pr = new double[nnz];
				for (k = jr[i], idx = 0; k < jr[i + 1]; k++, idx++) {
					ir[idx] = ic[k];
					pr[idx] = pr_X[valCSRIndices[k]];
				}
				ir[idx] = d;
				pr[idx] = 1;
				XVs[i] = new SparseVector(ir, pr, nnz, d + 1);
			}
			res = sparseRowVectors2SparseMatrix(XVs);
		} else {
			double[][] data = ((DenseMatrix) X).getData();
			double[] dataVector = new double[d + 1];
			double[][] resData = new double[l][];
			for (int i = 0; i < l; i++) {
				System.arraycopy(data[i], 0, dataVector, 0, d);
				dataVector[d] = 1;
				// XVs[i] = new DenseVector(dataVector);
				resData[i] = dataVector;
			}
			res = new DenseMatrix(resData);
		}
		return res;
		
	}
	
}
