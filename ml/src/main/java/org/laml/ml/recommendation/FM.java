package org.laml.ml.recommendation;

import static org.laml.la.io.IO.saveVector;
import static org.laml.ml.recommendation.util.Utility.exit;
import static org.laml.ml.recommendation.util.Utility.loadMap;
import static org.laml.ml.recommendation.util.Utility.loadTestUserEventRelation;
import static org.laml.ml.recommendation.util.Utility.predictColdStart;
import static org.laml.ml.recommendation.util.Utility.saveMeasures;
import static org.laml.la.utils.ArrayOperator.allocate1DArray;
import static org.laml.la.utils.ArrayOperator.allocate2DArray;
import static org.laml.la.utils.ArrayOperator.assign;
import static org.laml.la.utils.ArrayOperator.sum;
import static org.laml.la.utils.IO.save;
import static org.laml.la.utils.Matlab.innerProduct;
import static org.laml.la.utils.Printer.errf;
import static org.laml.la.utils.Printer.fprintf;
import static org.laml.la.utils.Printer.printf;
import static org.laml.la.utils.Printer.sprintf;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.TreeMap;

import org.laml.la.io.DataSet;
import org.laml.la.io.InvalidInputDataException;
import org.laml.la.matrix.Matrix;
import org.laml.la.matrix.SparseMatrix;
import org.laml.la.vector.DenseVector;
import org.laml.ml.recommendation.util.Utility;

/**
 * A Java implementation of factorization machines (FM) by ALS.
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 14th, 2015
 */
public class FM {

	public static String Method = "FM";
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String appPath = FM.class.getProtectionDomain().getCodeSource().getLocation().getPath();
		AppDirPath = new File(appPath).getParent();
		
		// Pass the arguments
		double lambda = 0.01;
		int maxIter = 50;
		calcOFV = true;
		int k = 8;
		
		String attribute = "";
		String value = "";
		for (int i = 0; i < args.length; i++) {

			if (args[i].charAt(0) != '-') {
				System.err.println("Wrong options.");
				exit(1);
			}

			if (++i >= args.length)
				exit(1);

			attribute = args[i - 1];
			value = args[i];
			
			if (attribute.equals("-MaxIter")) {
				maxIter = Integer.parseInt(value);
			} else if (attribute.equals("-lambda")) {
				lambda = Double.parseDouble(value);
			}

		}
		
		System.out.println("Running FM...");
		fprintf("lambda = %f\n", lambda);
		String trainFilePath = AppDirPath + File.separator + "Train.libfm.txt";
		String testFilePath = AppDirPath + File.separator + "Test.libfm.txt";
		String outputFilePath = AppDirPath + File.separator + "FM-YijPredOnTest.txt";
		
		// Load training data
		int idxStart = 0;
		feedTrainingData(trainFilePath, idxStart);
		
		// Initialization
		allocateResource(k);
		feedParams(maxIter, lambda);
		initialize();
		
		// Train FM model parameters by training data
		train();
		
		// Prediction: generate and save FM-YijPredOnTest.txt
		DataSet testData = loadData(testFilePath, idxStart);
		// predict(testData.X, outputFilePath);
		double[] Yij_pred = predict(testData.X);
		save(Yij_pred, outputFilePath);
		
		// Predict
		// double[] Yij_pred = loadDenseVector(outputFilePath).getPr();
		loadMap(TestIdx2TrainIdxUserMap, AppDirPath + File.separator + "TestIdx2TrainIdxUserMap.txt");
		loadMap(TestIdx2TrainIdxItemMap, AppDirPath + File.separator + "TestIdx2TrainIdxItemMap.txt");
		TestUserIndices = loadTestUserEventRelation(
				AppDirPath + File.separator + "Test-Events.txt",
				TestUser2EventIndexSetMap
				);
		
		double[] measures = null;
		measures = Utility.predict(
				testData.Y, 
				Yij_pred, 
				TestUser2EventIndexSetMap
				);
		saveMeasures(AppDirPath, sprintf("%s-Measures", Method), measures);
		
		measures = predictColdStart(
				testData.Y, 
				Yij_pred, 
				TestUserIndices,
				TestUser2EventIndexSetMap,
				TestIdx2TrainIdxUserMap,
				TestIdx2TrainIdxItemMap
				);
		saveMeasures(AppDirPath, sprintf("%s-ColdStart-Measures", Method), measures);
		
		System.out.println("\nMission complete.");
		
	}
	
	/**
	 * Just need call this method once to allocate required resources.
	 * Feature sizes will be set. This method should be called after 
	 * training data is loaded and K is set.
	 * 
	 * @param K
	 */
	public static void allocateResource(int K) {
		W = new DenseVector(p, 0.0000);
		V = new DenseVector[K];
		for (int k = 0; k < K; k++) {
			V[k] = new DenseVector(p, 0);
		}
		e = allocate1DArray(n, 0);
		// q = allocate1DArray(n, 0);
		Q = allocate2DArray(K, n);
		y_hat = allocate1DArray(n, 0);
	}
	
	public static void initialize() {
		b = 0;
		W.clear();
		Random generator = new Random();
		double sigma = 0.0001;
		for (int k = 0; k < K; k++) {
			double[] pr = V[k].getPr();
			for (int j = 1; j < p; j++) {
				pr[j] = generator.nextGaussian() * sigma;
				// pr[j] = 0;
			}
		}
		assign(y_hat, 0);
		assign(e, 0);
		// assign(q, 0);
	}
	
	public static void feedParams (
			int MaxIter,
			double lambda
			) {
		FM.MaxIter = MaxIter;
		FM.lambda = lambda * n;
	}
	
	@SuppressWarnings("unused")
	private static void predict(String outputFilePath) {
		int[] ic = ((SparseMatrix) X).getIc();
		/*int[] ir = ((SparseMatrix) X).getIr();
		int[] jc = ((SparseMatrix) X).getJr();*/
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		// compute y_hat
		double[] w = W.getPr();
		for (int r = 0; r < n; r++) {
			double s = b;
			/*
			 * Compute <W, X[r, :]>
			 */
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			double A = 0;
			double B = 0;
			for (int f = 0; f < K; f++) {
				double[] v = V[f].getPr();
				double a = 0;
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					int j = ic[k];
					double vj = v[j];
					double xj = pr[valCSRIndices[k]];
					a += vj * xj;
					B += vj * vj * xj * xj;
				}
				A += a * a;
			}
			s += (A - B) / 2;
			y_hat[r] = s;
		}
		saveVector(outputFilePath, new DenseVector(y_hat));
	}
	
	public static double[] predict(Matrix X, String outputFilePath) {
		double[] y_pred = predict(X);
		saveVector(outputFilePath, new DenseVector(y_pred));
		return y_pred;
	}
	
	public static double[] predict(Matrix X) {
		int n = X.getRowDimension();
		double[] y_hat = allocate1DArray(n);
		int[] ic = ((SparseMatrix) X).getIc();
		/*int[] ir = ((SparseMatrix) X).getIr();
		int[] jc = ((SparseMatrix) X).getJr();*/
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		// compute y_hat
		double[] w = W.getPr();
		for (int r = 0; r < n; r++) {
			double s = b;
			/*
			 * Compute <W, X[r, :]>
			 */
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			double A = 0;
			double B = 0;
			for (int f = 0; f < K; f++) {
				double[] v = V[f].getPr();
				double a = 0;
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					int j = ic[k];
					double vj = v[j];
					double xj = pr[valCSRIndices[k]];
					a += vj * xj;
					B += vj * vj * xj * xj;
				}
				A += a * a;
			}
			s += (A - B) / 2;
			y_hat[r] = s;
		}
		return y_hat;
	}
	
	public static void feedTrainingData(DataSet dataSet) {
		X = dataSet.X;
		y = dataSet.Y;
		n = X.getRowDimension();
		p = X.getColumnDimension();
	}
	
	/**
	 * Read a data set from a LIBSVM formatted file. Note that an empty 
	 * string will be viewed as an empty sparse feature vector. Empty 
	 * sparse vectors commonly occur when a test example only has new 
	 * features. Note that the default starting index is 0.
	 * 
	 * @param trainFilePath file path for the training data
	 */
	public static void feedTrainingData(String trainFilePath) {
		feedTrainingData(trainFilePath, 0);
	}
	
	/**
	 * Read a data set from a LIBSVM formatted file. Note that an empty 
	 * string will be viewed as an empty sparse feature vector. Empty 
	 * sparse vectors commonly occur when a test example only has new 
	 * features.
	 * 
	 * @param trainFilePath file path for the training data
	 * 
	 * @param idxStart starting index
	 */
	public static void feedTrainingData(String trainFilePath, int idxStart) {
		feedTrainingData(loadData(trainFilePath, idxStart));
	}
	
	public static DataSet loadData(String filePath, int idxStart) {
		DataSet.IdxStart = idxStart;
		DataSet dataSet = null;
		try {
			dataSet = DataSet.readDataSetFromFile(filePath);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		return dataSet;
	}
	
	/**
	 * Train FM.
	 */
	static void train() {
		
		System.out.printf("Training %s...\n", Method);
		double[] OFVs = null;
		boolean debug = !true;
		if (calcOFV) {
			OFVs = allocate1DArray(MaxIter + 1, 0);
			double ofv = 0;
			for (int i = 0; i < n; i++) {
				ofv += y[i] * y[i];
			}
			OFVs[0] = ofv;
			fprintf("Iter %d: %.10g\n", 0, ofv);
		}
		int cnt = 0;
		double[] w = W.getPr();
		double ofv_old = 0;
		double ofv_new = 0;
		int[] ic = ((SparseMatrix) X).getIc();
		int[] ir = ((SparseMatrix) X).getIr();
		int[] jc = ((SparseMatrix) X).getJc();
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		
		// Compute y_hat and cache e
		for (int r = 0; r < n; r++) {
			double s = b;
			/*
			 * Compute <W, X[r, :]>
			 */
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			double A = 0;
			double B = 0;
			for (int f = 0; f < K; f++) {
				double[] v = V[f].getPr();
				double a = 0;
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					int j = ic[k];
					double vj = v[j];
					double xj = pr[valCSRIndices[k]];
					a += vj * xj;
					B += vj * vj * xj * xj;
				}
				A += a * a;
			}
			s += (A - B) / 2;
			y_hat[r] = s;
			e[r] = y[r] - s; // Why recalculated e[r] != e[r] updated till the end in last iteration
		}
		
		for (int f = 0; f < K; f++) {
			double[] v = V[f].getPr();
			// Cache q for factor f
			q = Q[f];
			for (int r = 0; r < n; r++) {
				/*
				 * Compute q[r] = <v, X[r, :]>
				 */
				int s = 0;
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					int j = ic[k];
					s += v[j] * pr[valCSRIndices[k]];
				}
				q[r] = s;
			}
		}

		while (true) {
			
			// Update model parameters in one full iteration
			
			// We should have recalculated e[i] here
			
			/*for (int i = 0; i < 10; i++) {
				print(e[i] + "\t");
			}
			println();*/
			
			// Update b
			ofv_old = 0;
			if (debug) {
				ofv_old = computOFV();
				printf("f(b): %f\n", ofv_old);
			}
			double b_new = (b * n + sum(e)) / (n + lambda);
			for (int i = 0; i < n; i++)
				e[i] -= (b_new - b);
			b = b_new;
			
			if (debug) {
				ofv_new = computOFV();
				printf("b updated: %f\n", ofv_new);
				if (ofv_old < ofv_new) {
					errf("Error when updating b\n");
				}
			}
			
			// println("b = " + b);
			
			// Update W
			for (int j = 0; j < p; j++) {
				ofv_old = 0;
				if (debug) {
					ofv_old = computOFV();
					printf("f(w[%d]): %f\n", j, ofv_old);
				}
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
				// e[i] for X[i,j] != 0
				for (int k = jc[j]; k < jc[j + 1]; k++) {
					int i = ir[k];
					double xj = pr[k];
					// double t = X.getEntry(i, j);
					e[i] -= (wj_new - w[j]) * xj;
				}
				w[j] = wj_new;
				
				if (debug) {
					ofv_new = computOFV();
					printf("w[%d] updated: %f\n", j, ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating w[%d]\n", j);
					}
				}
			}
			
			// Update V by columns
			for (int f = 0; f < K; f++) {
				double[] v = V[f].getPr();
				// We should have recalculated Q[f] here
				// Cache q for factor f
				/*for (int r = 0; r < n; r++) {
					
					// Compute q[r] = <v, X[r, :]>
					 
					int s = 0;
					for (int k = jr[r]; k < jr[r + 1]; k++) {
						int j = ic[k];
						s += v[j] * pr[valCSRIndices[k]];
					}
					q[r] = s;
				}*/
				q = Q[f];
				for (int j = 0; j < p; j++) {
					ofv_old = 0;
					if (debug) {
						ofv_old = computOFV();
						printf("f(V[%d, %d]): %f\n", j, f, ofv_old);
					}
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
						double hj = xj * (q[i] - v[j] * xj);
						v1 += hj * hj;
						v2 += hj * e[i];
					}
					// Update v[j]
					double vj_new = (v[j] * v1 + v2) / (v1 + lambda);
					// Update e[i] for X[i,j] != 0
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						int i = ir[k];
						double xj = pr[k];
						double hj = xj * (q[i] - v[j] * xj);
						e[i] -= (vj_new - v[j]) * hj;
						/*
						 * We traverse i from 0 to n - 1 for X[i,j] != 0
						 * thus updating q[i] will not affect e[i'] for
						 * i' > i.
						 */
						// q[i] += (vj_new - v[j]) * xj;
					}
					// Update q[i] for X[i,j] != 0
					for (int k = jc[j]; k < jc[j + 1]; k++) {
						int i = ir[k];
						double xj = pr[k];
						q[i] += (vj_new - v[j]) * xj;
					}
					v[j] = vj_new;
					
					if (debug) {
						ofv_new = computOFV();
						printf("V[%d, %d] updated: %f\n", j, f, ofv_new);
						if (ofv_old < ofv_new) {
							errf("Error when updating V[%d,%d]\n", j, f);
						}
					}
				}
			}
			
			cnt++;
			if (calcOFV) {
				double ofv = 0;
				/*for (int i = 0; i < n; i++) {
					ofv += e[i] * e[i];
				}
				ofv += lambda * b * b;
				ofv += lambda * innerProduct(W, W);
				for (int f = 0; f < K; f++) {
					ofv += lambda * innerProduct(V[f], V[f]);
				}*/
				ofv = computOFV();
				OFVs[cnt] = ofv;
				if (cnt % 10 == 0)
					fprintf(".Iter %d: %.8g\n", cnt, ofv);
				else
					fprintf(".");
			} else {
				if (cnt % 10 == 0)
					fprintf(".Iter %d\n", cnt);
				else
					fprintf(".");
			}
			if (cnt >= MaxIter) {
				break;
			}
		}
	}
	
	private static double computOFV() {
		/*double ofv = 0;
		for (int i = 0; i < n; i++) {
			ofv += e[i] * e[i];
		}
		ofv += lambda * b * b;
		ofv += lambda * innerProduct(W, W);
		for (int f = 0; f < K; f++) {
			ofv += lambda * innerProduct(V[f], V[f]);
		}
		return ofv;*/
		
		double ofv = 0;
		ofv += lambda * b * b;
		ofv += lambda * innerProduct(W, W);
		for (int f = 0; f < K; f++) {
			ofv += lambda * innerProduct(V[f], V[f]);
		}
		int[] ic = ((SparseMatrix) X).getIc();
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		// compute y_hat
		double[] w = W.getPr();
		for (int r = 0; r < n; r++) {
			double s = b;
			// Compute <W, X[r, :]>
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			double A = 0;
			double B = 0;
			for (int f = 0; f < K; f++) {
				double[] v = V[f].getPr();
				double a = 0;
				for (int k = jr[r]; k < jr[r + 1]; k++) {
					int j = ic[k];
					double vj = v[j];
					double xj = pr[valCSRIndices[k]];
					a += vj * xj;
					B += vj * vj * xj * xj;
				}
				A += a * a;
			}
			s += (A - B) / 2;
			double e = (y[r] - s);
			ofv += e * e;
		}
		return ofv;
	}
	
	/**
	 * File path for the directory where SLFM.jar is located, which will be 
	 * inferred in runtime.
	 */
	private static String AppDirPath = "";
	
	private static int MaxIter = 20;
	
	private static boolean calcOFV = false;
	
	/**
	 * Feature size.
	 */
	private static int p;
	
	/**
	 * Dimensionality of the latent space.
	 */
	private static int K = 8;
	
	/**
	 * Data size.
	 */
	private static int n;
	
	/**
	 * Bias.
	 */
	private static double b;
	
	/**
	 * First order weights.
	 * W[j] is the linear weight for feature j.
	 */
	private static DenseVector W;
	
	/**
	 * Second order latent vectors.
	 * V[f] is the latent vector for the f-th latent dimension.
	 */
	private static DenseVector[] V;
	
	/**
	 * Labels.
	 */
	private static int[] y;
	
	/**
	 * y_hat[i] is the predicted value for instance i.
	 */
	private static double[] y_hat;
	
	/**
	 * e[i] = y[i] - y_hat[i].
	 */
	private static double[] e;
	
	/**
	 * q[i] = \sum_j V[j][f] * X[i][j]
	 */
	private static double[] q;
	
	/**
	 * Q[f][i] = \sum_j V[j][f] * X[i][j]
	 */
	private static double[][] Q;
	
	/**
	 * Array of feature vectors for all instances.
	 *//*
	private static Vector[] X = null;*/
	
	/**
	 * n x d data matrix.
	 */
	private static Matrix X;
	
	/**
	 * Let the data matrix X be an n x p matrix, then 
	 * XColumns[j] = X[:,j].
	 *//*
	private static Vector[] XColumns = null;*/
	
	private static double lambda;
	
	/**
	 * Test index: Training index or -1 if is new
	 */
	static TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap = new TreeMap<Integer, Integer>();
	
	/**
	 * Test index: Training index or -1 if is new
	 */
	static TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap = new TreeMap<Integer, Integer>();
	
	/**
	 * TestUser2EventIndexSetMap[i] = {indexOf(i, j) | (i, j) \in C}.
	 */
	public static HashMap<Integer, LinkedList<Integer>> TestUser2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
	
	/**
	 * Item2EventIndexSetMap[j] = {indexOf(i, j) | (i, j) \in C}.
	 */
	// public static HashMap<Integer, LinkedList<Integer>> TestItem2EventIndexSetMap = new HashMap<Integer, LinkedList<Integer>>();
	
	/**
	 * TestUserIndices[k] is the user index for the k-th event.
	 */
	public static int[] TestUserIndices = null;

}
