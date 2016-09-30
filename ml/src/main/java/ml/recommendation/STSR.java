package ml.recommendation;

import static la.io.IO.loadDenseVector;
import static la.io.IO.saveVector;
import static ml.recommendation.util.Utility.exit;
import static ml.recommendation.util.Utility.getFeatureSize;
import static ml.recommendation.util.Utility.loadMap;
import static ml.recommendation.util.Utility.loadTestUserEventRelation;
import static ml.recommendation.util.Utility.predictColdStart;
import static ml.recommendation.util.Utility.saveMeasures;
import static la.utils.ArrayOperator.allocate1DArray;
import static la.utils.ArrayOperator.allocate2DArray;
import static la.utils.ArrayOperator.assign;
import static la.utils.ArrayOperator.clear;
import static la.utils.ArrayOperator.innerProduct;
import static la.utils.ArrayOperator.sum;
import static la.utils.IO.save;
import static la.utils.Printer.errf;
import static la.utils.Printer.fprintf;
import static la.utils.Printer.printf;
import static la.utils.Printer.sprintf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.TreeMap;

import la.io.DataSet;
import la.io.IO;
import la.io.InvalidInputDataException;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import ml.recommendation.util.Utility;

/**
 * A Java implementation of structured sparse (polynomial) 
 * regression (STSR) by ALS.
 * 
 * @author Mingjie Qian
 * @version 1.0 Mar. 20th, 2015
 */
public class STSR {

	public static String Method = "STSR";
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String appPath = STSR.class.getProtectionDomain().getCodeSource().getLocation().getPath();
		AppDirPath = new File(appPath).getParent();
		String OSName = System.getProperty("os.name");
		if (OSName.toLowerCase().contains("windows")) {
			AppDirPath = AppDirPath.replace("%20", " ");
		}
		
		// Pass the arguments
		double lambda = 0.01;
		double nu = 0.0000001;
		int maxIter = 30;
		calcOFV = false;
		
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
			
			if (attribute.equals("-maxIter")) {
				maxIter = Integer.parseInt(value);
			} else if (attribute.equals("-lambda")) {
				lambda = Double.parseDouble(value);
			} else if (attribute.equals("-nu")) {
				nu = Double.parseDouble(value);
			}

		}
		
		fprintf("Running %s...\n", Method);
		fprintf("lambda = %f, nu = %f\n", lambda, nu);
		String trainFilePath = AppDirPath + File.separator + "Train.txt";
		String testFilePath = AppDirPath + File.separator + "Test.txt";
		String outputFilePath = AppDirPath + File.separator + "STSR-YijPredOnTest.txt";
		
		// Load training data
		int idxStart = 0;
		feedTrainingData(trainFilePath, idxStart);
	
		// Build tree structured pair groups
		/*
		 * featureSize format:
		 * User[\t]383
		 * Item[\t]1175
		 * Event[\t]1
		 */
		String featureSizeFileName = "FeatureSize.txt";
		String featureSizeFilePath = AppDirPath + File.separator + featureSizeFileName;
		/*
		 * Each line is a group feature index pair (idx1, idx2) separated by 
		 * a tab character, e.g.
		 * (157, 158)
		 * (157, 236)
		 * (24, 157)[\t](157, 158)[\t](157, 236)
		 */
		String userFeatureGroupListFilePath = AppDirPath + File.separator + "UserTreeStructuredPairGroupList.txt";
		String itemFeatureGroupListFilePath = AppDirPath + File.separator + "ItemTreeStructuredPairGroupList.txt";
		buildTreeStructuredPairGroupList(
				featureSizeFilePath,
				userFeatureGroupListFilePath,
				itemFeatureGroupListFilePath
				);
		
		// Initialization
		allocateResource();
		
		feedParams(maxIter, lambda, nu);
		initialize();
		// initializeWij();
		// Train STSR model parameters by training data
		train();
		
		// Prediction: generate and save STSR-YijPredOnTest.txt
		DataSet testData = loadData(testFilePath, idxStart);
		// predict(testData.X, outputFilePath);
		double[] Yij_pred = predict(testData.X);
		save(Yij_pred, outputFilePath);
		
		// Predict
		Yij_pred = loadDenseVector(outputFilePath).getPr();
		double[] measures = null;
		loadMap(TestIdx2TrainIdxUserMap, AppDirPath + File.separator + "TestIdx2TrainIdxUserMap.txt");
		loadMap(TestIdx2TrainIdxItemMap, AppDirPath + File.separator + "TestIdx2TrainIdxItemMap.txt");
		TestUserIndices = loadTestUserEventRelation(
				AppDirPath + File.separator + "Test-Events.txt",
				TestUser2EventIndexSetMap
				);
		
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
		
		IO.saveVector(new DenseVector(new double[] {b}), "STR-b.txt");
		IO.saveVector(new DenseVector(w), "STR-w.txt");
		// IO.saveMatrix(Matlab.sparse(new DenseMatrix(W)), "STR-W2.txt");
		
		System.out.println("\nMission complete.");
		
	}
	
	/**
	 * Just need call this method once to allocate required resources.
	 * Feature sizes will be set. This method should be called after 
	 * training data is loaded.
	 */
	public static void allocateResource() {
		w = allocate1DArray(p, 0);
		/*V = new DenseVector[K];
		for (int k = 0; k < K; k++) {
			V[k] = new DenseVector(p, 0);
		}*/
		e = allocate1DArray(n, 0);
		// q = allocate1DArray(n, 0);
		// Q = allocate2DArray(K, n);
		y_hat = allocate1DArray(n, 0);
		L = allocate2DArray(p, p, 0);
		computeL();
		W = allocate2DArray(p, p, 0);
	}
	
	public static void initialize() {
		b = 0;
		double sigma = 0.0001;
		clear(w);
		Random generator = new Random();
		/*for (int i = 0; i < p; i++) {
			w[i] = generator.nextGaussian() * sigma;
		}*/
		for (int i = 0; i < p; i++) {
			for (int j = i + 1; j < p; j++) {
				W[i][j] = generator.nextGaussian() * sigma;
			}
		}
		assign(y_hat, 0);
		assign(e, 0);
		// assign(q, 0);
		/*lambda *= n;
		nu *= n;*/
		// computeL();
	}
	
	public static void feedParams (
			int maxIter,
			double lambda,
			double nu
			) {
		STSR.maxIter = maxIter;
		STSR.lambda = lambda * n;
		STSR.nu = nu * n;
		STSR.maxCnt = 2;
	}
	
	public static double[] predict(Vector[] X, String outputFilePath) {
		double[] y_pred = predict(X);
		saveVector(outputFilePath, new DenseVector(y_pred));
		return y_pred;
	}
	
	public static double[] predict(Vector[] X) {
		int n = X.length;
		double[] y_hat = allocate1DArray(n);
		// compute y_hat
		// compute y_hat
		for (int r = 0; r < n; r++) {
			SparseVector x = (SparseVector) X[r];
			int[] ir = x.getIr();
			double[] pr = x.getPr();
			double s = b;
			/*
			 * Compute <w, X[r]>
			 */
			for (int k = 0; k < x.getNNZ(); k++) {
				int j = ir[k];
				s += w[j] * pr[k];
			}
			/*
			 * Compute \sum_{ij} x[i] * x[j] * W[i][j]
			 */
			int i, j;
			double xi, xj;
			double hij;
			for (int k1 = 0; k1 < x.getNNZ(); k1++) {
				i = ir[k1];
				xi = pr[k1];
				for (int k2 = k1 + 1; k2 < x.getNNZ(); k2++) {
					j = ir[k2];
					xj = pr[k2];
					hij = xi * xj;
					s += hij * W[i][j];
				}
			}
			y_hat[r] = s;
		}
		return y_hat;
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
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		// compute y_hat
		// double[] w = W.getPr();
		for (int r = 0; r < n; r++) {
			double s = b;
			/*
			 * Compute <W, X[r, :]>
			 */
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			for (int k1 = jr[r]; k1 < jr[r + 1]; k1++) {
				int i = ic[k1];
				double xi = pr[valCSRIndices[k1]];
				for (int k2 = k1 + 1; k2 < jr[r + 1]; k2++) {
					int j = ic[k2];
					double xj = pr[valCSRIndices[k2]];
					s += W[i][j] * xi * xj;
				}
			}
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
	
	/*static void feedTrainingData(DataVectors dataSet) {
		Xs = dataSet.Vs;
		y = dataSet.Y;
		n = Xs.length;
		p = Xs[0].getDim();
	}
	
	static void feedTrainingData(String trainFilePath) {
		feedTrainingData(loadData(trainFilePath));
	}
	
	public static DataVectors loadData(String filePath) {
		DataSet.IdxStart = 0;
		DataSet dataSet = null;
		try {
			dataSet = DataSet.readDataSetFromFile(filePath);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		X = dataSet.X;
		
		DataVectors.IdxStart = 0;
		DataVectors dataVectors = null;
		try {
			dataVectors = la.io.DataVectors.readDataSetFromFile(filePath);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		return dataVectors;
	}*/
	
	static void computeL() {
		int[] ic = ((SparseMatrix) X).getIc();
		int[] jr = ((SparseMatrix) X).getJr();
		double[] pr = ((SparseMatrix) X).getPr();
		int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
		for (int r = 0; r < n; r++) {
			for (int k1 = jr[r]; k1 < jr[r + 1]; k1++) {
				int i = ic[k1];
				double xi = pr[valCSRIndices[k1]];
				for (int k2 = k1 + 1; k2 < jr[r + 1]; k2++) {
					int j = ic[k2];
					double xj = pr[valCSRIndices[k2]];
					int index = i * p + j;
					double hij = xi * xj;
					L[i][j] += 2 * hij * hij;
					if (map.containsKey(index)) {
						map.get(index).add(new float[] {r, (float) hij});
					} else {
						ArrayList<float[]> list = new ArrayList<float[]>();
						list.add(new float[] {r, (float) hij});
						map.put(index, list);		
					}
				}
			}
		}
	}
	
	/**
	 * Train FM.
	 */
	public static void train() {
		
		System.out.printf("Training %s...\n", Method);
		double[] OFVs = null;
		boolean debug = !true;
		int blockSize = 10;
		if (calcOFV) {
			OFVs = allocate1DArray(maxIter + 1, 0);
			double ofv = computeOFV();
			OFVs[0] = ofv;
			fprintf("Iter %d: %.10g\n", 0, ofv);
		}
		
		int cnt = 0;
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
			// Compute <W, X[r, :]>
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			/*
			 * Compute \sum_{ij} x[i] * x[j] * W[i][j]
			 */
			for (int k1 = jr[r]; k1 < jr[r + 1]; k1++) {
				int i = ic[k1];
				double xi = pr[valCSRIndices[k1]];
				for (int k2 = k1 + 1; k2 < jr[r + 1]; k2++) {
					int j = ic[k2];
					double xj = pr[valCSRIndices[k2]];
					s += W[i][j] * xi * xj;
				}
			}
			y_hat[r] = s;
			e[r] = y[r] - s; // Why recalculated e[r] != e[r] updated till the end in last iteration
		}
		
		HashMap<Integer, Double> instIdx2SumMap = new HashMap<Integer, Double>();
		ArrayList<Double> WArr = new ArrayList<Double>(10);
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
				ofv_old = computeOFV();
				printf("f(b): %f\n", ofv_old);
			}
			double b_new = (b * n + sum(e)) / (n + lambda);
			for (int i = 0; i < n; i++)
				e[i] -= (b_new - b);
			b = b_new;
			// println("b = " + b);
			
			if (debug) {
				ofv_new = computeOFV();
				printf("b updated: %f\n", ofv_new);
				if (ofv_old < ofv_new) {
					errf("Error when updating b\n");
				}
			}
			
			// Update w
			for (int j = 0; j < p; j++) {
				
				ofv_old = 0;
				/*if (debug) {
					ofv_old = computeOFV();
					printf("f(w[%d]): %f\n", j, ofv_old);
				}*/
				// v1 = \sum_i h^2(x_i)
				double v1 = 0;
				// v2 = \sum_i h(x_i) * e[i]
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
				
				/*if (debug) {
					ofv_new = computeOFV();
					printf("w[%d] updated: %f\n", j, ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating w[%d]\n", j);
					}
				}*/
			
			}
			
			/*if (cnt == 3) {
				int a = 1;
				a = a + 1;
			}*/

			// Update W
			if (debug) {
				ofv_old = computeOFV();
				printf("f(W): %f\n", ofv_old);
			}
			// for (int[][] group : TreeStructuredPairGroupList) {
			for (int gIdx = 0; gIdx < TreeStructuredPairGroupList.size(); gIdx++) {
				int[][] group = TreeStructuredPairGroupList.get(gIdx);
				int groupSize = group.length;
				if (groupSize == 1) {
					int[] pair = group[0];
					int i = pair[0];
					int j = pair[1];
					int index = i * p + j;
					if (!map.containsKey(index)) {
						continue;
					}
					if (W[i][j] == 0) {
						continue;
					}
					ArrayList<float[]> list = map.get(index);
					double lipschitz = 1 * L[i][j] / (nu / n);
					// double lipschitz = 1 * L[i][j];
					double c = nu / lipschitz;
					// int maxCnt = 2;
					for (int k = 0; k < maxCnt; k++) {
						double Wij = W[i][j];
						double grad = 0;
						for (float[] element : list) {
							int r = (int) element[0];
							double hij = element[1];
							grad += -2 * hij * e[r];
							/*if (Double.isInfinite(e[r])) {
								int a = 1;
								a = a + 1;
							}*/
						}
						// Wij^{t+1} = prox_{c||.||_1} (Wij^{t} - grad / L)
						Wij = Wij - grad / lipschitz;
						if (Math.abs(Wij) <= c) {
							Wij = 0;
						} else if (Wij > c) {
							Wij = Wij - c;
						} else {
							Wij = Wij + c;
						}
						/*if (W[i][j] == 0 && Wij != 0) {
							int a = 1;
							a = a + 1;
						}*/
						/*if (Double.isNaN(Wij)) {
							int a = 1;
							a = a + 1;
						}*/
						// Update e[r] for r \in {r: X[r][i] * X[r][j] != 0}
						double diff = Wij - W[i][j];
						if (diff == 0) {
							continue;
						}
						for (float[] element : list) {
							int r = (int) element[0];
							double hij = element[1];
							// e[r] -= (Wij - W[i][j]) * hij;
							// double e_r_old = e[r];
							e[r] -= diff * hij;
							/*if (Double.isInfinite(e[r]) && !Double.isInfinite(e_r_old)) {
								int a = 1;
								a = a + 1;
							}*/
						}
						W[i][j] = Wij;
					}
					continue;
				}
				/*if (debug) {
					ofv_old = computeOFV();
					printf("f(W[group(%d)]): %f\n", gIdx, ofv_old);
				}*/
				instIdx2SumMap.clear();
				for (int[] pair : group) {
					int i = pair[0];
					int j = pair[1];
					int index = i * p + j;
					if (!map.containsKey(index)) {
						continue;
					}
					ArrayList<float[]> list = map.get(index);
					for (float[] element : list) {
						int r = (int) element[0];
						double hij = element[1];
						if (instIdx2SumMap.containsKey(r)) {
							instIdx2SumMap.put(r, instIdx2SumMap.get(r) + hij);
						} else {
							instIdx2SumMap.put(r, hij);
						}
					}
				}
				// Compute \sum_r h1(Xr) (h1(Xr) + h2(Xr) + h3(Xr))
				double lipschitz = 0;
				for (int[] pair : group) {
					int i = pair[0];
					int j = pair[1];
					int index = i * p + j;
					if (!map.containsKey(index)) {
						continue;
					}
					ArrayList<float[]> list = map.get(index);
					double sum = 0;
					for (float[] element : list) {
						int r = (int) element[0];
						double hij = element[1];
						sum += hij * instIdx2SumMap.get(r);
					}
					if (lipschitz < sum) {
						lipschitz = sum;
					}
				}
				lipschitz /= (nu / n);
				double c = nu / lipschitz;
				
				WArr.clear();
				/*for (int k = 0; k < groupSize; k++) {
					int[] pair = group[k];*/
				for (int[] pair : group) {
					int i = pair[0];
					int j = pair[1];
					int index = i * p + j;
					if (!map.containsKey(index)) {
						WArr.add(0.0);
						continue;
					}
					ArrayList<float[]> list = map.get(index);
					double grad = 0;
					for (float[] element : list) {
						int r = (int) element[0];
						double hij = element[1];
						grad += -2 * hij * e[r];
					}
					WArr.add(W[i][j] - grad / lipschitz);
				}
				
				double norm = 0;
				if (groupSize == 1) {
					// norm = W[group[0][0]][group[0][1]];
					norm = WArr.get(0);
				} else {
					/*for (int[] pair : group) {
						double Wij = W[pair[0]][pair[1]];
						norm += Wij * Wij;
					}*/
					for (double Wij : WArr) {
						norm += Wij * Wij;
					}
					norm = Math.sqrt(norm);
				}
				if (norm <= c) {
					/*for (int[] pair : group) {
						W[pair[0]][pair[1]] = 0;
					}*/
					for (int k = 0; k < groupSize; k++) {
						WArr.set(k, 0.0);
					}
				} else {
					double s = 1 - c / norm;
					/*for (int[] pair : group)
						W[pair[0]][pair[1]] *= s;*/
					for (int k = 0; k < groupSize; k++) {
						WArr.set(k, WArr.get(k) * s);
					}
				}
				// Now we have the new W value for this group
				// Update e[r] for r \in {r: X[r][i] * X[r][j] != 0}
				for (int k = 0; k < groupSize; k++) {
					int[] pair = group[k];
					int i = pair[0];
					int j = pair[1];
					int index = i * p + j;
					if (!map.containsKey(index)) {
						continue;
					}
					ArrayList<float[]> list = map.get(index);
					double Wij = WArr.get(k);
					double diff = Wij - W[i][j];
					if (diff == 0) {
						continue;
					}
					for (float[] element : list) {
						int r = (int) element[0];
						double hij = element[1];
						// e[r] -= (Wij - W[i][j]) * hij;
						// double e_r_old = e[r];
						e[r] -= diff * hij;
						/*if (Double.isInfinite(e[r]) && !Double.isInfinite(e_r_old)) {
							int a = 1;
							a = a + 1;
						}*/
					}
					W[i][j] = Wij;
				}
				
				/*if (debug) {
					ofv_new = computeOFV();
					printf("W[group(%d)] updated: %f\n", gIdx, ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating W[group(%d)]\n", gIdx);
					}
				}*/
				
			}
			
			if (debug) {
				ofv_new = computeOFV();
				printf("W updated: %f\n", ofv_new);
				if (ofv_old < ofv_new) {
					errf("Error when updating W]%n");
				}
			}
			
			cnt++;
			if (calcOFV) {
				double ofv = computeOFV();
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
			if (cnt >= maxIter) {
				break;
			}
		}
		
	}
	
	static void initializeWij() {
		System.out.println("Initialize Wij...");
		double[] OFVs = null;
		boolean debug = !true;
		int blockSize = 10;
		int maxIter = 1;
		if (calcOFV) {
			OFVs = allocate1DArray(maxIter + 1, 0);
			double ofv = 0;
			/*for (int i = 0; i < n; i++) {
				ofv += y[i] * y[i];
			}*/
			ofv = computeOFV();
			OFVs[0] = ofv;
			fprintf("Iter %d: %.10g\n", 0, ofv);
		}
		int cnt = 0;
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
			// Compute <W, X[r, :]>
			for (int k = jr[r]; k < jr[r + 1]; k++) {
				int j = ic[k];
				s += w[j] * pr[valCSRIndices[k]];
			}
			/*
			 * Compute \sum_{ij} x[i] * x[j] * W[i][j]
			 */
			for (int k1 = jr[r]; k1 < jr[r + 1]; k1++) {
				int i = ic[k1];
				double xi = pr[valCSRIndices[k1]];
				for (int k2 = k1 + 1; k2 < jr[r + 1]; k2++) {
					int j = ic[k2];
					double xj = pr[valCSRIndices[k2]];
					s += W[i][j] * xi * xj;
				}
			}
			y_hat[r] = s;
			e[r] = y[r] - s; // Why recalculated e[r] != e[r] updated till the end in last iteration
		}

		
			
		while (true) {
			
			// Update model parameters in one full iteration
			
			// We should have recalculated e[i] here
			/*if (cnt % blockSize == 0) {
				// Compute y_hat and cache e
				for (int r = 0; r < n; r++) {
					double s = b;
					// Compute <W, X[r, :]>
					for (int k = jr[r]; k < jr[r + 1]; k++) {
						int j = ic[k];
						s += w[j] * pr[valCSRIndices[k]];
					}
					
					 * Compute \sum_{ij} x[i] * x[j] * W[i][j]
					 
					for (int k1 = jr[r]; k1 < jr[r + 1]; k1++) {
						int i = ic[k1];
						double xi = pr[valCSRIndices[k1]];
						for (int k2 = k1 + 1; k2 < jr[r + 1]; k2++) {
							int j = ic[k2];
							double xj = pr[valCSRIndices[k2]];
							s += W[i][j] * xi * xj;
						}
					}
					y_hat[r] = s;
					e[r] = y[r] - s; // Why recalculated e[r] != e[r] updated till the end in last iteration
				}
			}*/
			
			/*for (int i = 0; i < 10; i++) {
				print(e[i] + "\t");
			}
			println();*/
			
			// Update b
			ofv_old = 0;
			if (debug) {
				ofv_old = computeOFV();
				printf("f(b): %f\n", ofv_old);
			}
			/*double b_new = (b * n + sum(e)) / (n + lambda);
			for (int i = 0; i < n; i++)
				e[i] -= (b_new - b);
			b = b_new;*/
			// println("b = " + b);
			
			if (debug) {
				ofv_new = computeOFV();
				printf("b updated: %f\n", ofv_new);
				if (ofv_old < ofv_new) {
					errf("Error when updating b\n");
				}
			}
			
			// Update w
			for (int j = p; j < p; j++) {
				
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
				
				/*if (debug) {
					ofv_new = computOFV();
					printf("w[%d] updated: %f\n", j, ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating w[%d]\n", j);
					}
				}*/
			}
			
			if (cnt == 3) {
				int a = 1;
				a = a + 1;
			}

			// Update W
			for (int index : map.keySet()) {
				int i = index / p;
				int j = index % p;
				ArrayList<float[]> list = map.get(index);
				// update W[i][j]
				if (debug && cnt >= 3) {
					ofv_old = computeOFV();
					// printf("f(W[%d, %d]): %f\n", i, j, ofv_old);
				}
				/*if (i == 1 && j == 1170 && cnt >= 3) {
					int a = 1;
					a = a + 1;
				}*/
				/*if (W[i][j] == 0) {
					continue;
				}*/
				// double lipschitz = 10 * L[i][j] / (nu / n);
				double lipschitz = 1 * L[i][j];
				double c = nu / lipschitz;
				// int maxCnt = 2;
				for (int k = 0; k < maxCnt; k++) {
					double Wij = W[i][j];
					double grad = 0;
					for (float[] element : list) {
						int r = (int) element[0];
						double hij = element[1];
						grad -= 2 * hij * e[r];
						if (Double.isInfinite(e[r])) {
							int a = 1;
							a = a + 1;
						}
					}
					// Wij^{t+1} = prox_{c||.||_1} (Wij^{t} - grad / L)
					Wij = Wij - grad / lipschitz;
					if (Math.abs(Wij) <= c) {
						Wij = 0;
					} else if (Wij > c) {
						Wij = Wij - c;
					} else {
						Wij = Wij + c;
					}
					/*if (W[i][j] == 0 && Wij != 0) {
						int a = 1;
						a = a + 1;
					}*/
					if (Double.isNaN(Wij)) {
						int a = 1;
						a = a + 1;
					}
					// Update e[r] for r \in {r: X[r][i] * X[r][j] != 0}
					double diff = Wij - W[i][j];
					if (diff == 0) {
						continue;
					}
					for (float[] element : list) {
						int r = (int) element[0];
						double hij = element[1];
						// e[r] -= (Wij - W[i][j]) * hij;
						double e_r_old = e[r];
						e[r] -= diff * hij;
						if (Double.isInfinite(e[r]) && !Double.isInfinite(e_r_old)) {
							int a = 1;
							a = a + 1;
						}
					}
					W[i][j] = Wij;
				}
				
				if (debug && cnt >= 3) {
					ofv_new = computeOFV();
					// printf("W[%d, %d] updated: %f\n", i, j, ofv_new);
					if (ofv_old < ofv_new) {
						errf("Error when updating W[%d,%d]\n", i, j);
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
				for (int i = 0; i < p; i++) {
					for (int j = i + 1; j < p; j++) {
						ofv += nu * Math.abs(W[i][j]);
					}
				}*/
				ofv = computeOFV();
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
			if (cnt >= maxIter) {
				break;
			}
		}
		
	}
	
	private static double computeOFV() {
	
		double ofv = 0;
		ofv += lambda * b * b;
		ofv += lambda * innerProduct(w, w);
		
		double T = 0;
		for (int[][] group : TreeStructuredPairGroupList) {
			double norm = 0;
			if (group.length == 1) {
				double Wij = W[group[0][0]][group[0][1]];
				T += Math.abs(Wij);
				continue;
			}
			for (int[] pair : group) {
				int i = pair[0];
				int j = pair[1];
				double Wij = W[i][j];
				norm += Wij * Wij;
			}
			T += Math.sqrt(norm);
		}
		ofv += nu * T;
		/*if (Double.isNaN(ofv)) {
			int a = 1;
			a = a + 1;
		}*/
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
				/*if (Double.isNaN(s)) {
					int a = 1;
					a = a + 1;
				}*/
			}
			// Compute \sum_{ij} x[i] * x[j] * W[i][j]
			for (int k1 = jr[r]; k1 < jr[r + 1]; k1++) {
				int i = ic[k1];
				double xi = pr[valCSRIndices[k1]];
				for (int k2 = k1 + 1; k2 < jr[r + 1]; k2++) {
					int j = ic[k2];
					double xj = pr[valCSRIndices[k2]];
					s += W[i][j] * xi * xj;
					/*if (Double.isNaN(s)) {
						int a = 1;
						a = a + 1;
					}*/
				}
			}
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
	
	private static int maxIter = 20;
	
	private static boolean calcOFV = false;
	
	/**
	 * Feature size.
	 */
	private static int p;
	
	/**
	 * Dimensionality of the latent space.
	 *//*
	private static int K = 8;*/
	
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
	 * w[j] is the linear weight for feature j.
	 */
	private static double[] w;
	
	/**
	 * Second order weights.
	 * W[i][j] is the weight for feature i and j.
	 */
	private static double[][] W;
	
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
	 *//*
	private static double[] q;*/
	
	/**
	 * Q[f][i] = \sum_j V[j][f] * X[i][j]
	 *//*
	private static double[][] Q;*/
	
	/**
	 * Array of feature vectors for all instances.
	 *//*
	private static Vector[] Xs = null;*/
	
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
	
	private static double nu;
	
	/**
	 * Lipschitz constant for W[i][j]
	 */
	private static double[][] L;
	
	private static int maxCnt;
	
	/**
	 * We need to record which instances have nonzero elements for
	 * pair(i, j).</br>
	 * HashMap<Pair<Integer, Integer>, ArrayList<Pair<Integer, Double>>> map.</br>
	 *                 i        j                        r  X[r][i] * X[r][j]
	 */
	static HashMap<Integer, ArrayList<float[]>> map = new
			HashMap<Integer, ArrayList<float[]>>();
	
	/**
	 * Test index: Training index or -1 if is new
	 */
	static TreeMap<Integer, Integer> TestIdx2TrainIdxUserMap = new TreeMap<Integer, Integer>();
	
	/**
	 * Test index: Training index or -1 if is new
	 */
	static TreeMap<Integer, Integer> TestIdx2TrainIdxItemMap = new TreeMap<Integer, Integer>();

	// static boolean drawMAPCurve = false;
	
	private static ArrayList<int[][]> UserTreeStructuredPairGroupList = new ArrayList<int[][]>();
	
	private static ArrayList<int[][]> ItemTreeStructuredPairGroupList = new ArrayList<int[][]>();
	
	/**
	 * Tree structured pair group list for the feature graph of the expanded 
	 * event feature vector.
	 */
	private static ArrayList<int[][]> TreeStructuredPairGroupList = new ArrayList<int[][]>();
	
	public static void buildTreeStructuredPairGroupList(
			String featureSizeFilePath, 
			String userFeatureGroupListFilePath, 
			String itemFeatureGroupListFilePath
			) {
		/*String appPath = StructuredRegression.class.getProtectionDomain().getCodeSource().getLocation().getPath();
		AppDirPath = new File(appPath).getParent();*/
		int[] featureSizes = getFeatureSize(featureSizeFilePath);
		loadUserTreeStructuredPairGroupList(userFeatureGroupListFilePath);
		loadItemTreeStructuredPairGroupList(itemFeatureGroupListFilePath);
		System.out.println("Loading TreeStructuredPairGroupList...");
		int Pu = featureSizes[0];
		int Pv = featureSizes[1];
		int Pe = featureSizes[2];
		TreeStructuredPairGroupList.addAll(UserTreeStructuredPairGroupList);
		TreeStructuredPairGroupList.addAll(ItemTreeStructuredPairGroupList);
		int p = Pu + Pv;
		int p2 = p + Pe;
		for (int i = 0; i < Pu; i++) {
			for (int j = Pu; j < p2; j++) {
				TreeStructuredPairGroupList.add(new int[][] {{i, j}});
			}
		}
		for (int i = Pu; i < p; i++) {
			for (int j = p; j < p2; j++) {
				TreeStructuredPairGroupList.add(new int[][] {{i, j}});
			}
		}
		for (int i = p; i < p2; i++) {
			for (int j = i + 1; j < p2; j++) {
				TreeStructuredPairGroupList.add(new int[][] {{i, j}});
			}
		}
	}
	
	/**
	 * If there isn't a hierarchical structure, we do nothing.
	 */
	private static void loadUserTreeStructuredPairGroupList(String userFeatureGroupListFilePath) {
		System.out.println("Loading UserTreeStructuredPairGroupList...");
		// String filePath = AppDirPath + File.separator + "UserTreeStructuredPairGroupList.txt";
		if (!new File(userFeatureGroupListFilePath).exists()) {
			System.out.println(String.format("File %s doesn't exist.\n", userFeatureGroupListFilePath));
			return;
		}
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(userFeatureGroupListFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		String line = "";
		String[] container = null;
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				container = line.split("\t");
				int[][] pairArr = new int[container.length][];
				if (container.length == 1) {
					int i = 0;
					int commaIdx = container[i].indexOf(',');
					int idx1 = Integer.parseInt(container[i].substring(1, commaIdx));
					int idx2 = Integer.parseInt(container[i].substring(commaIdx + 2, container[i].length() - 1));
					if (idx1 == idx2) {
						continue;
					}
					// pairArr[i] = new int[] {idx1, idx2};
					pairArr[i] = idx1 < idx2 ? new int[] {idx1, idx2} : new int[] {idx2, idx1};
					UserTreeStructuredPairGroupList.add(pairArr);
					continue;
				}
				for (int i = 0; i < container.length; i++) {
					int commaIdx = container[i].indexOf(',');
					int pIdx = Integer.parseInt(container[i].substring(1, commaIdx));
					int idx = Integer.parseInt(container[i].substring(commaIdx + 2, container[i].length() - 1));
					// Matcher matcher = pattern.matcher(line);
					// pairArr[i] = new int[] {pIdx, idx};
					pairArr[i] = pIdx < idx ? new int[] {pIdx, idx} : new int[] {idx, pIdx};
				}
				UserTreeStructuredPairGroupList.add(pairArr);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		/*int[] idxArr = UserTreeStructuredIndexGroupList.get(2401);
		System.out.println(idxArr);*/
	}
	
	/**
	 * If there isn't a hierarchical structure, we do nothing.
	 */
	private static void loadItemTreeStructuredPairGroupList(String itemFeatureGroupListFilePath) {
		System.out.println("Loading ItemTreeStructuredPairGroupList...");
		// String filePath = AppDirPath + File.separator + "ItemTreeStructuredPairGroupList.txt";
		if (!new File(itemFeatureGroupListFilePath).exists()) {
			System.out.println(String.format("File %s doesn't exist.\n", itemFeatureGroupListFilePath));
			return;
		}
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(itemFeatureGroupListFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			exit(1);
		}
		String line = "";
		String[] container = null;
		try {
			while ((line = br.readLine()) != null) {
				if (line.isEmpty())
					continue;
				container = line.split("\t");
				int[][] pairArr = new int[container.length][];
				if (container.length == 1) {
					int i = 0;
					int commaIdx = container[i].indexOf(',');
					int idx1 = Integer.parseInt(container[i].substring(1, commaIdx));
					int idx2 = Integer.parseInt(container[i].substring(commaIdx + 2, container[i].length() - 1));
					if (idx1 == idx2) {
						continue;
					}
					pairArr[i] = idx1 < idx2 ? new int[] {idx1, idx2} : new int[] {idx2, idx1};
					ItemTreeStructuredPairGroupList.add(pairArr);
					continue;
				}
				for (int i = 0; i < container.length; i++) {
					int commaIdx = container[i].indexOf(',');
					int pIdx = Integer.parseInt(container[i].substring(1, commaIdx));
					int idx = Integer.parseInt(container[i].substring(commaIdx + 2, container[i].length() - 1));
					// Matcher matcher = pattern.matcher(line);
					pairArr[i] = pIdx < idx ? new int[] {pIdx, idx} : new int[] {idx, pIdx};
				}
				ItemTreeStructuredPairGroupList.add(pairArr);
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	// private static Pattern pattern = Pattern.compile("[(](\\d+), (\\d+)[)]");

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
