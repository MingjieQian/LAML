package ml.clustering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import ml.options.ClusteringOptions;

/**
 * Abstract class for clustering algorithms.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 30th, 2014
 */
public abstract class Clustering {

	/**
	 * Number of clusters.
	 */
	public int nClus;
	
	/**
	 * Number of features.
	 */
	public int nFeature;
	
	/**
	 * Number of examples.
	 */
	public int nExample;
	
	/**
	 * Data matrix (nExample x nFeature), each row is a feature vector
	 */
	protected Matrix dataMatrix;
	
	/**
	 * Cluster indicator matrix (nExample x nClus).
	 */
	protected Matrix indicatorMatrix;
	
	/**
	 * Basis matrix (nClus x nFeature).
	 */
	protected Matrix centers;
	
	/**
	 * Default constructor for this clustering algorithm.
	 */
	public Clustering() {
		this.nClus = 0;
	}
	
	/**
	 * Constructor for this clustering algorithm initialized with options
	 * wrapped in a {@code ClusteringOptions} object.
	 * 
	 * @param clusteringOptions clustering options
	 * 
	 */
	public Clustering(ClusteringOptions clusteringOptions) {
		this.nClus = clusteringOptions.nClus;
	}
	
	/**
	 * Constructor for this clustering algorithm given number of
	 * clusters to be set.
	 * 
	 * @param nClus number of clusters
	 * 
	 */
	public Clustering(int nClus) {
		if (nClus < 1) {
			System.err.println("Number of clusters less than one!");
			System.exit(1);
		}
		this.nClus = nClus;
	}
	
	/**
	 * Feed training data for this clustering algorithm.
	 * 
	 * @param dataMatrix an nExample x nFeature data matrix with each row being
	 *                   a data example
	 * 
	 */
	public void feedData(Matrix dataMatrix) {
		this.dataMatrix = dataMatrix;
		nExample = dataMatrix.getRowDimension();
		nFeature = dataMatrix.getColumnDimension();
	}
	
	/**
	 * Feed training data for this feature selection algorithm.
	 * 
	 * @param data an nExample x nFeature 2D {@code double} array with each
	 *             row being a data example
	 * 
	 */
	public void feedData(double[][] data) {
		this.feedData(new DenseMatrix(data));
	}
	
	/**
	 * Initialize the indicator matrix.
	 * 
	 * @param G0 initial indicator matrix
	 * 
	 */
	public void initialize(Matrix G0) {
		
		if (G0 != null) {
			this.indicatorMatrix = G0;
			return;
		}
		List<Integer> indList = new ArrayList<Integer>();
		for (int i = 0; i < nExample; i++) {
			indList.add(i);
		}
		
		Random rdn = new Random(System.currentTimeMillis());
		Collections.shuffle(indList, rdn);
		
		indicatorMatrix = new SparseMatrix(nExample, nClus);
		
		for (int i = 0; i < nClus; i++) {
			indicatorMatrix.setEntry(indList.get(i), i, 1);
		}
		
	}
	
	/**
	 * Do clustering. Please call initialize() before
	 * using this method. 
	 */
	public abstract void clustering();
	
	/**
	 * Do clustering with a specified initializer. Please use null if
	 * you want to use random initialization.
	 * 
	 * @param G0 initial indicator matrix, if null random initialization
	 *           will be used
	 */
	public void clustering(Matrix G0) {
		initialize(G0);
		clustering();
	}
	
	/**
	 * Fetch data matrix.
	 * 
	 * @return an nExample x nFeature data matrix
	 * 
	 */
	public Matrix getData() {
		return dataMatrix;
	}
	
	/**
	 * Get cluster centers.
	 * 
	 * @return an nClus x nFeature basis matrix
	 * 
	 */
	public Matrix getCenters() {
		return centers;
	}
	
	/**
	 * Get cluster indicator matrix.
	 * 
	 * @return an nExample x nClus cluster indicator matrix
	 * 
	 */
	public Matrix getIndicatorMatrix() {
		return indicatorMatrix;
	}
	
	/**
	 * Evaluating the clustering performance of this clustering algorithm
	 * by using the ground truth.
	 * 
	 * @param G predicted cluster indicator matrix
	 * 
	 * @param groundTruth true cluster assignments
	 * 
	 * @return evaluation metrics
	 * 
	 */
	public static double getAccuracy(Matrix G, Matrix groundTruth) {
		// To do
		System.out.println("Sorry, this function has not been implemented yet...");
		return 0;
	}
	
}
