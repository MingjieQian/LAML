package ml.clustering;

import static la.io.IO.loadMatrix;
import static la.vector.DenseVector.buildDenseVector;
import static ml.utils.InPlaceOperator.clear;
import static ml.utils.InPlaceOperator.plusAssign;
import static ml.utils.InPlaceOperator.timesAssign;
import static ml.utils.Matlab.full;
import static ml.utils.Matlab.getTFIDF;
import static ml.utils.Matlab.l2DistanceSquare;
import static ml.utils.Matlab.max;
import static ml.utils.Matlab.min;
import static ml.utils.Matlab.norm;
import static ml.utils.Matlab.normalizeByColumns;
import static ml.utils.Matlab.sum;
import static ml.utils.Printer.printMatrix;
import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import la.vector.DenseVector;
import la.vector.SparseVector;
import la.vector.Vector;
import ml.options.KMeansOptions;
import ml.utils.ArrayOperator;
import ml.utils.Matlab;

/***
 * A Java implementation for KMeans.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 30th, 2014
 */
public class KMeans extends Clustering {

	KMeansOptions options;
	
	public KMeans(int nClus) {
		super(nClus);
		options.maxIter = 100;
		options.verbose = false;
	}
	
	public KMeans(int nClus, int maxIter) {
		super(nClus);
		options.maxIter = maxIter;
		options.verbose = false;
	}
	
	public KMeans(int nClus, int maxIter, boolean verbose) {
		super(nClus);
		options.maxIter = maxIter;
		options.verbose = verbose;
	}
	
	public KMeans(KMeansOptions options) {
		super(options.nClus);
		/*if (options.nClus == -1) {
			System.err.println("Number of clusters undefined!");
			System.exit(1);
		} else if (options.nClus == 0) {
			System.err.println("Number of clusters is zero!");
			System.exit(1);
		}*/
		this.options = options;
	}
	
	/*@Override
	public void initialize(Matrix G0) {
		
		if (G0 != null) {
			this.indicatorMatrix = G0;
			return;
		}
		List<Integer> indList = new ArrayList<Integer>();
		for (int i = 0; i < nSample; i++) {
			indList.add(i);
		}
		
		Random rdn = new Random(System.currentTimeMillis());
		Collections.shuffle(indList, rdn);
		
		indicatorMatrix = new SparseMatrix(nSample, nClus);
		
		for (int i = 0; i < nClus; i++) {
			indicatorMatrix.setEntry(indList.get(i), i, 1);
		}
		
	}*/
	
	/**
	 * Initializer needs not be explicitly specified. If the initial
	 * indicator matrix is not given, random initialization will be
	 * used.
	 */
	@Override
	public void clustering() {
		
		if (indicatorMatrix == null) {
			initialize(null);
		}
		
		Vector[] maxRes = max(indicatorMatrix, 2);
		double[] indicators = new double[nExample];
		for (int i = 0; i < nExample; i++) {
			if (maxRes[0].get(i) != 0)
				indicators[i] = maxRes[1].get(i);
			else
				indicators[i] = -1;
		}
		
		double[] clusterSizes = ArrayOperator.allocate1DArray(nClus, 0);
		
		// double[] indicators = full(max(indicatorMatrix, 2)[1]).getPr();
		Vector[] examples = null;
		if (dataMatrix instanceof SparseMatrix) {
			examples = Matlab.sparseMatrix2SparseRowVectors(dataMatrix);
		} else if (dataMatrix instanceof DenseMatrix) {
			double[][] data = ((DenseMatrix) dataMatrix).getData();
			examples = new Vector[nExample];
			for (int i = 0; i < nExample; i++) {
				examples[i] = buildDenseVector(data[i]);
			}
		}
		
		Vector[] centers = new Vector[nClus];
		if (dataMatrix instanceof DenseMatrix) {
			for (int k = 0; k < nClus; k++) {
				centers[k] = new DenseVector(nFeature);
			}
		} else {
			for (int k = 0; k < nClus; k++) {
				centers[k] = new SparseVector(nFeature);
			}
		}
		
		// Compute the initial centers
		for (int i = 0; i < nExample; i++) {
			int k = (int) indicators[i];
			if (k == -1)
				continue;
			plusAssign(centers[k], examples[i]);
			clusterSizes[k]++;
		}
		for (int k = 0; k < nClus; k++) {
			timesAssign(centers[k], 1 / clusterSizes[k]);
		}
		
		int cnt = 0;
		
		Matrix DistMatrix = null;
		double mse = 0;
		
		/*TreeMap<String, Matrix> minResult = null;
		Matrix minMatrix = null;
		Matrix idxMatrix = null;*/
		
		/*List<Integer> indList = new ArrayList<Integer>();
		for (int i = 0; i < nSample; i++) {
			indList.add(i);
		}*/
		
		/*Random rdn = new Random(System.currentTimeMillis());
		Collections.shuffle(indList, rdn);*/
		
		/*for (int i = 0; i < nClus; i++) {
			indicatorMatrix.setEntry(indList.get(i), i, 1);
		}*/
		
		while (cnt < options.maxIter) {
			
			Matrix indOld = indicatorMatrix;
			
			long start = System.currentTimeMillis();
			
			DistMatrix = l2DistanceSquare(centers, examples);
			// Printer.disp(DistMatrix);
			
			Vector[] minRes = min(DistMatrix);
			Vector minVals = minRes[0];
			Vector IX = minRes[1];
			
			indicatorMatrix = new SparseMatrix(nExample, nClus);
			for (int i = 0; i < nExample; i++) {
				indicatorMatrix.setEntry(i, (int)IX.get(i), 1);
				indicators[i] = IX.get(i);
			}			
			
			mse = sum(minVals) / nExample;
			
			// Debug
			/*if (Double.isNaN(sse)) {
				int a = 1;
				a = a + 1;
				Matlab.display(DistMatrix);
				Matlab.display(dataMatrix.getColumnMatrix(6));
				Matlab.display(centers.getColumnMatrix(0));
				Matlab.display(Matlab.norm(dataMatrix.getColumnMatrix(6).minus(centers.getColumnMatrix(0))));
				Matlab.display(Matlab.l2Distance(dataMatrix.getColumnMatrix(6), centers));
				Matlab.display(Matlab.l2DistanceSquare(dataMatrix.getColumnMatrix(6), centers));
			}*/
			
			if (norm(indOld.minus(indicatorMatrix), "fro") == 0) {
				System.out.println("KMeans complete.");
				break;
			}
			
			double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
			
			cnt += 1;
			
			if (options.verbose) {
				System.out.format("Iter %d: mse = %.3f (%.3f secs)\n", cnt, mse, elapsedTime);
			}
			
			// Compute the centers
			clear(clusterSizes);
			for (int k = 0; k < nClus; k++) {
				centers[k].clear();
			}
			for (int i = 0; i < nExample; i++) {
				int k = (int) indicators[i];
				plusAssign(centers[k], examples[i]);
				clusterSizes[k]++;
			}
			for (int k = 0; k < nClus; k++) {
				timesAssign(centers[k], 1 / clusterSizes[k]);
			}
			
		}
		
		if (dataMatrix instanceof SparseMatrix) {
			this.centers = Matlab.sparseRowVectors2SparseMatrix(centers);
		} else if (dataMatrix instanceof DenseMatrix) {
			this.centers = Matlab.denseRowVectors2DenseMatrix(centers);
		}
		
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		runKMeans();

		int K = 3;
		int maxIter = 100;
		boolean verbose = true;
		KMeansOptions options = new KMeansOptions(K, maxIter, verbose);
		Clustering KMeans = new KMeans(options);
		
		//double[][] matrixData = { {1d, 0d, 3d}, {2d, 5d, 3d}, {4d, 1d, 0d}, {3d, 0d, 1d}, {2d, 5d, 3d}};
		double[][] matrixData2 = { {1d, 0d, 3d, 2d, 0d}, 
				                   {2d, 5d, 3d, 1d, 0d}, 
				                   {4d, 1d, 0d, 0d, 1d}, 
				                   {3d, 0d, 1d, 0d, 2d}, 
				                   {2d, 5d, 3d, 1d, 6d} };
		
		Matrix dataMatrix = new DenseMatrix(matrixData2);
		printMatrix(dataMatrix);
		
		Matrix X = loadMatrix("CNNTest-TrainingData.txt");
		Matrix X2 = normalizeByColumns(getTFIDF(X));
		X2 = X2.transpose();
		KMeans.feedData(X2);
		Matrix initializer = loadMatrix("indicators");
		initializer = null;
		KMeans.initialize(initializer);
		KMeans.clustering();
		
		/*System.out.println("Input data matrix:");
		Matlab.printMatrix(dataMatrix);*/
		
		System.out.println("Indicator Matrix:");
		printMatrix(full(KMeans.getIndicatorMatrix()));
		
	}
	
	public static void runKMeans() {
        
		double[][] data = {
				{3.5, 5.3, 0.2, -1.2},
				{4.4, 2.2, 0.3, 0.4},
				{1.3, 0.5, 4.1, 3.2}
			    };
		
		KMeansOptions options = new KMeansOptions();
        options.nClus = 2;
        options.verbose = true;
        options.maxIter = 100;
    
        KMeans KMeans= new KMeans(options);
        
        KMeans.feedData(data);
        // KMeans.initialize(null);
        Matrix initializer = null;
        initializer = new SparseMatrix(3, 2);
        initializer.setEntry(0, 0, 1);
        initializer.setEntry(1, 1, 1);
        initializer.setEntry(2, 0, 1);
        KMeans.clustering(initializer); // Use null for random initialization
        
        System.out.println("Indicator Matrix:");
		printMatrix(full(KMeans.getIndicatorMatrix()));
		
    }

}
