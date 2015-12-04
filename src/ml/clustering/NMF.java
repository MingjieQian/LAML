package ml.clustering;

import static la.io.IO.loadMatrixFromDocTermCountFile;
import static la.io.IO.saveDenseMatrix;
import static ml.utils.Matlab.full;
import static ml.utils.Matlab.getTFIDF;
import static ml.utils.Matlab.normalizeByColumns;
import static ml.utils.Printer.printMatrix;
import static ml.utils.Time.tic;
import static ml.utils.Time.toc;
import la.matrix.Matrix;
import ml.options.KMeansOptions;
import ml.options.NMFOptions;
import ml.options.Options;

/***
 * A Java implementation for NMF which solves the following
 * optimization problem:
 * <p>
 * min || X - G * F ||_F^2</br>
 * s.t. G >= 0, F >= 0
 * </p>
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 31st, 2014
 */
public class NMF extends L1NMF {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		runNMF();

		String dataMatrixFilePath = "CNN - DocTermCount.txt";
		
		tic();
		Matrix X = loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		X = getTFIDF(X);
		X = normalizeByColumns(X);
		X = X.transpose();
		
		KMeansOptions kMeansOptions = new KMeansOptions();
		kMeansOptions.nClus = 10;
		kMeansOptions.maxIter = 50;
		kMeansOptions.verbose = true;
		
		KMeans KMeans = new KMeans(kMeansOptions);
		KMeans.feedData(X);
		KMeans.initialize(null);
		KMeans.clustering();
		
		Matrix G0 = KMeans.getIndicatorMatrix();
		
		// Matrix X = Data.loadSparseMatrix("X.txt");
		// Matrix G0 = Data.loadDenseMatrix("G0.txt");
		NMFOptions NMFOptions = new NMFOptions();
		NMFOptions.maxIter = 300;
		NMFOptions.verbose = true;
		NMFOptions.calc_OV = false;
		NMFOptions.epsilon = 1e-5;
		Clustering NMF = new NMF(NMFOptions);
		NMF.feedData(X);
		NMF.initialize(G0);
		
		// Matlab takes 12.5413 seconds
		// jblas takes 29.368 seconds
		// Commons-math takes 129 seconds (Using Array2DRowRealMatrix)
		// Commons-math takes 115 seconds (Using BlockRealMatrix)
		// start = System.currentTimeMillis();
		
		NMF.clustering();
		
		System.out.format("Elapsed time: %.3f seconds\n", toc());
		
		saveDenseMatrix("F.txt", NMF.centers);
		saveDenseMatrix("G.txt", NMF.indicatorMatrix);
		
	}
	
	public NMF(Options options) {
		super(options);
		gamma = 0;
		mu = 0;
	}
	
	public NMF(NMFOptions NMFOptions) {
		nClus = NMFOptions.nClus;
		maxIter = NMFOptions.maxIter;
		epsilon = NMFOptions.epsilon;
		verbose = NMFOptions.verbose;
		calc_OV = NMFOptions.calc_OV;
		gamma = 0;
		mu = 0;
	}
	
	public NMF() {
		Options options = new Options();
		nClus = options.nClus;
		maxIter = options.maxIter;
		epsilon = options.epsilon;
		verbose = options.verbose;
		calc_OV = options.calc_OV;
		gamma = 0;
		mu = 0;
	}
	
	public static void runNMF() {
		
		double[][] data = { 
				{3.5, 4.4, 1.3},
                {5.3, 2.2, 0.5},
                {0.2, 0.3, 4.1},
                {1.2, 0.4, 3.2} 
                };
        
		KMeansOptions options = new KMeansOptions();
        options.nClus = 2;
        options.verbose = true;
        options.maxIter = 100;
    
        KMeans KMeans= new KMeans(options);

        KMeans.feedData(data);
        KMeans.initialize(null);
        KMeans.clustering();
        Matrix G0 = KMeans.getIndicatorMatrix();
        
        NMFOptions NMFOptions = new NMFOptions();
        NMFOptions.nClus = 2;
		NMFOptions.maxIter = 50;
		NMFOptions.verbose = true;
		NMFOptions.calc_OV = false;
		NMFOptions.epsilon = 1e-5;
		Clustering NMF = new NMF(NMFOptions);
		
		NMF.feedData(data);
		// NMF.initialize(null);
		NMF.clustering(G0); // If null, KMeans will be used for initialization
		
		System.out.println("Basis Matrix:");
		printMatrix(full(NMF.getCenters()));
		
		System.out.println("Indicator Matrix:");
		printMatrix(full(NMF.getIndicatorMatrix()));
		
	}

}
