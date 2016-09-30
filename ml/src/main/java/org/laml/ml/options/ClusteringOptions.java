package ml.options;

public class ClusteringOptions {
	
	public int nClus;
	public boolean verbose;
	public int maxIter;
	
	public ClusteringOptions() {
		nClus = 0;
		verbose = false;
		maxIter = 100;
	}
	
	public ClusteringOptions(int nClus) {
		if (nClus < 1) {
			System.err.println("Number of clusters less than one!");
			System.exit(1);
		}
		this.nClus = nClus;
		verbose = false;
		maxIter = 100;
	}
	
	public ClusteringOptions(int nClus, boolean verbose, int maxIter) {
		this.nClus = nClus;
		this.verbose = verbose;
		this.maxIter = maxIter;
	}
	
	public ClusteringOptions(ClusteringOptions options) {
		nClus = options.nClus;
		verbose = options.verbose;
		maxIter = options.maxIter;
	}

}
