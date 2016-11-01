package org.laml.ml.options;

public class KMeansOptions {

	public int nClus;
	public int maxIter;
	public boolean verbose;
	
	public KMeansOptions() {

		nClus = -1;
		maxIter = 100;
		verbose = false;
		
	}
	
	public KMeansOptions(int nClus, int maxIter, boolean verbose) {

		this.nClus = nClus;
		this.maxIter = maxIter;
		this.verbose = verbose;
		
	}
	
	public KMeansOptions(int nClus, int maxIter) {
		
		this.nClus = nClus;
		this.maxIter = maxIter;
		this.verbose = false;
		
	}

}
