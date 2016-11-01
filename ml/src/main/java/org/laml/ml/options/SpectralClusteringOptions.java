package org.laml.ml.options;

public class SpectralClusteringOptions extends ClusteringOptions {
	
	public String graphType;
	public double graphParam;
	public String graphDistanceFunction;
	public String graphWeightType;
	public double graphWeightParam;
	
	/**
	 * Constructor with default data member values:
	 * <p>
	 * graphType = "nn";</br>
     * graphParam = 6;</br>
	 * graphDistanceFunction = "euclidean";</br>
	 * graphWeightType = "heat";</br>
	 * graphWeightParam = 1;</br>
	 * </p>
	 */
	public SpectralClusteringOptions() {
		super();
		graphType = "nn";
		graphParam = 6;
		graphDistanceFunction = "euclidean";
		graphWeightType = "heat";
		graphWeightParam = 1;
	}
	
	public SpectralClusteringOptions(int nClus) {
		super(nClus);
		graphType = "nn";
		graphParam = 6;
		graphDistanceFunction = "euclidean";
		graphWeightType = "heat";
		graphWeightParam = 1;
	}
	
	public SpectralClusteringOptions(int nClus, 
			                         boolean verbose, 
			                         int maxIter,
			                         String graphType,
									 double graphParam,
									 String graphDistanceFunction,
									 String graphWeightType,
									 double graphWeightParam) {
		super(nClus, verbose, maxIter);
		this.graphType = graphType;
		this.graphParam = graphParam;
		this.graphDistanceFunction = graphDistanceFunction;
		this.graphWeightType = graphWeightType;
		this.graphWeightParam = graphWeightParam;
	}

	public SpectralClusteringOptions(ClusteringOptions clusteringOptions) {
		super(clusteringOptions);
		if (clusteringOptions instanceof SpectralClusteringOptions) {
			SpectralClusteringOptions options = (SpectralClusteringOptions)clusteringOptions;
			graphType = options.graphType;
			graphParam = options.graphParam;
			graphDistanceFunction = options.graphDistanceFunction;
			graphWeightType = options.graphWeightType;
			graphWeightParam = options.graphWeightParam;
		} else {
			graphType = "nn";
			graphParam = 6;
			graphDistanceFunction = "euclidean";
			graphWeightType = "heat";
			graphWeightParam = 1;
		}
		
	}

}
