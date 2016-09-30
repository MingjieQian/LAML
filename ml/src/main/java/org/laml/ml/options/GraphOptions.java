package ml.options;

/**
 * A structure to build a data graph.
 * <pre>
 *              Field Name:  Description                                         : default
 * -------------------------------------------------------------------------------------------
 *             'graphType':  'nn' | 'epsballs'                                   : 'nn'   
 *            'graphParam':  number of nearest neighbor size of 'epsballs'       :  6
 *            'kernelType':  'linear' | 'rbf' | 'poly' | 'cosine'                : 'linear'
 *           'kernelParam':     --    | sigma | degree |    --                   :  1
 * 'graphDistanceFunction':  distance function for graph: 'euclidean' | 'cosine' : 'euclidean'
 *       'graphWeightType':  'binary' | 'distance' | 'heat'                      : 'binary'
 *      'graphWeightParam':  e.g. for heat kernel, width to use                  :  1
 *        'graphNormalize':  Use normalized graph Laplacian (1) or not (0)       :  1
 *            'classEdges':  Disconnect edges across classes:yes(1) no (0)       :  0
 *               'gamma_A':  RKHS norm regularization parameter (Ambient)        :  1
 *               'gamma_I':  Manifold regularization parameter  (Intrinsic)      :  1
 * -------------------------------------------------------------------------------------------
 * Note: Kernel and KernelParam are meant for calcKernel function.
 * </pre>
 * 
 * @version 1.0 Jan. 27th, 2014
 * @author Mingjie Qian
 */
public class GraphOptions {
	
	public String graphType;
	public double graphParam;
	public String kernelType;
	public double kernelParam;
	public String graphDistanceFunction;
	public String graphWeightType;
	public double graphWeightParam;
	public boolean graphNormalize;
	public boolean classEdges;
	
	/**
	 * Generate/alter {@code GraphOptions} structure to build a data graph.
	 * <pre>
	 *              Field Name:  Description                                         : default
	 * -------------------------------------------------------------------------------------------
     *             'graphType':  'nn' | 'epsballs'                                   : 'nn'   
     *            'graphParam':  number of nearest neighbor size of 'epsballs'       :  6
     *            'kernelType':  'linear' | 'rbf' | 'poly' | 'cosine'                : 'linear'
     *           'kernelParam':     --    | sigma | degree |    --                   :  1
     * 'graphDistanceFunction':  distance function for graph: 'euclidean' | 'cosine' : 'euclidean'
     *       'graphWeightType':  'binary' | 'distance' | 'heat'                      : 'binary'
     *      'graphWeightParam':  e.g. for heat kernel, width to use                  :  1
     *        'graphNormalize':  Use normalized graph Laplacian (1) or not (0)       :  1
     *            'classEdges':  Disconnect edges across classes:yes(1) no (0)       :  0
     *               'gamma_A':  RKHS norm regularization parameter (Ambient)        :  1
     *               'gamma_I':  Manifold regularization parameter  (Intrinsic)      :  1
     * -------------------------------------------------------------------------------------------
     * Note: Kernel and KernelParam are meant for calcKernel function.
     * </pre>
	 */
	public GraphOptions() {
		graphType = "nn";
		kernelType = "linear";
		kernelParam = 1;
		graphParam = 6;
		graphDistanceFunction = "euclidean";
		graphWeightType = "binary";
		graphWeightParam = 1;
		graphNormalize = true;
		classEdges = false;
	}
	
	public GraphOptions(GraphOptions graphOtions) {
		graphType = graphOtions.graphType;
		kernelType = graphOtions.kernelType;
		kernelParam = graphOtions.kernelParam;
		graphParam = graphOtions.graphParam;
		graphDistanceFunction = graphOtions.graphDistanceFunction;
		graphWeightType = graphOtions.graphWeightType;
		graphWeightParam = graphOtions.graphWeightParam;
		graphNormalize = graphOtions.graphNormalize;
		classEdges = graphOtions.classEdges;
	}

}
