package org.laml.la.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.laml.la.vector.SparseVector;
import org.laml.la.vector.Vector;

public class DataVectors {
	
	/**
	 * The starting index for input data. Default value is 1 
	 * since libsvm and lda use 1.
	 */
	public static int IdxStart = 1;
	
	public Vector[] Vs;
	
	public int[] Y;
	
	public DataVectors() {
		Vs = null;
		Y = null;
	}

	public DataVectors(Vector[] Vs, int[] Y) {
		this.Vs = Vs;
		this.Y = Y;
	}

	/**
	 * @param s the string to parse for the double value
	 * @throws IllegalArgumentException if s is empty or represents NaN or Infinity
	 * @throws NumberFormatException see {@link Double#parseDouble(String)}
	 */
	static double atof(String s) {
		if (s == null || s.length() < 1) throw new IllegalArgumentException("Can't convert empty string to integer");
		double d = Double.parseDouble(s);
		if (Double.isNaN(d) || Double.isInfinite(d)) {
			throw new IllegalArgumentException("NaN or Infinity in input: " + s);
		}
		return (d);
	}

	/**
	 * @param s the string to parse for the integer value
	 * @throws IllegalArgumentException if s is empty
	 * @throws NumberFormatException see {@link Integer#parseInt(String)}
	 */
	static int atoi(String s) throws NumberFormatException {
		if (s == null || s.length() < 1) throw new IllegalArgumentException("Can't convert empty string to integer");
		// Integer.parseInt doesn't accept '+' prefixed strings
		if (s.charAt(0) == '+') s = s.substring(1);
		return Integer.parseInt(s);
	}
	
	/**
	 * Read data vectors from a string array. Note that an empty string 
	 * or null will be viewed as an empty sparse vector. Empty sparse
	 * vectors commonly occur when a test example only has new 
	 * features.
	 * 
	 * @param feaArray a {@code ArrayList<String>}, each element
	 *                 is a string with LIBSVM data format
	 *                 
	 * @return a {@code DataVectors} instance
	 * 
	 * @throws InvalidInputDataException
	 * 
	 */
	public static DataVectors readDataVectorsFromStringArray(ArrayList<String> feaArray) throws InvalidInputDataException {
		
		DataVectors dataVectors = new DataVectors();
		List<Integer> vy = new ArrayList<Integer>();

		int max_index = 0;

		int lineNr = 0;

		StringTokenizer labelTokenizer = null;
		StringTokenizer featureTokenizer = null;
		String token;
		
		List<Vector> dataVectorList = new ArrayList<Vector>();
		
		int exampleIndex = 0;
		int featureIndex = -1;
		double value = 0;
		String line = null;
		Iterator<String> lineIter = feaArray.iterator();
		while (lineIter.hasNext()) {
			line = lineIter.next();
			
			if (line == null || line.isEmpty()) {
				dataVectorList.add(new SparseVector(max_index + 1));
				exampleIndex++;
				continue;
			}
			
			ArrayList<Integer> indexList = new ArrayList<Integer>();
			ArrayList<Double> valueList = new ArrayList<Double>();
			
			lineNr++;
			labelTokenizer = new StringTokenizer(line, " \t\n\r\f");
			featureTokenizer = new StringTokenizer(line, " \t\n\r\f:");

			try {
				token = labelTokenizer.nextToken();
			} catch (NoSuchElementException e) {
				continue;
			}

			if (token.contains(":")) { // No label available
				vy.add(0);
			} else {
				token = featureTokenizer.nextToken();
				try {
					vy.add(atoi(token));
				} catch (NumberFormatException e) {
					/*
					 * Sometimes label can be 1.0000
					 */
					try {
						vy.add((int)atof(token));
					} catch (NumberFormatException e2) {
						throw new InvalidInputDataException("invalid label: " + token, lineNr, e);
					}
				}
			}

			int m = featureTokenizer.countTokens() / 2;

			for (int j = 0; j < m; j++) {

				token = featureTokenizer.nextToken();
				try {
					featureIndex = atoi(token) - IdxStart;
				} catch (NumberFormatException e) {
					throw new InvalidInputDataException("invalid index: " + token, lineNr, e);
				}

				// assert that indices are valid
				if (featureIndex < 0) throw new InvalidInputDataException("invalid index: " + featureIndex, lineNr);

				token = featureTokenizer.nextToken();
				try {
					value = atof(token);
				} catch (NumberFormatException e) {
					throw new InvalidInputDataException("invalid value: " + token, lineNr);
				}
				max_index = Math.max(max_index, featureIndex);
				if (value == 0) {
					continue;
				}
				indexList.add(featureIndex);
				valueList.add(value);

			}
			int nnz = m;
			int[] ir = new int[nnz];
			double[] pr = new double[nnz];
			Iterator<Integer> indexIter = indexList.iterator();
			Iterator<Double> valueIter = valueList.iterator();
			for (int k = 0; k < nnz; k++) {
				ir[k] = indexIter.next();
				pr[k] = valueIter.next();
			}
			dataVectorList.add(new SparseVector(ir, pr, nnz, max_index + 1));
			exampleIndex++;
			
		}
		
		int numRows = exampleIndex;
		int numColumns = max_index + 1;
		
		int[] Y = new int[numRows];
		Iterator<Integer> iter = vy.iterator();
		int rIdx = 0;
		while (iter.hasNext()) {
			Y[rIdx] = iter.next();
			rIdx++;
		}
		
		Vector[] Vs = new Vector[numRows];
		Iterator<Vector> dataVectorIter = dataVectorList.iterator();
		for (int k = 0; k < numRows; k++) {
			Vs[k] = dataVectorIter.next();
			((SparseVector) Vs[k]).setDim(numColumns);
		}
		dataVectors = new DataVectors(Vs, Y);

		return dataVectors;
		
	}
	
	/**
	 * Read data vectors from a LIBSVM formatted file. 
	 * Data format (index starts from 1):<p/>
	 * <pre>
	 * +1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1 
	 * -1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1 
	 * </pre>
	 * Note that an empty string will be viewed as an empty sparse 
	 * vector. Empty sparse vectors commonly occur when a test 
	 * example only has new features.
	 * 
	 * @param filePath file path
	 * 
	 * @return a {@code DataVectors} instance
	 * 
	 * @throws IOException
	 * 
	 * @throws InvalidInputDataException
	 * 
	 */
	public static DataVectors readDataSetFromFile(String filePath) throws IOException, InvalidInputDataException {

		DataVectors dataVectors = new DataVectors();
		BufferedReader fp = new BufferedReader(new FileReader(filePath));
		List<Integer> vy = new ArrayList<Integer>();

		int max_index = 0;

		int lineNr = 0;

		StringTokenizer labelTokenizer = null;
		StringTokenizer featureTokenizer = null;
		String token;

		List<Vector> dataVectorList = new ArrayList<Vector>();
		
		int exampleIndex = 0;
		int featureIndex = -1;
		double value = 0;
		String line = null;
		while ((line = fp.readLine()) != null) {
			
			if (line.isEmpty()) {
				dataVectorList.add(new SparseVector(max_index + 1));
				exampleIndex++;
				continue;
			}

			lineNr++;
			labelTokenizer = new StringTokenizer(line, " \t\n\r\f");
			featureTokenizer = new StringTokenizer(line, " \t\n\r\f:");

			try {
				token = labelTokenizer.nextToken();
			} catch (NoSuchElementException e) {
				continue;
			}
			
			ArrayList<Integer> indexList = new ArrayList<Integer>();
			ArrayList<Double> valueList = new ArrayList<Double>();

			if (token.contains(":")) { // No label available
				vy.add(0);
			} else {
				token = featureTokenizer.nextToken();
				try {
					vy.add(atoi(token));
				} catch (NumberFormatException e) {
					/*
					 * Sometimes label can be 1.0000
					 */
					try {
						vy.add((int)atof(token));
					} catch (NumberFormatException e2) {
						fp.close();
						throw new InvalidInputDataException("invalid label: " + token, lineNr, e);
					}
				}
			}

			int m = featureTokenizer.countTokens() / 2;

			for (int j = 0; j < m; j++) {

				token = featureTokenizer.nextToken();
				try {
					featureIndex = atoi(token) - IdxStart;
				} catch (NumberFormatException e) {
					fp.close();
					throw new InvalidInputDataException("invalid index: " + token, filePath, lineNr, e);
				}

				// assert that indices are valid
				if (featureIndex < 0) {
					fp.close();
					throw new InvalidInputDataException("invalid index: " + featureIndex, filePath, lineNr);
				}

				token = featureTokenizer.nextToken();
				try {
					value = atof(token);
				} catch (NumberFormatException e) {
					fp.close();
					throw new InvalidInputDataException("invalid value: " + token, filePath, lineNr);
				}
				max_index = Math.max(max_index, featureIndex);
				if (value == 0) {
					continue;
				}
				indexList.add(featureIndex);
				valueList.add(value);

			}
			int nnz = m;
			int[] ir = new int[nnz];
			double[] pr = new double[nnz];
			Iterator<Integer> indexIter = indexList.iterator();
			Iterator<Double> valueIter = valueList.iterator();
			for (int k = 0; k < nnz; k++) {
				ir[k] = indexIter.next();
				pr[k] = valueIter.next();
			}
			dataVectorList.add(new SparseVector(ir, pr, nnz, max_index + 1));
			exampleIndex++;
		}
		fp.close();

		int numRows = exampleIndex;
		int numColumns = max_index + 1;
		
		int[] Y = new int[numRows];
		Iterator<Integer> iter = vy.iterator();
		int rIdx = 0;
		while (iter.hasNext()) {
			Y[rIdx] = iter.next();
			rIdx++;
		}
		
		Vector[] Vs = new Vector[numRows];
		Iterator<Vector> dataVectorIter = dataVectorList.iterator();
		for (int k = 0; k < numRows; k++) {
			Vs[k] = dataVectorIter.next();
			((SparseVector) Vs[k]).setDim(numColumns);
		}
		dataVectors = new DataVectors(Vs, Y);

		return dataVectors;
		
	}
	
}
