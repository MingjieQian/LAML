package la.io;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;
import java.util.TreeMap;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.matrix.SparseMatrix;
import ml.utils.Pair;

import static ml.utils.Printer.sprintf;

public class DataSet {
	
	/**
	 * The starting index for input data. Default value is 1 
	 * since LIBSVM and LDA use 1.
	 */
	public static int IdxStart = 1;

	/**
	 * Data matrix (nExample x nFeature).
	 */
	public Matrix X;

	/**
	 * Integer labels.
	 */
	public int[] Y;

	public DataSet() {
		X = null;
		Y = null;
	}

	public DataSet(Matrix X, int[] Y) {
		this.X = X;
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
	 * Write a data set to a text file in the LIBSVM format. Note that 
	 * an empty sparse feature vector will be written as an empty string. 
	 * Empty sparse vectors commonly occur when a test example only has 
	 * new features.
	 * 
	 * @param X an nExample x nFeature data matrix
	 * 
	 * @param Y labels
	 * 
	 * @param filePath file path
	 */
	public static void writeDataSet(Matrix X, int[] Y, String filePath) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new FileWriter(filePath));
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + filePath);
		}
		int numRows = X.getRowDimension();
		int numColumns = X.getColumnDimension();
		if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			double[] pr = ((SparseMatrix) X).getPr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			for (int i = 0; i < numRows; i++) {
				if (Y != null) {
					pw.printf("%s\t", Y[i]);
				}
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					int j = ic[k];
					double v = pr[valCSRIndices[k]];
					pw.printf("%d:%.8g ", j + IdxStart, v);
				}
				pw.println();
			}
		} else if (X instanceof DenseMatrix) {
			double[][] data = X.getData();
			for (int i = 0; i < numRows; i++) {
				if (Y != null) {
					pw.printf("%s\t", Y[i]);
				}
				double[] row = data[i];
				for (int j = 0; j < numColumns; j++) {
					double v = row[j];
					pw.printf("%d:%.8g ", j + IdxStart, v);
				}
				pw.println();
			}
		}
		if (!pw.checkError()) {
			pw.close();
			System.out.println("Dataset file written: " + filePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}
	}
	
	/**
	 * Write a data set to a string array in the LIBSVM format. Note that 
	 * an empty sparse feature vector will be written as an empty string. 
	 * Empty sparse vectors commonly occur when a test example only has 
	 * new features.
	 * 
	 * @param X an nExample x nFeature data matrix
	 * 
	 * @param Y labels
	 * 
	 * @return a {@code ArrayList<String>} instance, each element
	 *         is a string with LIBSVM data format
	 */
	public static ArrayList<String> writeDataSet(Matrix X, int[] Y) {
		ArrayList<String> res = new ArrayList<String>();
		int numRows = X.getRowDimension();
		int numColumns = X.getColumnDimension();
		StringBuilder sb = new StringBuilder();
		if (X instanceof SparseMatrix) {
			int[] ic = ((SparseMatrix) X).getIc();
			int[] jr = ((SparseMatrix) X).getJr();
			double[] pr = ((SparseMatrix) X).getPr();
			int[] valCSRIndices = ((SparseMatrix) X).getValCSRIndices();
			for (int i = 0; i < numRows; i++) {
				sb.setLength(0);
				if (Y != null) {
					sb.append(sprintf("%s\t", Y[i]));
				}
				for (int k = jr[i]; k < jr[i + 1]; k++) {
					int j = ic[k];
					double v = pr[valCSRIndices[k]];
					sb.append(sprintf("%d:%.8g ", j + IdxStart, v));
				}
				/*sb.append("\n");
				String line = sb.toString();
				if (!new StringTokenizer(line, " \t\n\r\f").hasMoreTokens())
					line = "";*/
				res.add(sb.toString());
			}
		} else if (X instanceof DenseMatrix) {
			double[][] data = X.getData();
			for (int i = 0; i < numRows; i++) {
				sb.setLength(0);
				if (Y != null) {
					sb.append(sprintf("%s\t", Y[i]));
				}
				double[] row = data[i];
				for (int j = 0; j < numColumns; j++) {
					double v = row[j];
					sb.append(sprintf("%d:%.8g ", j + IdxStart, v));
				}
				/*sb.append("\n");
				String line = sb.toString();
				if (!new StringTokenizer(line, " \t\n\r\f").hasMoreTokens())
					line = "";*/
				res.add(sb.toString());
			}
		}
		return res;
	}
	
	/**
	 * Read a data set from a string array. Note that an empty string 
	 * will be viewed as an empty sparse feature vector. Empty sparse 
	 * vectors commonly occur when a test example only has new features.
	 * 
	 * @param feaArray a {@code ArrayList<String>} instance, each element
	 *                 is a string with LIBSVM data format
	 *                 
	 * @return a {@code DataSet} instance
	 * 
	 * @throws InvalidInputDataException
	 * 
	 */
	public static DataSet readDataSetFromStringArray(ArrayList<String> feaArray) throws InvalidInputDataException {
		
		DataSet dataSet = new DataSet();
		List<Integer> vy = new ArrayList<Integer>();

		int max_index = 0;

		int lineNr = 0;

		StringTokenizer labelTokenizer = null;
		StringTokenizer featureTokenizer = null;
		String token;
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		int exampleIndex = 0;
		int featureIndex = -1;
		double value = 0;
		int nzmax = 0;
		String line = null;
		Iterator<String> lineIter = feaArray.iterator();
		while (lineIter.hasNext()) {
			line = lineIter.next();
			
			/*if (line == null || line.isEmpty())
				continue;*/
			if (line == null)
				continue;
			/*line = line.trim();
			if (line.isEmpty()) {
				vy.add(0);
				exampleIndex++;
				continue;
			}*/

			lineNr++;
			labelTokenizer = new StringTokenizer(line, " \t\n\r\f");
			featureTokenizer = new StringTokenizer(line, " \t\n\r\f:");

			try {
				token = labelTokenizer.nextToken();
			} catch (NoSuchElementException e) {
				vy.add(0);
				exampleIndex++;
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
				// Order by column then row
				map.put(Pair.of(featureIndex, exampleIndex), value);
				nzmax++;

			}
			exampleIndex++;
		}
		
		int numRows = exampleIndex;
		int numColumns = max_index + 1;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		// jc[numColumns] = k;

		Matrix X = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		int[] Y = new int[numRows];
		Iterator<Integer> iter = vy.iterator();
		rIdx = 0;
		while (iter.hasNext()) {
			Y[rIdx] = iter.next();
			rIdx++;
		}
		dataSet = new DataSet(X, Y);

		return dataSet;
		
	}

	/**
	 * Read a data set from a string array. Note that an empty string 
	 * will be viewed as an empty sparse feature vector. Empty sparse 
	 * vectors commonly occur when a test example only has new features.
	 * 
	 * @param feaArray a {@code ArrayList<String>} instance, each element
	 *                 is a string with LIBSVM data format
	 * @return a {@code DataSet} instance
	 * 
	 * @throws InvalidInputDataException
	 */
	public static DataSet readDataSet(ArrayList<String> feaArray) throws InvalidInputDataException {
		return readDataSetFromStringArray(feaArray);
	}
	
	/**
	 * Read a data set from a LIBSVM formatted file. Note that an empty 
	 * string will be viewed as an empty sparse feature vector. Empty 
	 * sparse vectors commonly occur when a test example only has new 
	 * features.
	 * 
	 * Data format (e.g., index starts from 1):<p/>
	 * <pre>
	 * +1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1 
	 * -1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1 
	 * </pre>
	 * @param filePath file path
	 * 
	 * @return a {@code DataSet} instance
	 * 
	 * @throws IOException
	 * 
	 * @throws InvalidInputDataException
	 * 
	 */
	public static DataSet readDataSetFromFile(String filePath) throws IOException, InvalidInputDataException {

		DataSet dataSet = new DataSet();
		BufferedReader fp = new BufferedReader(new FileReader(filePath));
		List<Integer> vy = new ArrayList<Integer>();

		int max_index = 0;

		int lineNr = 0;

		StringTokenizer labelTokenizer = null;
		StringTokenizer featureTokenizer = null;
		String token;
		TreeMap<Pair<Integer, Integer>, Double> map = new TreeMap<Pair<Integer, Integer>, Double>();
		int exampleIndex = 0;
		int featureIndex = -1;
		double value = 0;
		int nzmax = 0;
		String line = null;
		while ((line = fp.readLine()) != null) {
			
			if (line.isEmpty()) {
				vy.add(0);
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
				// Order by column then row
				map.put(Pair.of(featureIndex, exampleIndex), value);
				nzmax++;

			}
			exampleIndex++;
		}
		fp.close();

		int numRows = exampleIndex;
		int numColumns = max_index + 1;
		int[] ir = new int[nzmax];
		int[] jc = new int[numColumns + 1];
		double[] pr = new double[nzmax];

		int rIdx = -1;
		int cIdx = -1;
		int k = 0;
		jc[0] = 0;
		int currentColumn = 0;
		for (Entry<Pair<Integer, Integer>, Double> entry : map.entrySet()) {
			rIdx = entry.getKey().second;
			cIdx = entry.getKey().first;
			pr[k] = entry.getValue();
			ir[k] = rIdx;
			while (currentColumn < cIdx) {
				jc[currentColumn + 1] = k;
				currentColumn++;
			}
			k++;
		}
		while (currentColumn < numColumns) {
			jc[currentColumn + 1] = k;
			currentColumn++;
		}
		jc[numColumns] = k;

		Matrix X = SparseMatrix.createSparseMatrixByCSCArrays(ir, jc, pr, numRows, numColumns, nzmax);
		int[] Y = new int[numRows];
		Iterator<Integer> iter = vy.iterator();
		rIdx = 0;
		while (iter.hasNext()) {
			Y[rIdx] = iter.next();
			rIdx++;
		}
		dataSet = new DataSet(X, Y);

		return dataSet;

	}

	/**
	 * Read a data set from a LIBSVM formatted file. Note that an empty 
	 * string will be viewed as an empty sparse feature vector. Empty 
	 * sparse vectors commonly occur when a test example only has new 
	 * features.
	 * 
	 * Data format (e.g., index starts from 1):<p/>
	 * <pre>
	 * +1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1 
	 * -1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1 
	 * </pre>
	 * @param filePath file path
	 * 
	 * @return a {@code DataSet} instance
	 * 
	 * @throws IOException
	 * 
	 * @throws InvalidInputDataException
	 */
	public static DataSet readDataSet(String filePath) throws IOException, InvalidInputDataException {
		return readDataSetFromFile(filePath);
	}
}
