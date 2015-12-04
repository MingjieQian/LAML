package ml.utils;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Utility {
	
	/**
     * Read the contents from a text file into a string.
     * 
     * @param filePath
     *        A {@code String} specifying the filename to be read into another {@code String}.
	 *  
     * @return A {@code String} containing the contents from the input file.
     * 
     * @throws java.io.IOException
     */
    public static String readFileAsString(String filePath) throws java.io.IOException{
	    byte[] buffer = new byte[(int) new File(filePath).length()];
	    BufferedInputStream f = null;
	    try {
	        f = new BufferedInputStream(new FileInputStream(filePath));
		    //System.out.println("read file "+filePath);
	        f.read(buffer);
	    } finally {
	        if (f != null) 
	        	try { f.close(); } catch (IOException ignored) { }
	    }
	    return new String(buffer);
	}

	/**
	 * Split a command line into an argument array. This routine 
	 * segments the given command line by whitespace characters 
	 * with the exception that strings between double quotation 
	 * marks are considered to be single arguments, and will not 
	 * be split.
	 * 
	 * @param command the original command line from a shell
	 * 
	 * @return argument array split from command 
	 * 
	 */
	public static String[] splitCommand(String command) {

		char[] commandCharArr = command.toCharArray();
		int idx = 0;
		int beginPos = 0;
		int endPos = 0;
		char ch = ' ';
		/*
		 * Quote Matching State:
		 * 0: Start
		 * 1: Left double quotation mark found
		 * 2: Right double quotation mark found
		 */
		int state = 0;
		String argument = "";
		ArrayList<String> argList = new ArrayList<String>();
		while (idx < commandCharArr.length) {
			ch = commandCharArr[idx];
			// System.out.print(ch);
			if (ch == '"' && state == 0) { // Left double quotation mark is found
				beginPos = idx + 1;
				state = 1;
			} else if (ch == '"' && state == 1) { // Right double quotation mark is found
				state = 2;
			} else if (ch == ' ') {
				if (state == 2) { // Right double quotation mark has already been found
					endPos = idx - 1;
					state = 0;
				} else if (state == 1) { // Between left and right quotation marks
					idx++;
					continue;
				} else {
					endPos = idx;
				}
				argument = command.substring(beginPos, endPos).trim();
				if (!argument.isEmpty())
					argList.add(argument);
				// System.out.println();
				while (idx < commandCharArr.length && commandCharArr[idx] == ' ') {
					idx++;
				}
				if (idx == commandCharArr.length)
					break;
				beginPos = idx;
				continue;
			} else if (idx == commandCharArr.length - 1) {
				endPos = idx + 1;
				argument = command.substring(beginPos, endPos).trim();
				if (!argument.isEmpty())
					argList.add(argument);
				break;
			}
			idx++;
		}

		return argList.toArray(new String[argList.size()]);

	}
	
	/**
	 * Generic comparator for {@code TreeMap} to sort the keys in a decreasing order.
	 * 
	 * @author Mingjie Qian
	 *
	 * @param <K>
	 *        Class type to be specified by declaration.
	 */
	public static class keyDescendComparator<K extends Comparable<K>> implements Comparator<K> {         
		public int compare(K k1, K k2) {
			return k2.compareTo(k1);  
		}    
	};  
	
	/**
	 * Generic comparator for {@code TreeMap} to sort the keys in a increasing order.
	 * 
	 * @author Mingjie Qian
	 *
	 * @param <K>
	 *        Class type to be specified by declaration.  
	 */
	public static class keyAscendComparator<K extends Comparable<K>> implements Comparator<K> {         
		public int compare(K k1, K k2) {
			return k1.compareTo(k2);  
		}    
	};
	
	/**
	 * Sort a map by its keys according to a specified order. Note: the 
	 * returned map does not allow access by keys. One should use entries
	 * instead. One can cast the returned map to {@code TreeMap} but not
	 * {@code HashMap}. The input map can be any map.
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend". 
	 * @return
	 *        A sorted map by a specified order. 
	 */
	public static <K extends Comparable<K>, V> Map<K, V> sortByKeys(final Map<K, V> map, final String order) {     
		Comparator<K> keyComparator =  new Comparator<K>() {         
			public int compare(K k1, K k2) {
				int compare = 0;
				if ( order.compareTo("descend") == 0 )
					compare = k2.compareTo(k1);
				else if ( order.compareTo("ascend") == 0 )
					compare = k1.compareTo(k2);
				else {
					System.err.println("order should be either \"descend\" or \"ascend\"!");
				}
				if (compare == 0) 
					return 1;       
				else 
					return compare;   
			}
		};     
		Map<K, V> sortedByKeys = new TreeMap<K, V>(keyComparator);
		sortedByKeys.putAll(map);
		return sortedByKeys;
	}
	
	/**
	 * Sort a map by its values according to a specified order. Note: the 
	 * returned map does not allow access by keys. One should use entries
	 * instead. One can cast the returned map to {@code TreeMap} but not
	 * {@code HashMap}. The input map can be any map.
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend".
	 *        
	 * @return
	 *        A sorted map by a specified order. 
	 */
	public static <K, V extends Comparable<V>> Map<K, V> sortByValues(final Map<K, V> map, final String order) {     
		Comparator<K> valueComparator =  new Comparator<K>() {       
			public int compare(K k1, K k2) { 
				int compare = 0;
				if ( order.compareTo("descend") == 0 )
					compare = map.get(k2).compareTo(map.get(k1));
				else if ( order.compareTo("ascend") == 0 )
					compare = map.get(k1).compareTo(map.get(k2));
				else {
					System.err.println("order should be either \"descend\" or \"ascend\"!");
				}
				if (compare == 0) 
					return 1;       
				else 
					return compare;   
			}
		};     
		Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
		sortedByValues.putAll(map);
		return sortedByValues;
	}
	
	/**
	 * Sort a map by its values according to a specified order. The input map 
	 * can be any map. One can cast the returned map to {@code HashMap} but 
	 * not {@code TreeMap}.
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted which can be {@code TreeMap} or {@code HashMap}.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend".
	 * @return
	 *        A sorted map by a specified order.
	 */
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(final Map<K, V> map, String order ) {  
		
		List<Map.Entry<K, V>> list =  new LinkedList<Map.Entry<K, V>>( map.entrySet() );   
		
		if ( order.compareTo("ascend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o1.getValue()).compareTo( o2.getValue() );      
				}    
			} );  
		} else if ( order.compareTo("descend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o2.getValue()).compareTo( o1.getValue() );      
				}    
			} ); 
		} else {
			System.err.println("order should be either \"descend\" or \"ascend\"!");
		}
		
		Map<K, V> result = new LinkedHashMap<K, V>();        
		for (Map.Entry<K, V> entry : list) {    
			result.put( entry.getKey(), entry.getValue() );   
		}
		
		return result;
	}
	
	/**
	 * Sort a map by its keys according to a specified order. The input map can be 
	 * any map. One can cast the returned map to {@code HashMap} but not {@code TreeMap}. 
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted which can be {@code TreeMap} or {@code HashMap}.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend".
	 * @return
	 *        A sorted map by a specified order.
	 */
	public static <K extends Comparable<? super K>, V> Map<K, V> sortByKey(final Map<K, V> map, String order ) {  
		
		List<Map.Entry<K, V>> list =  new LinkedList<Map.Entry<K, V>>( map.entrySet() );   
		
		if ( order.compareTo("ascend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o1.getKey()).compareTo( o2.getKey() );      
				}    
			} );  
		} else if ( order.compareTo("descend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o2.getKey()).compareTo( o1.getKey() );      
				}    
			} ); 
		} else {
			System.err.println("order should be either \"descend\" or \"ascend\"!");
		}
		
		Map<K, V> result = new LinkedHashMap<K, V>();        
		for (Map.Entry<K, V> entry : list) {    
			result.put( entry.getKey(), entry.getValue() );   
		}
		
		return result;    
	}
	
	/**
	 * A generic {@code Class} that implements Comparator<Integer> which provide
	 * a override comparator function sorting a array's indices based on its values.
	 * <p>
	 * Usage:
	 * <code>
	 * <p>
	 * String[] countries = { "France", "Spain", ... };
	 * <p>
	 * ArrayIndexComparator<String> comparator = new ArrayIndexComparator<String>(countries);
	 * <p>
	 * Integer[] idxVector = comparator.createIndexArray();
	 * <p>
	 * Arrays.sort(idxVector, comparator);
	 * </code>
	 * </p>
	 * <p>
	 * Now the indexes are in appropriate order.
	 *
	 * @param <V>
	 *        Class type that extends the {@code Comparable} interface.
	 */
	public static class ArrayIndexComparator<V extends Comparable<? super V>> implements Comparator<Integer> { 
		
		private final V[] array;
		
		public ArrayIndexComparator(V[] array) {
			this.array = array;
		}
		
		public Integer[] createIndexArray() {
			Integer[] idxVector = new Integer[array.length];
			for (int i = 0; i < array.length; i++) {
				idxVector[i] = i; // Autoboxing
			}
			return idxVector;
		}
		
		@Override    
		public int compare(Integer index1, Integer index2) {
			// Autounbox from Integer to int to use as array indexes
			return array[index2].compareTo(array[index1]);
		}
	}

	/**
	 * Call System.exit(code).
	 * 
	 * @param code status code
	 */
	public static void exit(int code) {
		System.exit(code);
	}
	
}
