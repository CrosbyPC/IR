package ir.vsr;

import java.io.*;
import java.util.*;
import java.lang.*;

import ir.utilities.*;
import ir.classifiers.*;

/**
 * An inverted index for vector space information retrieval. Contains
 * methods for creating an inverted index from a set of documents
 * and retrieving ranked matches to queries using TF/IDF weighting
 * and cosine similarity, but also weights documents based on the
 * percentage of query terms that appear in each of them.
 * 
 * @author Crosby Cook
 */
 
public class InvertedIndexWithQueryCount extends InvertedIndex {
	
	/**
	 * Create an inverted index of the documents in a directory.
	 * @param dirFile	The directory of files to index.
	 * @param docType	The type of documents to index (See docType in DocumentIterator)
	 * @param stem		Whether the tokens should be stemmed with the Porter stemmer.
	 * @param feedback	Whether relevance feedback should be used.
	 */
	public InvertedIndexWithQueryCount(File dirFile, short docType, boolean stem, boolean feedback) {
		super(dirFile, docType, stem, feedback);
	}
	
	/**
	 * Create an inverted index of the documents in a List of Example objects
	 * of documents for text categorization.
	 * 
	 * @param examples	A list containing the Example objects for text categorization
	 *					to index.
	 */
	public InvertedIndexWithQueryCount(List<Example> examples) {
		super(examples);
	}	
	
	/**
	 * Performed ranked retrieval on this input query Document vector.
	 * Weight the ranking of each document by the percentage of query
	 * terms that appear in that document.
	 */
	public Retrieval[] retrieve(HashMapVector vector) {
	    // Create a hashtable to store the retrieved documents.  Keys
		// are docRefs and values are DoubleValues which indicate the
		// partial score accumulated for this document so far.
		// As each token in the query is processed, each document
		// it indexes is added to this hashtable and its retrieval
		// score (similarity to the query) is appropriately updated.
		Map<DocumentReference, DoubleValue> retrievalHash =	new HashMap<DocumentReference, DoubleValue>();
		
		// Create a second hashtable to store the number of individual
		// query terms that appear in those documents. Keys are
		// docRefs and values are DoubleValues which count the number
		// of individual query terms that appear in the document.
		Map<DocumentReference, DoubleValue> retrievalMatchHash = new HashMap<DocumentReference, DoubleValue>();
		
		// Initialize a variable to store the length of the query vector.
		double queryLength = 0.0;
		// Initialize a variable to store the total number of tokens in
		// the query vector. Used to calculate the percentage of terms
		// from the query that are present in any given document.
		double queryTokenCount = 0.0;
		
		// Iterate through each token in the query input Document
		for (Map.Entry<String, Weight> entry : vector.entrySet()) {
			// Take each token
			String token = entry.getKey();
			// Count how many times it appears in the query
			double count = entry.getValue().getValue();
			// Determine the score added to the similarity of each document
			// indexed under this token and update the length of the query
			// vector with the square of the weight for this token.
			queryLength += incorporateToken(token, count, retrievalHash, retrievalMatchHash);
			
			// Update the total query token count
			queryTokenCount += 1.0;
		}
		
		
		// Now, finalize the length of the query vector by taking the square
		// root of the final sum of the squares of its token weights.
		queryLength = Math.sqrt(queryLength);
		
		// Make an array to store the final ranked Retrievals.
		Retrieval[] retrievals = new Retrieval[retrievalHash.size()];
		
		// Iterate through each of the retrieved documents stored in the
		// final retrievalHash.
		int retrievalCount = 0;
		for (Map.Entry<DocumentReference, DoubleValue> entry : retrievalHash.entrySet()) {
			DocumentReference docRef = entry.getKey();
			double score = entry.getValue().value;
			double hits = retrievalMatchHash.get(docRef).value;
			
			// Get the document's hit rate, to weight in favor of documents that contain more of the query's terms.
			double hitRate = hits/queryTokenCount;
			
			retrievals[retrievalCount++] = getRetrieval(queryLength, docRef, score, hitRate);
		}
		// Sort the retrievals to produce a final ranked list using the
		// Comparator for retrievals that produces a best to worst ordering.
		Arrays.sort(retrievals);
		return retrievals;
	}
	
	/**
	 * Calculate the final score for a retrieval and return a Retrieval object representing
	 * the retrieval with its final score.
	 * 
	 * @param queryLength	The length of the query vector, incorporated into the final score
	 * @param docRef		The document reference for the document concerned
	 * @param score			The partially computed score
	 * @param hitRate		The percentage of query terms present in the document
	 * @return				The retrieval object for the document described by docRef
	 * 					and score under the query with length queryLength
	 */
	protected Retrieval getRetrieval(double queryLength, DocumentReference docRef, double score, double hitRate) {
		// Normalize the score for the lengths of the two document vectors
		score = score / (queryLength * docRef.length);
		// Weight based on hit rate
		score += hitRate;
		
		// Add a Retrieval for this document to the result array
		return new Retrieval(docRef, score);
	}

	/**
	 * Retrieve the documents indexed by this token in the inverted index,
	 * add them to the retrievalHash if needed, and update their running
	 * total scores and hit counts.
	 * 
	 * @param token			The token in the query to incorporate.
	 * @param count			The count of this token in the query.
	 * @param retrievalHash	The hash table of retrieved DocumentReferences
	 * 						and their corresponding scores.
	 * @param retrievalMatchHash	The hash table of retrieved DocumentReferences
	 *								and their corresponding hit counts.
	 * @return	The square of the weight of this token in the query vector
	 * 			for use in calculating the length of the query vector.
	 */
	public double incorporateToken(String token, double count,
			Map<DocumentReference, DoubleValue> retrievalHash,
			Map<DocumentReference, DoubleValue> retrievalMatchHash) {
		TokenInfo tokenInfo = tokenHash.get(token);
		// If token is not in the index, it adds nothing and its squared weight is 0.
		if (tokenInfo == null) {
			return 0.0;
		}
		// The weight of a token in the query is its IDF factor
		// times the number of times it occurs in the query.
		double weight = tokenInfo.idf * count;
		
		// For each document occurrence indexed for this token...
		for (TokenOccurrence occ : tokenInfo.occList) {
			// Get the current score for this document in the retrievalHash.
			DoubleValue val = retrievalHash.get(occ.docRef);
			if (val == null) {
				// If this is a new retrieved document, create an initial score
				// for it and store in the retrievalHash
				val = new DoubleValue(0.0);
				retrievalHash.put(occ.docRef, val);
			}
			// Do the same for hitcount in the retrievalMatchHash
			DoubleValue hitCount = retrievalMatchHash.get(occ.docRef);
			if (hitCount == null) {
				// A newly retrieved document will need its hitCount initialized
				hitCount = new DoubleValue(0.0);
				retrievalMatchHash.put(occ.docRef, hitCount);
			}
			
			// Update the score for this document by adding the product
			// of the weight of this token in the query and its weight
			// in the retrieved document (IDF * occurrence count)
			val.value = val.value + weight * tokenInfo.idf * occ.count;
			// Increment the hit count for this document.
			hitCount.value += 1.0;
		}
		
		//Return the square of the weight of this token in the query
		return weight * weight;
	}
	
	/**
	 * Index a directory of files and then interactively accept retrieval queries.
	 * Command format: "InvertedIndexWithQueryCount [OPTION]* [DIR]"
	 * where Dir is the name of the directory whose files should be indexed,
	 * and OPTIONs can be:
	 * "-html" to specify HTML files whose HTML tags should be removed.
	 * "-stem" to specify tokens should be stemmed with the Porter stemmer.
	 * "-feedback" to allow relevance feedback from the user.
	 */
	public static void main(String[] args) {
		// Parse the arguments into a directory name and optional flag.
		
		String dirName = args[args.length - 1];
		short docType = DocumentIterator.TYPE_TEXT;
		boolean stem = false;
		boolean feedback = false;
		for (int i = 0; i < args.length - 1; i++) {
			String flag = args[i];
			if (flag.equals("-html")) {
				// Create HTMLFileDocuments to filter HTML tags
				docType = DocumentIterator.TYPE_HTML;
			}
			else if (flag.equals("-stem")) {
				// Stem tokens with the Porter stemmer
				stem = true;
			}
			else if (flag.equals("-feedback")) {
				// Use relevance feedback
				feedback = true;
			}
			else {
				throw new IllegalArgumentException("Unknown flag: " + flag);
			}
		}
		
		// Create an inverted index with query count
		// for the files in the given directory.
		InvertedIndexWithQueryCount index = 
			new InvertedIndexWithQueryCount(new File(dirName), docType, stem, feedback);
		// Interactively process queries to this index.
		index.processQueries();
	}
}
