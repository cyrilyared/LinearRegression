import java.util.Scanner;

/**
 * Uses linear regression (batch gradient descent) to model the relationship between the features and the target variable.
 * 
 * @author cyrilyared
 *
 */
public class LinearRegression {
	
	public static void main(String[] args) {
		Scanner userInput = new Scanner(System.in);
		
		int featureNum = getInteger(userInput, "Enter the number of parameters/features: ", 1, Integer.MAX_VALUE);
		int sizeTrainingSet = getInteger(userInput, "Enter the size of the training set: ", 1, Integer.MAX_VALUE);
		double alpha = getDouble(userInput, "Enter the learning rate: ", 0, Double.MAX_VALUE);
		double convergenceError = getDouble(userInput, "Enter the precision required for convergence: ", 0, Double.MAX_VALUE);
		int maxEpochs = getInteger(userInput, "Enter the maximum number of epochs: ", 1, Integer.MAX_VALUE);
		
		RowData[] inputDataset = new RowData[featureNum];		
		for(int i = 0; i < featureNum; i++) {
			inputDataset[i] = new RowData("Enter the values for parameter " + String.valueOf(i+1) + " in a comma-separated format: ", sizeTrainingSet);
		}
		RowData targetDataset = new RowData("Enter the values for the target variable in a comma-separated format: ", sizeTrainingSet);
		
		userInput.close();
		
		calculateLinearRegression(featureNum, sizeTrainingSet, alpha, convergenceError, maxEpochs, inputDataset, targetDataset);
	}
	
	/**
	 * Uses linear regression to find relationship.
	 * Prints linear model found.
	 * 
	 * @param featureNum integer number of features
	 * @param sizeTrainingSet integer size of training set
	 * @param alpha double learning rate
	 * @param convergenceError double acceptable error for convergence
	 * @param maxEpochs integer maximum number of epochs
	 * @param inputDataset features array of objects of RowData for input variables
	 * @param targetDataset object of RowData containing output data
	 */
	public static void calculateLinearRegression(int featureNum, int sizeTrainingSet, double alpha, double convergenceError, int maxEpochs, RowData inputDataset[], RowData targetDataset) {
		double[] weights = new double[featureNum];
		double bias = 0;
		int epoch;
		
		for(epoch = 0; epoch < maxEpochs; epoch++) {
			double currentError = 0;
			bias = bias - alpha*getBiasGradient(sizeTrainingSet, weights, inputDataset, bias, targetDataset);
			
			for(int i = 0; i < featureNum; i++) {
				double gradient = getGradient(i, sizeTrainingSet, weights, inputDataset, bias, targetDataset);
				weights[i] = weights[i] - alpha*gradient;
				currentError = findMax(currentError, findAbs(alpha*gradient));
			}
			if(currentError < findAbs(convergenceError)) {
				break;
			}
		}
		System.out.println("After " + String.valueOf(epoch) + " epochs, the following relationship was found:");
		printWeights(weights, bias);
	}

	/**
	 * Returns gradient for weights.
	 * 
	 * @param index integer value of weight to update
	 * @param sizeTrainingSet integer size of the training set
	 * @param weights double array of current weights
	 * @param inputDataset features array of objects of RowData for input variables
	 * @param bias double for bias
	 * @param targetDataset object of RowData containing output data
	 * @return gradient double
	 */
	public static double getGradient(int index, int sizeTrainingSet, double weights[], RowData inputDataset[], double bias, RowData targetDataset) {
		double gradient = 0;
		for(int i = 0; i < sizeTrainingSet; i++) {
			gradient += (calculateHypothesis(weights, inputDataset, bias, i)-targetDataset.data[i])*inputDataset[index].data[i];
		}
		return gradient;
	}
	
	/**
	 * Returns gradient for bias.
	 * 
	 * @param sizeTrainingSet integer size of the training set
	 * @param weights double array of current weights, this will be updated
	 * @param inputDataset features array of objects of RowData for input variables
	 * @param bias double for bias
	 * @param targetDataset object of RowData containing output data
	 * @return gradient double
	 */
	public static double getBiasGradient(int sizeTrainingSet, double weights[], RowData inputDataset[], double bias, RowData targetDataset) {
		double gradient = 0;
		for(int i = 0; i < sizeTrainingSet; i++) {
			gradient += (calculateHypothesis(weights, inputDataset, bias, i)-targetDataset.data[i]);
		}
		return gradient;
	}

	/**
	 * Returns the value of the hypothesis.
	 * 
	 * @param weights array of doubles for current weights
	 * @param features array of objects of RowData for input variables
	 * @param bias double for bias
	 * @param index integer for current training set evaluated
	 * @return double hypothesis
	 */
	public static double calculateHypothesis(double weights[], RowData features[], double bias, int index) {
		double sum = 0;
		for(int i = 0; i < weights.length; i++) {
			sum += weights[i]*features[i].data[index];
		}
		return sum + bias;
	}
	
	/**
	 * Prints the model found to the console.
	 * 
	 * @param weights array of doubles for current weights
	 * @param bias double for bias
	 */
	public static void printWeights(double weights[], double bias) {
		String output = String.valueOf(bias);
		for(int i = 0; i < weights.length; i++) {
			output = output.concat(" + " + String.valueOf(weights[i]) + " x" + String.valueOf(i+1));
		}
		System.out.println(output);
	}
	
	/**
	 * Prompts the user for a positive integer between a specified minimum and maximum.
	 * Returns the integer if formatted correctly.
	 * 
	 * @param userInput scanner
	 * @param prompt string prompt
	 * @param min minimum value of integer
	 * @param max maximum value of integer
	 * @return formatted integer
	 */
	public static int getInteger(Scanner userInput, String prompt, int min, int max) {
		int result = -1;
		
		while(result == -1) {
			System.out.println(prompt);
			String input = userInput.nextLine();
			try {
				result = Integer.parseInt(input);
			} catch(NumberFormatException e) {
				result = -1;
			}
		
			if(result >= min && result <= max) {
				break;
			} else {
				result = -1;
			}
		}
		return result;
	}

	/**
	 * Prompts the user for a positive double between a specified minimum and maximum.
	 * Returns the double if formatted correctly.
	 * 
	 * @param userInput scanner
	 * @param prompt string prompt
	 * @param min minimum value of double
	 * @param max maximum value of double
	 * @return formatted double
	 */
	public static double getDouble(Scanner userInput, String prompt, double min, double max) {
		double result = -1.0;

		while(result == -1.0) {
			System.out.println(prompt);
			String input = userInput.nextLine();
			try {
				result = Double.parseDouble(input);
			} catch(NumberFormatException e) {
				result = -1.0;
			}

			if(result > min && result <= max) {
				break;
			} else {
				result = -1.0;
			}
		}
		return result;
	}
	
	/**
	 * Returns maximum of two doubles.
	 * 
	 * @param val1
	 * @param val2
	 * @return maximum
	 */
	public static double findMax(double val1, double val2) {
		if(val1 > val2) {
			return val1;
		} else {
			return val2;
		}
	}
	
	/**
	 * Returns absolute value of double input.
	 * 
	 * @param val double input
	 * @return double absolute value of input
	 */
	public static double findAbs(double val) {
		if(val < 0) {
			return val * -1;
		} else {
			return val;
		}
	}
}