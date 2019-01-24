#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

const int IMAGE_WIDTH = 20;
const int IMAGE_HEIGHT = 30;

const char* INPUT_TRAINING_IMAGE_PATH = "./Input/training_chars.png";

std::vector<int> validDigits = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
std::vector<int> validChars = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };

cv::Mat matClassificationInts, matClassificationImages;

cv::Mat imagePreprocessThresholding(const cv::Mat &inputImage)
{
	cv::Mat grayscaleImage, blurredImage, thresholdedImage;

	// Convert to grayscale
	cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

	// Apply gaussian blur
	cv::Size kernelSize = cv::Size(5, 5);
	cv::GaussianBlur(grayscaleImage, blurredImage, kernelSize, 0);

	// Apply adaptive threshold
	double maxValue = 255; // Replace the pixels that pass the threshold value
	int gaussianWindowBlockSize = 11; // Size of the window used to calculate the threshold value
	double c = 2; // Value to subtract from the weighted mean of the window used
	cv::adaptiveThreshold(blurredImage, 
		thresholdedImage,
		maxValue, 
		cv::ADAPTIVE_THRESH_GAUSSIAN_C,
		cv::THRESH_BINARY_INV,
		gaussianWindowBlockSize,
		c);

	// Display processed image
	cv::imshow("thresholdedInputImage", thresholdedImage);

	return thresholdedImage;
}

void writeClassificationToFile(const std::string &fileName, const std::string &sectionName, const cv::Mat &classificationData) 
{
	// Open the classification file
	cv::FileStorage fs(fileName, cv::FileStorage::WRITE);

	if (fs.isOpened() == false)
	{
		std::cout << "Failed to open the classification file: " << fileName << "\n";
		return;
	}

	// Write classification data to file
	fs << sectionName << classificationData;
	fs.release();
}

int main(int argc [[maybe_unused]], char **argv [[maybe_unused]])
{
	// Input image
	cv::Mat inputTrainingNumbersImage;

	// Load training numbers image
	inputTrainingNumbersImage = cv::imread(INPUT_TRAINING_IMAGE_PATH);
	if (inputTrainingNumbersImage.empty()) 
	{
		std::cout << "Failed to read input training image: " << INPUT_TRAINING_IMAGE_PATH << "\n";
		return -1;
	}

	// Process the input image to get the thresholded image
	cv::Mat thresholdedImage = imagePreprocessThresholding(inputTrainingNumbersImage);

	// Make a copy of the thresholded image to find the contours
	cv::Mat thresholdedImageCopy = thresholdedImage.clone();

	// Find contours
	// Use binary images
	// The subject to be found should be white and background should be black
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(thresholdedImageCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Iterate through each of the contours found
	for (auto &contour : contours)
	{
		const double minContourArea = 100.0f;
		// Check if the detected contour is valid
		if (cv::contourArea(contour) > minContourArea)
		{
			// Convert the list of points in the contour to a cv::Rect
			cv::Rect contourRectangle = cv::boundingRect(contour);

			// Draw a rectangle around the detected contour
			cv::rectangle(inputTrainingNumbersImage, contourRectangle, cv::Scalar(0.0f, 0.0f, 255.0f), 2);

			// Extract and resize the ROI area
			cv::Mat originalROI = thresholdedImage(contourRectangle);
			cv::Mat threasholdImageROI;
			cv::resize(originalROI, threasholdImageROI, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

			// Display the extracted ROI and the resized ROI
			cv::imshow("originalROI", originalROI);
			cv::imshow("resizedROI", threasholdImageROI);
			// Display the input image together with the detected contours
			cv::imshow("originalInputImage", inputTrainingNumbersImage);

			// Read input character/digit
			int inputChar = cv::waitKey(0);

			bool isDigit = std::find(validDigits.begin(), validDigits.end(), inputChar) != validDigits.end();
			bool isChar = std::find(validChars.begin(), validChars.end(), inputChar) != validChars.end();
			if (isDigit || isChar) 
			{
				// Append the classification integers 
				matClassificationInts.push_back(inputChar);

				// Append the classification images
				cv::Mat floatImage;
				threasholdImageROI.convertTo(floatImage, CV_32FC1);
				cv::Mat flattenedImage = floatImage.reshape(1, 1);
				matClassificationImages.push_back(flattenedImage);
			}
		}
	}

	// Write classification data to classification files
	writeClassificationToFile("./Output/classifications.xml", "classifications", matClassificationInts);
	writeClassificationToFile("./Output/images.xml", "images", matClassificationImages);

	return 0;
}