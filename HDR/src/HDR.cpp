// HDR.cpp : Defines the entry point for the application.
//

#include "HDR.h"
#include "Tonemap.h"

using namespace std;
using namespace cv;

// mentioned in lecture 3 at start
#define THRESHOLD 10e-4f

float robWeight(int bit8)
{
	float q = 255 / 4.f;
	float value = bit8 / q - 2.f;
	value = exp(-value * value);
	return value;
}

struct PixelIndex
{
	int row;
	int col;
	int channel;
};



int main()
{
	vector<float> exposure;
	vector<Mat> images;
	string topLevelDirectory = "C:/Users/benja/Development/CapturingReality/HDR/data/";
	string hdrgenFile = topLevelDirectory + "max.hdrgen";
	ifstream file(hdrgenFile);
	if (file.is_open())
	{
		string line;
		while (getline(file, line))
		{
			cout << line << '\n';
			int whitespace = line.find(' ');
			int length = line.length();
			string imageFilename = line.substr(0, whitespace);
			string exposureValue = line.substr(whitespace + 1, length);

			// read file
			cout << "Filename: " << imageFilename << endl;
			Mat image = imread(topLevelDirectory + imageFilename);
			images.push_back(image);

			cout << "Exposure: " << exposureValue << endl;
			exposure.push_back(1.f/stof(exposureValue));

		}
		file.close();
	}

	Mat HDR = Mat::zeros(images[0].size(), CV_32FC3);

	int maxEpochs = 5;
	const int rows = images[0].rows;
	const int cols = images[0].cols;
	vector<vector<float>> estimatedIrradiance(rows, vector<float>(cols));

	// create sample image with 3 channels
	int nImages = images.size();
	int nChannels = images[0].channels();
	constexpr int nValues = 256;
	vector<vector<float>> cameraResponse(nChannels, vector<float>(nValues));

	// create a vector of indices for each pixel value
	vector<vector<PixelIndex>> indices(nValues, vector<PixelIndex>());
	for (int channel = 0; channel < nChannels; channel++)
	{
		// init linear response curve
		for (int value = 0; value < nValues; value++)
			cameraResponse[channel][value] = value / 128.f;

		// estimate irradiance \tilde x_j eq(8)
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				estimatedIrradiance[r][c] = 0.f;
				float weightedSum = 0.f;
				// i = image;
				for (int p = 0; p < nImages; p++)
				{
					// j = pixel;
					int pixelValue = images[p].at<Vec3b>(r, c)(channel);

					estimatedIrradiance[r][c] += robWeight(pixelValue) * exposure[p] * cameraResponse[channel][pixelValue];
					weightedSum += robWeight(pixelValue) * exposure[p] * exposure[p];

					// find indices
					for (int v = 0; v < nValues; v++)
					{
						if (pixelValue == v)
						{
							PixelIndex index;
							index.row = r;
							index.col = c;
							index.channel = channel;
							indices[v].push_back(index);
						}
					}
				}
				estimatedIrradiance[r][c] /= weightedSum;
			}
		}
	}

	// estimate response function
	vector<float> responseFunction(nValues);
	for (int epoch = 0; epoch < maxEpochs; epoch++) 
	{
		for (int v = 0; v < nValues; v++)
		{
			float response = 0.f;
			for (int i = 0; i < nImages; i++)
			{
				float sum = 0.f;
				int size = indices[v].size();
				for (auto idx: indices[v])
				{
					PixelIndex index = idx;
					sum += exposure[i] * estimatedIrradiance[index.row][index.col];
				}
				response += (1.f / size) * sum;
			}
			responseFunction[v] += response;
		}
	}

	// normalize
	float normFactor = responseFunction[127];
	for (int i = 0; i < responseFunction.size(); i++)
		responseFunction[i] /= normFactor;

	// write response function to file
	ofstream responseFunctionFile;
	responseFunctionFile.open(topLevelDirectory + "responseCurve.txt");
	for (int j = 0; j < nValues; j++)
	{
		responseFunctionFile << responseFunction[j] << '\n';
	}
	responseFunctionFile.close();

	// recover HDR image from response function
	for (int channel = 0; channel < nChannels; channel++)
	{
		for (int r = 0; r < rows; r++)
		{
			for (int c = 0; c < cols; c++)
			{
				float sum = 0.f;
				for (int i = 0; i < nImages; i++)
				{
					int pixelValue = images[i].at<Vec3b>(r, c)(channel);
					sum += robWeight(pixelValue) * (responseFunction[pixelValue] - log(exposure[i]));
				}
				HDR.at<Vec3f>(r, c)(channel) = exp(sum) / 127.f;
			}
		}
	}

	imwrite(topLevelDirectory + "hdr_out.png", HDR.clone());
	cout << "Wrote recoved HDR image to png" << endl;

	// write to OpenEXR
	// works only in debug mode even though a warning is displayed
	// launching in release mode results in a crash
	// known OpenCV bug
//#define OPENCV_IO_ENABLE_OPENEXR 1
//	imwrite(topLevelDirectory + "hdr_out.exr", HDR);
//	cout << "Wrote recoved HDR image to exr" << endl;

	// tonemapping
	Mat tonemapMat = Mat::zeros(HDR.size(), CV_8UC3);
	tonemap(HDR, tonemapMat);

	// write tonemapped image
	imwrite(topLevelDirectory + "tonemapped.jpg", tonemapMat);
	cout << "Wrote tonemapped image to png" << endl;
	return EXIT_SUCCESS;
}
